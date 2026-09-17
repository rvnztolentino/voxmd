"""Stage 6: watch a folder, and turn what lands in it into notes.

Four things about this module are constraints rather than design preferences,
and none of them should be "simplified" later:

* **It never auto-starts (C1).** There is no daemonizing, no double fork, no
  ``setsid``, no LaunchAgent, no plist, and nothing that writes one. ``voxmd
  watch`` runs in the foreground, in the terminal you started it in, and stops
  when you press Ctrl-C or close that terminal. Nothing survives a reboot.
* **It is event-driven, never polling (C4).** watchdog silently falls back to a
  ``PollingObserver`` when the native backend is unavailable, which would burn
  CPU forever without ever saying so. Startup therefore *refuses to run* on a
  polling observer instead of trusting that the right one was picked. The one
  place a poll exists is the per-file stability gate, which runs only while a
  file is actually arriving.
* **Every wake and every file is logged (C2).** Including the ones that turn
  out to be uninteresting: a trace with gaps in it isn't a trace.
* **One memo at a time.** whisper already uses every performance core, so a
  second concurrent transcription makes both slower. Arrivals queue up and are
  processed in order by the main thread, which also means Ctrl-C lands where
  you'd expect it to.
"""

from __future__ import annotations

import contextlib
import os
import queue
import signal
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import safe
from .config import Config, WatchConfig
from .errors import ConfigError, DependencyError, InputError, PartialFailure, VoxmdError
from .logging_setup import EventLog
from .pipeline import (
    ProcessResult,
    prepare_state_dir,
    process,
    resolve_archive_dir,
    resolve_notes_dir,
)
from .status import WATCH_LOCK_NAME, WatchState, state_path, write_state
from .transcribe import AUDIO_SUFFIXES

WATCH_LOCK_TIMEOUT_S = 0.0
"""A second watcher on the same state directory fails at once, not eventually."""

STOP_SIGNALS = ("SIGTERM", "SIGHUP")
"""SIGHUP is the one that arrives when you close the terminal — the C1 case."""


class Stopped(BaseException):  # noqa: N818 - not an error; it is how a watcher ends
    """Raised in the main thread by SIGTERM or SIGHUP, exactly like Ctrl-C.

    Raising rather than queueing a message is what makes the handler safe: it
    takes no lock, touches no file, and unblocks a waiting ``queue.get`` the
    same instant, so closing the terminal stops the watcher then and there
    instead of after the memo in flight.
    """

    def __init__(self, signal_name: str) -> None:
        super().__init__(signal_name)
        self.signal_name = signal_name


@dataclass(frozen=True)
class Arrival:
    """One file the observer told us about."""

    path: Path
    trigger: str
    """``created``, ``moved`` (a sync client's temp file being renamed), or ``scan``."""


def watch(
    settings: Config,
    *,
    log: EventLog,
    run: Callable[..., ProcessResult] | None = None,
    observer: Any = None,
) -> Watcher:
    """Watch ``watch.dir`` until stopped, and return the watcher for its counts.

    Blocks. Everything it can check cheaply — the folder, the vault, the layout,
    the observer backend — is checked before the first event is accepted, so a
    misconfigured watcher fails in the first second rather than on the first
    memo of the day.
    """
    watch_dir = resolve_watch_dir(settings.watch)
    archive_dir = resolve_archive_dir(settings.archive)
    state_dir = prepare_state_dir(settings.state.dir)
    check_layout(watch_dir, archive_dir, state_dir)
    # Fail now if the vault is wrong, rather than after transcribing a memo.
    resolve_notes_dir(settings.vault)

    if observer is None:
        observer = make_observer()
    kind = check_observer(observer)

    watcher = Watcher(settings, log=log, run=run)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_only_one_watcher(state_dir))
        # Announced before the observer starts, so no event can arrive before
        # there is a state file for it to be counted in.
        watcher.begin(watch_dir, observer=kind)
        try:
            observer.schedule(ArrivalHandler(watcher), str(watch_dir), recursive=False)
            observer.start()
            stack.callback(_stop_observer, observer, log)
            stack.enter_context(_stop_on_signal())
            watcher.scan(watch_dir)
            watcher.serve()
        except KeyboardInterrupt:
            # Ctrl-C is how a watcher is stopped, not a failure.
            watcher.finish("interrupt")
            return watcher
        except Stopped as stop:
            # Closing the terminal (SIGHUP) or `kill` (SIGTERM). Also not a failure.
            watcher.finish(stop.signal_name)
            return watcher
        except BaseException as exc:
            log.event("watch.crash", error=type(exc).__name__, detail=_reason(exc))
            watcher.finish("crash")
            raise
        watcher.finish("stopped")
    return watcher


class Watcher:
    """The queue, the counters, and the one worker that drains them.

    The observer's threads only ever call :meth:`offer`. Everything that costs
    real time happens on the main thread in :meth:`serve`.
    """

    def __init__(
        self,
        settings: Config,
        *,
        log: EventLog,
        run: Callable[..., ProcessResult] | None = None,
    ) -> None:
        self.settings = settings
        self.log = log
        self.wakes = 0
        self.processed = 0
        self.failed = 0
        self.skipped = 0
        self._run = run or process
        self._queue: queue.Queue[Arrival | None] = queue.Queue()
        self._pending: set[Path] = set()
        self._guard = threading.Lock()
        self._stopping = threading.Event()
        self._state: WatchState | None = None

    def begin(self, watch_dir: Path, *, observer: str) -> None:
        """Record that a watcher is running here, and say so in the log."""
        self._state = WatchState(
            pid=os.getpid(),
            started_at=datetime.now().astimezone(),
            watch_dir=str(watch_dir),
            log=str(self.log.path),
            observer=observer,
        )
        self.log.event(
            "watch.start",
            pid=os.getpid(),
            dir=str(watch_dir),
            observer=observer,
            log=str(self.log.path),
        )
        self._save()

    def scan(self, watch_dir: Path) -> int:
        """Queue whatever was already sitting in the folder when we started.

        Memos that synced while the watcher was off would otherwise wait for an
        event that already happened. The ledger stops anything being done twice.
        """
        try:
            found = sorted(p for p in watch_dir.iterdir() if p.is_file() and interesting(p)[0])
        except OSError as exc:
            self.log.event("scan.failed", error=exc.strerror or str(exc))
            return 0
        self.log.event("scan", found=len(found))
        for path in found:
            self.offer(path, "scan")
        return len(found)

    def offer(self, path: Path, trigger: str) -> bool:
        """Accept one arrival. Called from the observer's thread; never blocks it."""
        wanted, why = interesting(path)
        with self._guard:
            self.wakes += 1
            if self._state is not None:
                self._state = self._state.model_copy(
                    update={
                        "last_wake": datetime.now().astimezone(),
                        "last_wake_trigger": trigger,
                        "wakes": self.wakes,
                    }
                )
            duplicate = path in self._pending
            if wanted and not duplicate:
                self._pending.add(path)
        self.log.event("wake", trigger=trigger, file=path.name)

        if not wanted:
            self.log.event("ignored", file=path.name, reason=why)
            return False
        if duplicate:
            self.log.event("ignored", file=path.name, reason="already queued")
            return False
        self._queue.put(Arrival(path, trigger))
        return True

    def serve(self) -> None:
        """Process arrivals until stopped. Blocks with no timer, so idle costs nothing."""
        while True:
            item = self._queue.get()
            if item is None or self._stopping.is_set():
                return
            try:
                self.handle(item)
            finally:
                with self._guard:
                    self._pending.discard(item.path)

    def stop(self) -> None:
        """Ask ``serve`` to return. Safe to call from a signal handler."""
        self._stopping.set()
        self._queue.put(None)

    def handle(self, arrival: Arrival) -> None:
        """Wait for one file to finish arriving, then run the pipeline on it."""
        path, cfg = arrival.path, self.settings.watch
        started = time.monotonic()
        try:
            info = settle(
                path,
                stable_seconds=cfg.stable_seconds,
                poll_seconds=cfg.poll_seconds,
                timeout_s=cfg.settle_timeout_s,
            )
        except InputError as exc:
            self.log.event("file.skipped", file=path.name, reason=str(exc))
            self._count(skipped=1)
            return
        self.log.event(
            "file.ready",
            file=path.name,
            bytes=info.st_size,
            waited=f"{time.monotonic() - started:.1f}s",
        )

        try:
            result = self._run(
                path,
                settings=self.settings,
                lock_timeout_s=cfg.lock_timeout_s,
            )
        except PartialFailure as exc:
            # The note exists; something after it didn't. Both facts get logged.
            self.log.event("file.partial", file=path.name, detail=str(exc))
            self._count(failed=1)
            return
        except (VoxmdError, OSError) as exc:
            self.log.event(
                "file.failed", file=path.name, error=type(exc).__name__, reason=_reason(exc)
            )
            self._count(failed=1)
            return

        if result.skipped:
            self.log.event("file.skipped", file=path.name, reason="already processed")
            self._count(skipped=1)
            return
        for problem in result.problems:
            self.log.event("file.warning", file=path.name, detail=problem)
        self.log.event(
            "file.done",
            file=path.name,
            note=str(result.note),
            transcript=str(result.transcript) if result.transcript else None,
            archived=str(result.archived) if result.archived else None,
            seconds=f"{result.seconds:.1f}",
        )
        self._count(processed=1)

    def finish(self, reason: str) -> None:
        """Log the shutdown and mark the state file stopped, so status won't guess."""
        with self._guard:
            if self._state is not None:
                self._state = self._state.model_copy(
                    update={"stopped_at": datetime.now().astimezone()}
                )
            self._write_state()
        self.log.event(
            "watch.stop",
            reason=reason,
            wakes=self.wakes,
            processed=self.processed,
            skipped=self.skipped,
            failed=self.failed,
        )

    def _count(self, *, processed: int = 0, skipped: int = 0, failed: int = 0) -> None:
        with self._guard:
            self.processed += processed
            self.skipped += skipped
            self.failed += failed
            if self._state is not None:
                self._state = self._state.model_copy(
                    update={"processed": self.processed, "failed": self.failed}
                )
            self._write_state()

    def _save(self) -> None:
        with self._guard:
            self._write_state()

    def _write_state(self) -> None:
        """Called with ``_guard`` held. A state file voxmd can't write is not fatal:
        it only costs ``voxmd status`` its answer, and the log still has everything."""
        if self._state is None:
            return
        try:
            write_state(state_path(self.settings.state.dir), self._state)
        except VoxmdError as exc:
            self.log.event("state.failed", detail=str(exc))


class ArrivalHandler:
    """watchdog event handler. Deliberately not a subclass of anything heavy.

    ``on_moved`` matters as much as ``on_created``: iCloud and Dropbox write to a
    temp name and rename it into place, so a watcher listening only for
    creations would miss most real memos.
    """

    def __init__(self, watcher: Watcher) -> None:
        self._watcher = watcher

    def dispatch(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        kind = getattr(event, "event_type", "")
        if kind == "created":
            self._watcher.offer(Path(os.fsdecode(event.src_path)), "created")
        elif kind == "moved":
            self._watcher.offer(Path(os.fsdecode(event.dest_path)), "moved")


def make_observer() -> Any:
    """watchdog's observer for this platform. Imported here, not at module scope."""
    from watchdog.observers import Observer

    return Observer()


def check_observer(observer: Any) -> str:
    """Return the observer's name, refusing a polling one (C4).

    watchdog picks ``PollingObserver`` when the native backend is unavailable,
    and it does so silently. A polling watcher would stat the whole folder on a
    timer forever — exactly the thing this tool promises not to do — so it is a
    startup failure rather than a warning nobody reads.
    """
    from watchdog.observers.polling import PollingObserver, PollingObserverVFS

    name = type(observer).__name__
    if isinstance(observer, PollingObserver | PollingObserverVFS):
        raise DependencyError(
            f"watchdog fell back to {name}, which checks the folder on a timer instead of "
            "waiting for filesystem events.\n"
            "voxmd refuses to run that way: an idle watcher must cost nothing. "
            "Reinstall watchdog for this platform (uv sync), and check that the folder is "
            "on a local disk rather than a network mount."
        )
    return name


def resolve_watch_dir(watch: WatchConfig) -> Path:
    """The folder to watch, resolved. Creates nothing."""
    if watch.dir is None:
        raise ConfigError(
            "No watch folder configured.\n"
            "Set watch.dir in your config, or pass --dir /path/to/your/inbox."
        )
    try:
        resolved = watch.dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"Watch folder not found: {watch.dir}\n"
            "voxmd never creates it; check watch.dir, or that your sync client has run."
        ) from exc
    if not resolved.is_dir():
        raise ConfigError(f"watch.dir is not a directory: {resolved}")
    if not os.access(resolved, os.R_OK | os.X_OK):
        raise ConfigError(f"Watch folder is not readable: {resolved}")
    return resolved


def check_layout(watch_dir: Path, archive_dir: Path | None, state_dir: Path) -> None:
    """Refuse a layout where voxmd's own writes would wake it up again."""
    if archive_dir is not None and _within(archive_dir, watch_dir):
        raise ConfigError(
            f"archive.dir ({archive_dir}) is inside the watch folder ({watch_dir}).\n"
            "Every archived recording would land back under the watcher and wake it again. "
            "Put the archive somewhere outside."
        )
    if _within(state_dir, watch_dir):
        raise ConfigError(
            f"state.dir ({state_dir}) is inside the watch folder ({watch_dir}).\n"
            "voxmd's ledger and log would wake the watcher. Put state.dir somewhere outside."
        )


def interesting(path: Path) -> tuple[bool, str]:
    """Whether a file is worth queueing, and if not, why not.

    Hidden files cover the noise sync clients make: ``.memo.m4a.icloud``
    placeholders, ``.dropbox.attr``, ``.DS_Store``, and the partial files
    written under a dot-name before being renamed into place.
    """
    if path.name.startswith("."):
        return False, "hidden file"
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        return False, f"not audio ({path.suffix or 'no extension'})"
    return True, ""


def settle(
    path: Path,
    *,
    stable_seconds: float,
    poll_seconds: float,
    timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> os.stat_result:
    """Wait until the file stops changing, then return its final stat.

    A sync client creates the file and then fills it, so an event means "a file
    is arriving", not "a file has arrived". Transcribing a half-written memo
    would produce a truncated note that looks perfectly fine.

    This is the only poll in the watcher, and it runs only while a file is
    actually being written — never while the watcher is idle (C4).
    """
    deadline = clock() + timeout_s
    mark: tuple[int, int] | None = None
    steady_since = clock()
    while True:
        try:
            info = path.stat()
        except FileNotFoundError as exc:
            raise InputError(f"{path.name} was gone before it could be read.") from exc
        except OSError as exc:
            raise InputError(f"{path.name} could not be read: {exc.strerror or exc}") from exc

        current = (info.st_size, info.st_mtime_ns)
        if current != mark:
            mark, steady_since = current, clock()
        # A zero-byte file is a placeholder a sync client hasn't filled yet,
        # not a memo that happens to be empty. Keep waiting for it.
        elif info.st_size > 0 and clock() - steady_since >= stable_seconds:
            return info

        if clock() >= deadline:
            what = "still empty" if info.st_size == 0 else "still changing"
            raise InputError(
                f"{path.name} was {what} after {timeout_s:.0f}s; leaving it for next time."
            )
        sleep(poll_seconds)


@contextlib.contextmanager
def _only_one_watcher(state_dir: Path) -> Iterator[None]:
    """Hold a lock for the watcher's lifetime, so two never share a state directory.

    A lock rather than a PID file: the kernel drops it when the process dies,
    however it dies, so a crashed watcher never leaves one behind to block the
    next one.
    """
    try:
        with safe.file_lock(
            state_dir / WATCH_LOCK_NAME, timeout_s=WATCH_LOCK_TIMEOUT_S, what=str(state_dir)
        ):
            yield
    except VoxmdError as exc:
        raise ConfigError(
            "Another voxmd watcher is already running on this state directory. "
            "Run `voxmd status` to see it."
        ) from exc


def _raise_stopped(signum: int, _frame: Any) -> None:
    """Signal handler. Deliberately does nothing but restore and raise.

    No logging, no locks, no file writes: a handler runs in the middle of
    whatever the main thread was doing, and the log it would want to write to
    may be the very thing that thread is holding open.
    """
    signal.signal(signum, signal.SIG_DFL)
    raise Stopped(signal.Signals(signum).name)


@contextlib.contextmanager
def _stop_on_signal() -> Iterator[None]:
    """Make SIGTERM and SIGHUP end the watcher the way Ctrl-C does.

    Restoring the default handler first means a second signal kills outright,
    so a watcher wedged in a long transcription can always be stopped.
    """
    if threading.current_thread() is not threading.main_thread():  # pragma: no cover
        yield
        return
    previous: list[tuple[int, Any]] = []
    for name in STOP_SIGNALS:
        number = getattr(signal, name, None)
        if number is None:  # pragma: no cover - both exist on every Unix
            continue
        previous.append((number, signal.signal(number, _raise_stopped)))
    try:
        yield
    finally:
        for number, old in previous:
            signal.signal(number, old)


def _stop_observer(observer: Any, log: EventLog) -> None:
    try:
        observer.stop()
        observer.join(timeout=5)
    except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
        log.event("observer.stop_failed", error=type(exc).__name__, detail=str(exc))


def _within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or resolved.is_relative_to(root)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return str(exc)
