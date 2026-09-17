"""Stage 6: the folder watcher, with the observer and the pipeline faked.

Nothing here starts a real observer or transcribes anything. What is real is
the queueing, the stability gate, the layout checks, the log, and the refusal
to run on a polling observer.
"""

from __future__ import annotations

import ast
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from voxmd import watcher as watch_mod
from voxmd.config import Config
from voxmd.errors import ConfigError, DependencyError, InputError, PartialFailure, ToolTimeout
from voxmd.logging_setup import EventLog
from voxmd.pipeline import ProcessResult
from voxmd.status import read_state, state_path
from voxmd.watcher import (
    ArrivalHandler,
    Watcher,
    check_layout,
    check_observer,
    interesting,
    resolve_watch_dir,
    settle,
    watch,
)

# --- fakes ------------------------------------------------------------------


@dataclass
class FakeObserver:
    """Stands in for watchdog's observer. Records what it was asked to watch."""

    scheduled: list[tuple[str, bool]] = field(default_factory=list)
    handler: Any = None
    started: bool = False
    stopped: bool = False

    def schedule(self, handler: Any, path: str, recursive: bool = False) -> None:
        self.handler = handler
        self.scheduled.append((path, recursive))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:
        return None


@dataclass
class FakeEvent:
    event_type: str
    src_path: str
    dest_path: str = ""
    is_directory: bool = False


@dataclass
class FakeProcess:
    """Stands in for pipeline.process. Replies, or raises, in order."""

    replies: list[object] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, audio: Path, **kwargs: Any) -> ProcessResult:
        self.calls.append({"audio": audio, **kwargs})
        reply = self.replies.pop(0) if self.replies else done(audio)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply(audio)
        return reply  # type: ignore[return-value]


def done(audio: Path, **kwargs: Any) -> ProcessResult:
    fields: dict[str, Any] = {
        "source": audio,
        "note": audio.parent / "note.md",
        "seconds": 1.5,
        **kwargs,
    }
    return ProcessResult(**fields)


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    path = tmp_path / "inbox"
    path.mkdir()
    return path


@pytest.fixture
def settings(tmp_path: Path, inbox: Path) -> Config:
    (tmp_path / "vault").mkdir()
    return Config.model_validate(
        {
            "vault": {"path": str(tmp_path / "vault"), "folder": "Memos"},
            "archive": {"dir": str(tmp_path / "archive")},
            "watch": {"dir": str(inbox), "stable_seconds": 0.5, "poll_seconds": 0.1},
            "state": {"dir": str(tmp_path / "state")},
            "entities": {"file": str(tmp_path / "entities.json")},
        }
    )


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    (tmp_path / "state").mkdir(exist_ok=True)
    with EventLog(tmp_path / "state" / "voxmd.log", max_bytes=1 << 20) as handle:
        yield handle


@pytest.fixture
def instant_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the stability wait; it has its own tests with a fake clock."""
    monkeypatch.setattr(watch_mod, "settle", lambda path, **_kwargs: path.stat())


def events(log: EventLog) -> list[tuple[str, str]]:
    """Each logged line as (event name, the rest)."""
    lines = log.path.read_text(encoding="utf-8").splitlines()
    return [(line.split(" ")[1], " ".join(line.split(" ")[2:])) for line in lines]


def names(log: EventLog) -> list[str]:
    return [name for name, _rest in events(log)]


def drain(watcher: Watcher) -> None:
    """Handle everything queued, then return.

    ``stop()`` deliberately abandons the queue (Ctrl-C should stop now, not
    after the backlog), so a test that wants the queue drained puts the
    sentinel on directly.
    """
    watcher._queue.put(None)
    watcher.serve()


def memo(inbox: Path, name: str = "memo.m4a") -> Path:
    path = inbox / name
    path.write_bytes(b"\x00" * 2048)
    return path


# --- what counts as a memo --------------------------------------------------


@pytest.mark.parametrize(
    ("name", "wanted"),
    [
        ("memo.m4a", True),
        ("memo.MP3", True),
        ("memo.wav", True),
        (".memo.m4a.icloud", False),
        (".DS_Store", False),
        (".dropbox.attr", False),
        ("notes.txt", False),
        ("memo", False),
        ("memo.m4a.part", False),
    ],
)
def test_only_real_audio_is_queued(tmp_path: Path, name: str, wanted: bool) -> None:
    assert interesting(tmp_path / name)[0] is wanted


def test_an_ignored_file_says_why(tmp_path: Path) -> None:
    assert interesting(tmp_path / ".memo.m4a.icloud")[1] == "hidden file"
    assert "not audio" in interesting(tmp_path / "notes.txt")[1]


# --- the stability gate -----------------------------------------------------


class Clock:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_a_file_that_is_not_changing_is_read_at_once(inbox: Path) -> None:
    path = memo(inbox)
    clock = Clock()

    info = settle(
        path, stable_seconds=3, poll_seconds=0.5, timeout_s=60, sleep=clock.sleep, clock=clock
    )

    assert info.st_size == 2048
    assert clock.now == pytest.approx(3.0), "it waits the full stable window, then reads"


def test_a_file_still_being_written_is_waited_for(inbox: Path) -> None:
    """The point of the gate: a half-synced memo would transcribe as a truncated note."""
    path = memo(inbox)
    clock = Clock()
    growth = iter([1024, 1024, 1024])

    def grow(seconds: float) -> None:
        clock.sleep(seconds)
        with contextlib_suppress_stop():
            path.write_bytes(path.read_bytes() + b"\x00" * next(growth))

    info = settle(path, stable_seconds=1, poll_seconds=0.5, timeout_s=60, sleep=grow, clock=clock)

    assert info.st_size == 2048 + 3 * 1024
    assert clock.now > 1.0, "the window restarts every time the file changes"


def contextlib_suppress_stop() -> Any:
    import contextlib

    return contextlib.suppress(StopIteration)


def test_an_empty_placeholder_is_not_mistaken_for_a_finished_memo(inbox: Path) -> None:
    """iCloud and Dropbox create the file first and fill it later."""
    path = inbox / "memo.m4a"
    path.write_bytes(b"")
    clock = Clock()

    with pytest.raises(InputError, match="still empty"):
        settle(
            path, stable_seconds=1, poll_seconds=0.5, timeout_s=5, sleep=clock.sleep, clock=clock
        )


def test_a_file_that_never_settles_is_left_for_next_time(inbox: Path) -> None:
    path = memo(inbox)
    clock = Clock()

    def keep_growing(seconds: float) -> None:
        clock.sleep(seconds)
        path.write_bytes(path.read_bytes() + b"\x00")

    with pytest.raises(InputError, match="still changing after 5s"):
        settle(
            path, stable_seconds=1, poll_seconds=0.5, timeout_s=5, sleep=keep_growing, clock=clock
        )


def test_a_file_deleted_while_waiting_is_reported_not_crashed(inbox: Path) -> None:
    path = memo(inbox)
    clock = Clock()

    def unlink(seconds: float) -> None:
        clock.sleep(seconds)
        path.unlink(missing_ok=True)

    with pytest.raises(InputError, match="was gone"):
        settle(path, stable_seconds=1, poll_seconds=0.5, timeout_s=5, sleep=unlink, clock=clock)


# --- C4: event-driven, never polling ----------------------------------------


def test_this_machines_observer_is_event_driven() -> None:
    """C4 asserted against the real watchdog install, not a fake."""
    expected = {"darwin": "FSEventsObserver", "linux": "InotifyObserver"}[sys.platform]
    observer = watch_mod.make_observer()
    try:
        assert check_observer(observer) == expected
    finally:
        observer.stop()


def test_a_polling_observer_is_refused_rather_than_quietly_accepted() -> None:
    from watchdog.observers.polling import PollingObserver

    observer = PollingObserver()
    try:
        with pytest.raises(DependencyError, match="checks the folder on a timer"):
            check_observer(observer)
    finally:
        observer.stop()


def test_a_native_observer_is_accepted() -> None:
    assert check_observer(FakeObserver()) == "FakeObserver"


# --- the folder and the layout ----------------------------------------------


def test_no_watch_folder_configured_says_what_to_set() -> None:
    with pytest.raises(ConfigError, match="watch.dir"):
        resolve_watch_dir(Config().watch)


def test_a_missing_watch_folder_is_never_created(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    settings = Config.model_validate({"watch": {"dir": str(missing)}})

    with pytest.raises(ConfigError, match="never creates it"):
        resolve_watch_dir(settings.watch)
    assert not missing.exists()


def test_a_watch_path_that_is_a_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "notafolder"
    path.write_text("")
    settings = Config.model_validate({"watch": {"dir": str(path)}})

    with pytest.raises(ConfigError, match="not a directory"):
        resolve_watch_dir(settings.watch)


def test_an_archive_inside_the_watch_folder_is_refused(tmp_path: Path, inbox: Path) -> None:
    """Otherwise every archived recording lands back under the watcher."""
    with pytest.raises(ConfigError, match="wake it again"):
        check_layout(inbox, inbox / "archive", tmp_path / "state")


def test_the_archive_being_the_watch_folder_itself_is_refused(tmp_path: Path, inbox: Path) -> None:
    with pytest.raises(ConfigError, match="inside the watch folder"):
        check_layout(inbox, inbox, tmp_path / "state")


def test_a_state_dir_inside_the_watch_folder_is_refused(inbox: Path) -> None:
    with pytest.raises(ConfigError, match="ledger and log would wake"):
        check_layout(inbox, None, inbox / "state")


def test_separate_folders_are_fine(tmp_path: Path, inbox: Path) -> None:
    check_layout(inbox, tmp_path / "archive", tmp_path / "state")


# --- the handler ------------------------------------------------------------


def test_created_and_moved_both_count_as_arrivals(settings: Config, log: EventLog) -> None:
    """Sync clients write a temp file and rename it, so `moved` is the common case."""
    watcher = Watcher(settings, log=log)
    handler = ArrivalHandler(watcher)

    handler.dispatch(FakeEvent("created", "/inbox/one.m4a"))
    handler.dispatch(FakeEvent("moved", "/inbox/.tmp", dest_path="/inbox/two.m4a"))
    handler.dispatch(FakeEvent("modified", "/inbox/three.m4a"))
    handler.dispatch(FakeEvent("created", "/inbox/sub", is_directory=True))

    assert [rest for name, rest in events(log) if name == "wake"] == [
        "trigger=created file=one.m4a",
        "trigger=moved file=two.m4a",
    ]


# --- queueing ---------------------------------------------------------------


def test_every_event_is_logged_even_the_uninteresting_ones(
    settings: Config, log: EventLog, inbox: Path
) -> None:
    """C2: a trace with gaps in it is not a trace."""
    watcher = Watcher(settings, log=log)

    assert watcher.offer(inbox / ".DS_Store", "created") is False

    assert names(log) == ["wake", "ignored"]
    assert watcher.wakes == 1


def test_the_same_file_is_not_queued_twice(settings: Config, log: EventLog, inbox: Path) -> None:
    watcher = Watcher(settings, log=log)
    path = memo(inbox)

    assert watcher.offer(path, "created") is True
    assert watcher.offer(path, "moved") is False

    assert [rest for name, rest in events(log) if name == "ignored"] == [
        'file=memo.m4a reason="already queued"'
    ]


def test_files_already_in_the_folder_are_picked_up_at_startup(
    settings: Config, log: EventLog, inbox: Path
) -> None:
    """Memos that synced while the watcher was off would otherwise wait forever."""
    memo(inbox, "a.m4a")
    memo(inbox, "b.m4a")
    memo(inbox, "notes.txt")
    (inbox / "subfolder.m4a").mkdir()
    watcher = Watcher(settings, log=log)

    assert watcher.scan(inbox) == 2
    assert ("scan", "found=2") in events(log)


# --- processing -------------------------------------------------------------


def test_a_memo_is_processed_and_the_outcome_logged(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    path = memo(inbox)
    run = FakeProcess([done(path, archived=inbox.parent / "archive" / "memo.m4a")])
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(path, "created")
    drain(watcher)

    assert watcher.processed == 1
    assert names(log) == ["wake", "file.ready", "file.done"]
    detail = dict(part.split("=", 1) for part in events(log)[-1][1].split(" "))
    assert detail["file"] == "memo.m4a"
    assert detail["note"].endswith("note.md")
    assert detail["seconds"] == "1.5"


def test_the_watcher_waits_for_a_manual_run_instead_of_giving_up_in_a_second(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    path = memo(inbox)
    run = FakeProcess()
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(path, "created")
    drain(watcher)

    assert run.calls[0]["lock_timeout_s"] == settings.watch.lock_timeout_s == 600.0


def test_a_recording_already_in_the_ledger_is_skipped_quietly(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    path = memo(inbox)
    run = FakeProcess([done(path, skipped=True)])
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(path, "created")
    drain(watcher)

    assert watcher.skipped == 1 and watcher.processed == 0
    assert events(log)[-1] == ("file.skipped", 'file=memo.m4a reason="already processed"')


def test_one_bad_memo_does_not_stop_the_watcher(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    first, second = memo(inbox, "bad.m4a"), memo(inbox, "good.m4a")
    run = FakeProcess([ToolTimeout("whisper-cli exceeded its 300s timeout"), done(second)])
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(first, "created")
    watcher.offer(second, "created")
    drain(watcher)

    assert (watcher.failed, watcher.processed) == (1, 1)
    assert [name for name in names(log) if name.startswith("file.")] == [
        "file.ready",
        "file.failed",
        "file.ready",
        "file.done",
    ]
    failure = next(rest for name, rest in events(log) if name == "file.failed")
    assert "error=ToolTimeout" in failure and "bad.m4a" in failure


def test_a_note_written_but_a_later_step_failed_is_logged_as_partial(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    path = memo(inbox)
    run = FakeProcess([PartialFailure("the ledger was not updated")])
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(path, "created")
    drain(watcher)

    assert events(log)[-1][0] == "file.partial"
    assert watcher.failed == 1


def test_problems_that_did_not_stop_the_note_are_still_logged(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    path = memo(inbox)
    run = FakeProcess([done(path, problems=("The recording was not archived: Permission denied",))])
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(path, "created")
    drain(watcher)

    assert names(log) == ["wake", "file.ready", "file.warning", "file.done"]


def test_a_file_that_never_settles_is_skipped_without_running_anything(
    settings: Config, log: EventLog, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = memo(inbox)
    monkeypatch.setattr(
        watch_mod, "settle", lambda *_a, **_k: (_ for _ in ()).throw(InputError("still changing"))
    )
    run = FakeProcess()
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(path, "created")
    drain(watcher)

    assert run.calls == []
    assert events(log)[-1] == ("file.skipped", 'file=memo.m4a reason="still changing"')


def test_stopping_abandons_the_queue_rather_than_working_through_it(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    """Ctrl-C means stop now. The files stay put and the next start rescans them."""
    run = FakeProcess()
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(memo(inbox, "a.m4a"), "created")
    watcher.offer(memo(inbox, "b.m4a"), "created")
    watcher.stop()
    watcher.serve()

    assert run.calls == []


def test_a_file_is_queued_again_after_it_has_been_handled(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    """Re-recorded under the same name, it must not be locked out by the dedupe set."""
    path = memo(inbox)
    run = FakeProcess()
    watcher = Watcher(settings, log=log, run=run)

    watcher.offer(path, "created")
    drain(watcher)
    assert watcher.offer(path, "created") is True


# --- start to finish --------------------------------------------------------


def test_the_watcher_starts_scans_processes_and_stops(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    memo(inbox)
    observer = FakeObserver()
    run = FakeProcess([lambda path: _stop_after(path)])

    watch(settings, log=log, run=run, observer=observer)

    assert observer.scheduled == [(str(inbox), False)]
    assert observer.started and observer.stopped
    assert names(log) == ["watch.start", "scan", "wake", "file.ready", "watch.stop"]
    assert "reason=SIGTERM" in events(log)[-1][1]
    start = dict(part.split("=", 1) for part in events(log)[0][1].split(" "))
    assert start["pid"] == str(os.getpid())
    assert start["observer"] == "FakeObserver"


def _stop_after(path: Path) -> ProcessResult:
    """Stop the watcher the way closing the terminal does (C1): SIGHUP-style, mid-memo."""
    os.kill(os.getpid(), signal.SIGTERM)
    raise AssertionError("the signal should have interrupted this")  # pragma: no cover


def test_ctrl_c_is_a_clean_stop_not_a_crash(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    memo(inbox)
    run = FakeProcess([KeyboardInterrupt()])

    watcher = watch(settings, log=log, run=run, observer=FakeObserver())

    assert events(log)[-1][0] == "watch.stop"
    assert "reason=interrupt" in events(log)[-1][1]
    assert watcher.processed == 0


def test_the_watcher_writes_state_that_status_can_read(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    memo(inbox)
    run = FakeProcess([lambda path: _stop_after(path)])

    watch(settings, log=log, run=run, observer=FakeObserver())

    state = read_state(state_path(settings.state.dir))
    assert state is not None
    assert state.pid == os.getpid()
    assert state.watch_dir == str(inbox)
    assert state.observer == "FakeObserver"
    assert state.wakes == 1
    assert state.last_wake_trigger == "scan"
    assert state.stopped_at is not None, "a clean stop must not look like a crash"


def test_two_watchers_cannot_share_a_state_directory(
    settings: Config, log: EventLog, inbox: Path
) -> None:
    from voxmd import safe
    from voxmd.status import WATCH_LOCK_NAME

    settings.state.dir.mkdir(parents=True, exist_ok=True)
    with (
        safe.file_lock(settings.state.dir / WATCH_LOCK_NAME, timeout_s=1, what="test"),
        pytest.raises(ConfigError, match="Another voxmd watcher is already running"),
    ):
        watch(settings, log=log, run=FakeProcess(), observer=FakeObserver())


def test_a_bad_vault_fails_at_startup_not_on_the_first_memo(
    settings: Config, log: EventLog, tmp_path: Path
) -> None:
    broken = settings.model_copy(
        update={"vault": settings.vault.model_copy(update={"path": tmp_path / "gone"})}
    )
    with pytest.raises(ConfigError, match="Vault not found"):
        watch(broken, log=log, run=FakeProcess(), observer=FakeObserver())


def test_an_unexpected_error_is_logged_before_it_escapes(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    """A bug should take the watcher down loudly, and leave the reason in the log."""
    memo(inbox)
    run = FakeProcess([RuntimeError("boom")])

    with pytest.raises(RuntimeError, match="boom"):
        watch(settings, log=log, run=run, observer=FakeObserver())

    assert [name for name in names(log)][-2:] == ["watch.crash", "watch.stop"]


# --- C1: nothing auto-starts ------------------------------------------------

FORBIDDEN = (
    "launchctl",
    "LaunchAgent",
    "plist",
    "crontab",
    "systemd",
    "login item",
    "setsid",
    "daemon",
    "nohup",
)


def code_without_docstrings(source: str) -> str:
    """The module's code with comments and docstrings removed.

    The docstrings explain at length what voxmd refuses to do, and those words
    are exactly the ones this test looks for, so they have to come out first.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
    return ast.unparse(tree)


def test_no_module_can_install_anything_that_survives_a_reboot() -> None:
    """C1, checked against the source rather than trusted."""
    package = Path(watch_mod.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        code = code_without_docstrings(path.read_text(encoding="utf-8")).lower()
        offenders += [(path.name, word) for word in FORBIDDEN if word.lower() in code]
    assert offenders == []


def test_a_full_watch_run_writes_nothing_outside_voxmds_own_folders(
    settings: Config, log: EventLog, inbox: Path, tmp_path: Path, instant_settle: None
) -> None:
    memo(inbox)
    before = {path for path in tmp_path.rglob("*")}

    watch(
        settings,
        log=log,
        run=FakeProcess([lambda path: _stop_after(path)]),
        observer=FakeObserver(),
    )

    created = {path for path in tmp_path.rglob("*")} - before
    assert all(path.is_relative_to(settings.state.dir) for path in created), created


def test_a_terminal_closing_stops_the_watcher_at_once_not_after_the_memo(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    """SIGHUP behaves exactly like Ctrl-C: the memo in flight is abandoned, not finished.

    The recording is untouched, so the next start picks it up in its scan.
    """
    memo(inbox)
    run = FakeProcess([lambda path: _stop_with(signal.SIGHUP)])

    watcher = watch(settings, log=log, run=run, observer=FakeObserver())

    assert watcher.processed == 0
    assert "reason=SIGHUP" in events(log)[-1][1]
    assert (inbox / "memo.m4a").exists()


def _stop_with(number: int) -> ProcessResult:
    os.kill(os.getpid(), number)
    raise AssertionError("the signal should have interrupted this")  # pragma: no cover


def test_the_signal_handlers_are_put_back_afterwards(
    settings: Config, log: EventLog, inbox: Path, instant_settle: None
) -> None:
    memo(inbox)
    before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}

    watch(settings, log=log, run=FakeProcess([KeyboardInterrupt()]), observer=FakeObserver())

    assert {n: signal.getsignal(n) for n in before} == before
