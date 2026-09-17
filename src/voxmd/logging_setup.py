"""The watcher's log: every wake, every file, with a timestamp (C2).

Nothing the watcher does is allowed to happen silently. Each line is one event,
written as ``<ISO timestamp> <event> key=value ...`` — a shape that stays
readable in a terminal and still parses with ``awk`` or ``cut``.

**What is never written here:** transcript text, summaries, decisions, actions,
people, or topics. A memo's content stays in the note and in memory.

**What is written, deliberately:** the note's path, which contains the title the
model wrote from the memo. A trace that won't say where the note went is not a
trace worth reading, and the ledger already records the same path at the same
0600 permissions. If that is more than you want on disk, point ``log.file`` at
a location you control, or read the log and delete it — nothing depends on it.

Values are quoted as JSON whenever they hold anything but a bare token, so a
recording called ``memo.m4a\\n2020-01-01 fake`` cannot forge a second log line.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO

from .errors import OutputError

LOG_NAME = "voxmd.log"
ROLLED_SUFFIX = ".1"
# A value needing no quoting: nothing a shell, a parser, or a human would
# misread, and above all no whitespace and no newline.
_BARE = re.compile(r"[A-Za-z0-9_@:%+=./~-]*")


def log_path(file: Path | None, state_dir: Path) -> Path:
    """Where the log lives: ``log.file`` if set, else ``voxmd.log`` in the state dir."""
    return file if file is not None else state_dir / LOG_NAME


class EventLog:
    """An append-only event log, 0600, flushed line by line.

    Flushing every line costs nothing at this volume (a memo takes seconds) and
    buys two things: ``tail -f`` shows the watcher working in real time, and a
    crash or a ``kill -9`` cannot take the last few events with it.

    Writes are serialized: the watcher logs arrivals from the observer's thread
    and outcomes from its own, and two half-written lines interleaved would be
    worse than no log at all. Nothing writes here from a signal handler, which
    is what keeps that lock from being able to deadlock.
    """

    def __init__(self, path: Path, *, max_bytes: int, echo: IO[str] | None = None) -> None:
        self.path = path
        self._max_bytes = max_bytes
        self._echo = echo
        self._writing = threading.Lock()
        self._handle = _open_append(path)

    def event(self, event: str, /, **fields: object) -> None:
        """Append one event. ``None`` values are dropped rather than logged as "None".

        The event name is positional-only so that any field may be called
        ``event`` or ``name`` without colliding with it.
        """
        parts = [f"{key}={_value(value)}" for key, value in fields.items() if value is not None]
        line = " ".join([f"{datetime.now().astimezone():%Y-%m-%dT%H:%M:%S%z}", event, *parts])
        with self._writing:
            if self._echo is not None:
                print(line, file=self._echo, flush=True)
            try:
                self._handle.write(line + "\n")
                self._handle.flush()
            except OSError as exc:
                raise OutputError(
                    f"Could not write the log at {self.path}: {exc.strerror or exc}"
                ) from exc
            self._roll_if_large()

    def _roll_if_large(self) -> None:
        """Past the size limit, move the log aside and start a new one. Call with the lock held.

        One generation is kept. A watcher left running for months must not fill
        the disk, and an old log nobody has read in weeks is not worth more.
        """
        try:
            if os.fstat(self._handle.fileno()).st_size <= self._max_bytes:
                return
            self._handle.close()
            self.path.replace(self.path.parent / (self.path.name + ROLLED_SUFFIX))
        except OSError:
            # Rolling is housekeeping. Losing it must never take down a watcher
            # that is otherwise working, so reopen and carry on.
            pass
        self._handle = _open_append(self.path)

    def close(self) -> None:
        with self._writing:
            self._handle.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _open_append(path: Path) -> IO[str]:
    """Open the log for appending, creating it 0600.

    ``O_NOFOLLOW`` refuses a symlink in the log's own place, so a planted link
    can't redirect voxmd's writes into a file it was never meant to touch.
    """
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise OutputError(
            f"Could not open the log at {path}: {exc.strerror or exc}\n"
            "Set log.file to somewhere writable, or fix state.dir."
        ) from exc
    return os.fdopen(fd, "a", encoding="utf-8", newline="\n")


def _value(value: object) -> str:
    text = str(value)
    return text if text and _BARE.fullmatch(text) else json.dumps(text)
