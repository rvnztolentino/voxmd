"""``voxmd status``: is the watcher running, and what has it done (C3).

**A PID file is not proof of life.** PIDs get recycled, and a watcher killed
with ``SIGKILL`` leaves its state file behind, so trusting the number in it
would report "running" forever. Two things are checked instead: that the
process exists, and that it is really a voxmd watcher rather than whatever
inherited the number. A state file whose process is gone is reported as stale
and not running — the one answer that is never allowed here is a confident
wrong one.

"Files processed today" comes from the ledger rather than a counter the watcher
keeps, so it survives a restart and includes manual ``voxmd process`` runs.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import safe
from .config import Config
from .errors import ConfigError, OutputError, VoxmdError
from .ledger import LEDGER_NAME, Ledger
from .logging_setup import log_path
from .schema import describe_errors

STATE_NAME = "watch.json"
WATCH_LOCK_NAME = "watch.lock"
PS_TIMEOUT_S = 5.0
PS_HINT = "ps is part of macOS and every Linux distribution; check your PATH."


class WatchState(BaseModel):
    """What the running watcher writes about itself, for ``status`` to read."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    pid: int = Field(ge=1)
    started_at: datetime
    watch_dir: str
    log: str
    observer: str
    """The watchdog observer class in use, e.g. ``FSEventsObserver`` (C4)."""
    last_wake: datetime | None = None
    last_wake_trigger: str | None = None
    wakes: int = 0
    processed: int = 0
    failed: int = 0
    stopped_at: datetime | None = None
    """Set when the watcher shut down cleanly, so status needn't guess."""


@dataclass(frozen=True)
class WatcherStatus:
    """What ``voxmd status`` found."""

    state: WatchState | None
    running: bool
    detail: str
    """How that was decided: "pid 412", "stale state file", and so on."""
    processed_today: int = 0
    total_processed: int = 0
    log: Path | None = None
    ledger_error: str | None = None


def state_path(state_dir: Path) -> Path:
    return state_dir / STATE_NAME


def write_state(path: Path, state: WatchState) -> None:
    """Replace the state file atomically, 0600."""
    try:
        safe.atomic_write_text(path, state.model_dump_json(indent=2) + "\n")
    except OSError as exc:
        raise OutputError(f"Could not write {path}: {exc.strerror or exc}") from exc


def read_state(path: Path) -> WatchState | None:
    """The state file, or None when no watcher has ever run here."""
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc.strerror or exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError(f"Watcher state is not a regular file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path} is not readable JSON: {exc}") from exc
    try:
        return WatchState.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid watcher state {path} ({describe_errors(exc)}).") from exc


def status(settings: Config, *, today: date | None = None) -> WatcherStatus:
    """Read the watcher's state and the ledger. Creates nothing."""
    state_dir = settings.state.dir
    state = read_state(state_path(state_dir)) if state_dir.is_dir() else None
    running, detail = _liveness(state)

    processed_today = total = 0
    ledger_error = None
    try:
        ledger = Ledger.load(state_dir / LEDGER_NAME, max_bytes=settings.limits.max_ledger_bytes)
        processed_today = ledger.processed_on(today or date.today())
        total = len(ledger)
    except VoxmdError as exc:
        ledger_error = str(exc)

    return WatcherStatus(
        state=state,
        running=running,
        detail=detail,
        processed_today=processed_today,
        total_processed=total,
        log=log_path(settings.log.file, state_dir),
        ledger_error=ledger_error,
    )


def _liveness(state: WatchState | None) -> tuple[bool, str]:
    if state is None:
        return False, "no watcher has run here"
    if state.stopped_at is not None:
        return False, f"stopped cleanly at {state.stopped_at:%Y-%m-%d %H:%M}"
    if not pid_alive(state.pid):
        return False, f"stale state file: pid {state.pid} is gone (killed, or the machine rebooted)"
    identified = pid_is_voxmd(state.pid)
    if identified is False:
        return False, f"stale state file: pid {state.pid} now belongs to another program"
    if identified is None:
        return True, f"pid {state.pid} is alive, but it could not be identified"
    return True, f"pid {state.pid}"


def pid_alive(pid: int) -> bool:
    """Whether a process with this id exists. Signal 0 checks without sending one."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists; it just isn't ours. pid_is_voxmd then rules it out.
        return True
    except OSError:  # pragma: no cover - defensive
        return False
    return True


def pid_is_voxmd(pid: int) -> bool | None:
    """Whether that process is really a voxmd watcher. None when ``ps`` couldn't say.

    Without this, a recycled PID would let ``status`` report a long-dead
    watcher as running.
    """
    try:
        found = safe.resolve_tool("ps", hint=PS_HINT)
        result = safe.run([found, "-p", str(pid), "-o", "args="], timeout=PS_TIMEOUT_S, what="ps")
    except VoxmdError:
        return None
    if result.returncode != 0:
        return False
    command = result.stdout.strip()
    if not command:
        return False
    return "voxmd" in command and "watch" in command
