"""voxmd status: liveness that is never a confident guess (C3)."""

from __future__ import annotations

import os
import stat
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from conftest import FakeProc, FakeRunner
from voxmd import safe
from voxmd import status as status_mod
from voxmd.config import Config
from voxmd.errors import ConfigError
from voxmd.ledger import LEDGER_NAME, Ledger, LedgerEntry
from voxmd.status import (
    WatchState,
    pid_alive,
    pid_is_voxmd,
    read_state,
    state_path,
    status,
    write_state,
)

DEAD_PID = 0x7FFFFFF0
"""Far above the usual pid ceiling, so nothing real is behind it."""


@pytest.fixture
def settings(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "watch": {"dir": str(tmp_path)},
            "state": {"dir": str(tmp_path / "state")},
        }
    )


def a_state(**overrides: object) -> WatchState:
    fields: dict[str, object] = {
        "pid": os.getpid(),
        "started_at": datetime.now().astimezone(),
        "watch_dir": "/inbox",
        "log": "/state/voxmd.log",
        "observer": "FSEventsObserver",
        **overrides,
    }
    return WatchState(**fields)  # type: ignore[arg-type]


def is_voxmd(monkeypatch: pytest.MonkeyPatch, answer: bool | None) -> None:
    monkeypatch.setattr(status_mod, "pid_is_voxmd", lambda _pid: answer)


def record(settings: Config, when: datetime, digest: str) -> None:
    settings.state.dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.load(settings.state.dir / LEDGER_NAME, max_bytes=1 << 20)
    ledger.record(
        digest,
        LedgerEntry(source="/memo.m4a", size=1, processed_at=when, note="/note.md"),
    )


# --- the state file ---------------------------------------------------------


def test_no_state_file_means_no_watcher_has_ever_run(tmp_path: Path) -> None:
    assert read_state(tmp_path / "watch.json") is None


def test_state_survives_a_round_trip_and_is_private_to_the_user(tmp_path: Path) -> None:
    path = tmp_path / "watch.json"
    write_state(path, a_state(wakes=3, processed=2))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    back = read_state(path)
    assert back is not None and (back.wakes, back.processed) == (3, 2)


def test_a_damaged_state_file_is_reported_not_ignored(tmp_path: Path) -> None:
    path = tmp_path / "watch.json"
    path.write_text("{ not json")
    with pytest.raises(ConfigError, match="not readable JSON"):
        read_state(path)


def test_state_with_the_wrong_shape_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "watch.json"
    path.write_text('{"version": 1, "pid": "not a number"}')
    with pytest.raises(ConfigError, match="Invalid watcher state"):
        read_state(path)


# --- liveness ---------------------------------------------------------------


def test_this_process_is_alive_and_a_made_up_one_is_not() -> None:
    assert pid_alive(os.getpid()) is True
    assert pid_alive(DEAD_PID) is False


def test_a_process_is_identified_by_its_command_line(
    monkeypatch: pytest.MonkeyPatch, fake_tools: None
) -> None:
    runner = FakeRunner()
    runner.responses["ps"] = FakeProc(stdout="/usr/bin/python voxmd watch -c x.yaml\n")
    monkeypatch.setattr(safe, "run", runner)

    assert pid_is_voxmd(4123) is True
    assert runner.called("ps")[0].argv[1:] == ["-p", "4123", "-o", "args="]


@pytest.mark.parametrize(
    ("stdout", "returncode", "expected"),
    [
        ("voxmd watch", 0, True),
        ("/bin/zsh -l", 0, False),
        ("voxmd process memo.m4a", 0, False),
        ("", 0, False),
        ("", 1, False),
    ],
)
def test_a_recycled_pid_is_not_mistaken_for_the_watcher(
    monkeypatch: pytest.MonkeyPatch,
    fake_tools: None,
    stdout: str,
    returncode: int,
    expected: bool,
) -> None:
    runner = FakeRunner()
    runner.responses["ps"] = FakeProc(stdout=stdout, returncode=returncode)
    monkeypatch.setattr(safe, "run", runner)

    assert pid_is_voxmd(4123) is expected


def test_an_unusable_ps_says_it_does_not_know_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safe, "resolve_tool", _raise_dependency)
    assert pid_is_voxmd(4123) is None


def _raise_dependency(name: str, *, hint: str) -> Path:
    from voxmd.errors import DependencyError

    raise DependencyError(f"{name} not found")


# --- what status reports ----------------------------------------------------


def test_nothing_has_ever_run_here(settings: Config) -> None:
    found = status(settings)

    assert found.running is False
    assert found.detail == "no watcher has run here"
    assert found.state is None
    assert found.log == settings.state.dir / "voxmd.log"


def test_a_running_watcher_is_reported_with_its_pid(
    settings: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.state.dir.mkdir(parents=True)
    write_state(state_path(settings.state.dir), a_state())
    is_voxmd(monkeypatch, True)

    found = status(settings)

    assert found.running is True
    assert found.detail == f"pid {os.getpid()}"


def test_a_watcher_that_stopped_cleanly_is_not_running(
    settings: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.state.dir.mkdir(parents=True)
    write_state(state_path(settings.state.dir), a_state(stopped_at=datetime.now().astimezone()))
    is_voxmd(monkeypatch, True)

    found = status(settings)

    assert found.running is False
    assert "stopped cleanly" in found.detail


def test_a_state_file_left_by_a_killed_watcher_is_reported_stale(settings: Config) -> None:
    """SIGKILL leaves the file behind. Trusting it would report running forever."""
    settings.state.dir.mkdir(parents=True)
    write_state(state_path(settings.state.dir), a_state(pid=DEAD_PID))

    found = status(settings)

    assert found.running is False
    assert "stale state file" in found.detail and "is gone" in found.detail


def test_a_pid_that_now_belongs_to_something_else_is_reported_stale(
    settings: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.state.dir.mkdir(parents=True)
    write_state(state_path(settings.state.dir), a_state())
    is_voxmd(monkeypatch, False)

    found = status(settings)

    assert found.running is False
    assert "belongs to another program" in found.detail


def test_an_unidentifiable_process_is_reported_honestly(
    settings: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.state.dir.mkdir(parents=True)
    write_state(state_path(settings.state.dir), a_state())
    is_voxmd(monkeypatch, None)

    found = status(settings)

    assert found.running is True
    assert "could not be identified" in found.detail


# --- counts -----------------------------------------------------------------


def test_files_processed_today_comes_from_the_ledger(settings: Config) -> None:
    now = datetime.now().astimezone()
    record(settings, now, "a" * 64)
    record(settings, now - timedelta(minutes=5), "b" * 64)
    record(settings, now - timedelta(days=3), "c" * 64)

    found = status(settings)

    assert found.processed_today == 2
    assert found.total_processed == 3


def test_the_count_covers_a_named_day(settings: Config) -> None:
    record(settings, datetime(2026, 9, 15, 23, 59).astimezone(), "d" * 64)

    assert status(settings, today=date(2026, 9, 15)).processed_today == 1
    assert status(settings, today=date(2026, 9, 16)).processed_today == 0


def test_a_damaged_ledger_costs_the_count_not_the_whole_command(settings: Config) -> None:
    settings.state.dir.mkdir(parents=True)
    (settings.state.dir / LEDGER_NAME).write_text("{")

    found = status(settings)

    assert found.ledger_error is not None
    assert found.processed_today == 0
    assert found.running is False


def test_status_creates_nothing(settings: Config) -> None:
    before = sorted(settings.state.dir.parent.rglob("*"))

    status(settings)

    assert sorted(settings.state.dir.parent.rglob("*")) == before
    assert not settings.state.dir.exists()
