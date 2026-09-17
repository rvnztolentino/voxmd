"""The watcher's event log: timestamps, quoting, permissions, rolling (C2)."""

from __future__ import annotations

import io
import json
import re
import stat
from pathlib import Path

import pytest

from voxmd.errors import OutputError
from voxmd.logging_setup import LOG_NAME, EventLog, log_path

TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4}")


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_every_line_starts_with_a_timestamp_and_the_event_name(tmp_path: Path) -> None:
    with EventLog(tmp_path / "voxmd.log", max_bytes=1 << 20) as log:
        log.event("watch.start", pid=412, dir="/inbox")
        log.event("wake", trigger="created", file="memo.m4a")

    first, second = lines(tmp_path / "voxmd.log")
    stamp, name, *fields = first.split(" ")
    assert TIMESTAMP.fullmatch(stamp)
    assert name == "watch.start"
    assert fields == ["pid=412", "dir=/inbox"]
    assert second.split(" ")[1:] == ["wake", "trigger=created", "file=memo.m4a"]


def test_the_log_is_created_private_to_the_user(tmp_path: Path) -> None:
    path = tmp_path / "voxmd.log"
    with EventLog(path, max_bytes=1 << 20) as log:
        log.event("watch.start")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_each_line_is_flushed_so_tail_f_shows_it_immediately(tmp_path: Path) -> None:
    """Read the file while the log is still open: nothing may sit in a buffer."""
    path = tmp_path / "voxmd.log"
    log = EventLog(path, max_bytes=1 << 20)
    try:
        log.event("wake", file="memo.m4a")
        assert len(lines(path)) == 1
        log.event("file.done", file="memo.m4a")
        assert len(lines(path)) == 2
    finally:
        log.close()


def test_a_filename_cannot_forge_a_second_log_line(tmp_path: Path) -> None:
    """A recording called "memo\\n<fake entry>" must not become two events."""
    path = tmp_path / "voxmd.log"
    hostile = "memo.m4a\n2020-01-01T00:00:00+0000 watch.stop reason=forged"
    with EventLog(path, max_bytes=1 << 20) as log:
        log.event("wake", file=hostile)

    assert len(lines(path)) == 1
    value = lines(path)[0].split("file=", 1)[1]
    assert json.loads(value) == hostile


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("memo.m4a", "memo.m4a"),
        ("/inbox/a-b_c.m4a", "/inbox/a-b_c.m4a"),
        ("my memo.m4a", '"my memo.m4a"'),
        ('say "hi".m4a', '"say \\"hi\\".m4a"'),
        ("", '""'),
    ],
)
def test_values_are_quoted_only_when_they_need_to_be(
    tmp_path: Path, value: str, expected: str
) -> None:
    path = tmp_path / "voxmd.log"
    with EventLog(path, max_bytes=1 << 20) as log:
        log.event("wake", file=value)
    assert lines(path)[0].endswith(f"file={expected}")


def test_fields_that_are_none_are_left_out_rather_than_logged_as_none(tmp_path: Path) -> None:
    path = tmp_path / "voxmd.log"
    with EventLog(path, max_bytes=1 << 20) as log:
        log.event("file.done", file="memo.m4a", archived=None)
    assert "archived" not in lines(path)[0]


def test_events_are_echoed_when_asked_so_the_terminal_shows_the_work(tmp_path: Path) -> None:
    echo = io.StringIO()
    with EventLog(tmp_path / "voxmd.log", max_bytes=1 << 20, echo=echo) as log:
        log.event("wake", file="memo.m4a")
    assert echo.getvalue().strip() == lines(tmp_path / "voxmd.log")[0]


def test_a_log_past_its_limit_is_rolled_aside_not_deleted(tmp_path: Path) -> None:
    path = tmp_path / "voxmd.log"
    rolled = tmp_path / "voxmd.log.1"
    with EventLog(path, max_bytes=120) as log:
        log.event("wake", file="memo-0.m4a")
        assert not rolled.exists(), "one short line is not over the limit"
        log.event("wake", file="memo-1.m4a")
        log.event("wake", file="memo-2.m4a")

        assert rolled.exists(), "the previous generation is kept, not dropped"
        assert "memo-0.m4a" in rolled.read_text(encoding="utf-8")
        log.event("wake", file="memo-3.m4a")
        assert lines(path)[-1].endswith("file=memo-3.m4a"), "logging continues after a roll"


def test_rolling_keeps_one_generation_so_a_long_run_cannot_fill_the_disk(tmp_path: Path) -> None:
    path = tmp_path / "voxmd.log"
    with EventLog(path, max_bytes=120) as log:
        for index in range(200):
            log.event("wake", file=f"memo-{index}.m4a")

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["voxmd.log", "voxmd.log.1"]
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) < 600


def test_appending_to_an_existing_log_keeps_what_is_already_there(tmp_path: Path) -> None:
    path = tmp_path / "voxmd.log"
    with EventLog(path, max_bytes=1 << 20) as log:
        log.event("watch.start")
    with EventLog(path, max_bytes=1 << 20) as log:
        log.event("watch.stop")
    assert [line.split(" ")[1] for line in lines(path)] == ["watch.start", "watch.stop"]


def test_a_symlink_in_the_logs_place_is_refused(tmp_path: Path) -> None:
    """O_NOFOLLOW: a planted link must not redirect voxmd's writes."""
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep me")
    link = tmp_path / "voxmd.log"
    link.symlink_to(target)

    with pytest.raises(OutputError, match="Could not open the log"):
        EventLog(link, max_bytes=1 << 20)
    assert target.read_text() == "keep me"


def test_a_log_in_a_missing_folder_says_what_to_fix(tmp_path: Path) -> None:
    with pytest.raises(OutputError, match="log.file"):
        EventLog(tmp_path / "nope" / "voxmd.log", max_bytes=1 << 20)


def test_the_log_defaults_to_the_state_directory(tmp_path: Path) -> None:
    assert log_path(None, tmp_path) == tmp_path / LOG_NAME
    assert log_path(tmp_path / "custom.log", tmp_path) == tmp_path / "custom.log"
