"""The shared subprocess and input-file policy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from voxmd import safe
from voxmd.errors import AudioError, DependencyError, ToolTimeout

AUDIO = {".m4a", ".wav"}


class TestRun:
    def test_arguments_are_never_interpreted_by_a_shell(self) -> None:
        # If any shell touched this, the substitutions would expand and the
        # output would differ from the input.
        payload = "; echo pwned && $(echo pwned) `echo pwned` | cat > /tmp/x"
        result = safe.run(["/bin/echo", payload], timeout=10, what="echo")
        assert result.returncode == 0
        assert result.stdout.rstrip("\n") == payload

    def test_timeout_kills_and_raises(self) -> None:
        with pytest.raises(ToolTimeout, match="sleep exceeded"):
            safe.run(["/bin/sleep", "5"], timeout=0.2, what="sleep")

    def test_stdin_is_closed(self) -> None:
        # `cat` with no args reads stdin; with stdin closed it returns at once
        # instead of blocking until the timeout.
        result = safe.run(["/bin/cat"], timeout=5, what="cat")
        assert result.returncode == 0
        assert result.stdout == ""

    def test_nonzero_exit_is_returned_not_raised(self) -> None:
        result = safe.run([sys.executable, "-c", "raise SystemExit(3)"], timeout=10, what="py")
        assert result.returncode == 3


class TestResolveTool:
    def test_finds_tool_on_path(self) -> None:
        assert safe.resolve_tool("sh", hint="").name == "sh"

    def test_missing_tool_carries_the_hint(self) -> None:
        with pytest.raises(DependencyError, match="brew install nothing"):
            safe.resolve_tool("voxmd-definitely-not-a-tool", hint="brew install nothing")

    def test_explicit_path_must_be_executable(self, tmp_path: Path) -> None:
        not_exec = tmp_path / "tool"
        not_exec.write_text("#!/bin/sh\n")
        with pytest.raises(DependencyError, match="not an executable"):
            safe.resolve_tool(str(not_exec), hint="")
        not_exec.chmod(0o755)
        assert safe.resolve_tool(str(not_exec), hint="") == not_exec


class TestResolveInputFile:
    def test_accepts_a_normal_file(self, tmp_path: Path) -> None:
        memo = tmp_path / "memo.m4a"
        memo.write_bytes(b"x" * 10)
        assert (
            safe.resolve_input_file(memo, allowed_suffixes=AUDIO, max_bytes=100) == memo.resolve()
        )

    def test_suffix_match_is_case_insensitive(self, tmp_path: Path) -> None:
        memo = tmp_path / "MEMO.M4A"
        memo.write_bytes(b"x")
        safe.resolve_input_file(memo, allowed_suffixes=AUDIO, max_bytes=100)

    def test_missing(self, tmp_path: Path) -> None:
        with pytest.raises(AudioError, match="No such file"):
            safe.resolve_input_file(tmp_path / "nope.m4a", allowed_suffixes=AUDIO, max_bytes=100)

    def test_directory(self, tmp_path: Path) -> None:
        folder = tmp_path / "folder.m4a"
        folder.mkdir()
        with pytest.raises(AudioError, match="Not a regular file"):
            safe.resolve_input_file(folder, allowed_suffixes=AUDIO, max_bytes=100)

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
    def test_fifo_is_rejected_before_anything_reads_it(self, tmp_path: Path) -> None:
        # Handing a FIFO to ffmpeg would block forever waiting for a writer.
        fifo = tmp_path / "trap.m4a"
        os.mkfifo(fifo)
        with pytest.raises(AudioError, match="Not a regular file"):
            safe.resolve_input_file(fifo, allowed_suffixes=AUDIO, max_bytes=100)

    def test_unsupported_suffix(self, tmp_path: Path) -> None:
        doc = tmp_path / "notes.txt"
        doc.write_text("hi")
        with pytest.raises(AudioError, match="Unsupported file type"):
            safe.resolve_input_file(doc, allowed_suffixes=AUDIO, max_bytes=100)

    def test_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.m4a"
        empty.touch()
        with pytest.raises(AudioError, match="empty"):
            safe.resolve_input_file(empty, allowed_suffixes=AUDIO, max_bytes=100)

    def test_oversized_names_the_setting_to_change(self, tmp_path: Path) -> None:
        big = tmp_path / "big.m4a"
        big.write_bytes(b"x" * 101)
        with pytest.raises(AudioError, match="max_audio_mb"):
            safe.resolve_input_file(big, allowed_suffixes=AUDIO, max_bytes=100)

    def test_symlink_is_resolved_to_its_target(self, tmp_path: Path) -> None:
        real = tmp_path / "real.m4a"
        real.write_bytes(b"x")
        link = tmp_path / "link.m4a"
        link.symlink_to(real)
        assert (
            safe.resolve_input_file(link, allowed_suffixes=AUDIO, max_bytes=100) == real.resolve()
        )


@pytest.mark.parametrize(
    ("size", "expected"),
    [(512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB"), (3 * 1024**3, "3.0 GB")],
)
def test_human_bytes(size: int, expected: str) -> None:
    assert safe.human_bytes(size) == expected
