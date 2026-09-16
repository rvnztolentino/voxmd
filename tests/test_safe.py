"""The shared subprocess and input-file policy."""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

import pytest

from voxmd import safe
from voxmd.errors import AudioError, DependencyError, ToolTimeout

AUDIO = {".m4a", ".wav"}


class TestAtomicWriteText:
    def test_writes_a_private_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"

        safe.atomic_write_text(path, "hi\n")

        assert path.read_text() == "hi\n"
        assert path.stat().st_mode & 0o777 == 0o600

    def test_replaces_the_old_file_and_leaves_no_temp_files(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("old")

        safe.atomic_write_text(path, "new")

        assert path.read_text() == "new"
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]

    def test_a_failed_write_keeps_the_old_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "state.json"
        path.write_text("old")

        def fail(fd: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="disk full"):
            safe.atomic_write_text(path, "new")

        assert path.read_text() == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


class TestFileLock:
    def test_a_second_holder_times_out(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        with (
            safe.file_lock(lock, timeout_s=1, what="x"),
            pytest.raises(ToolTimeout, match="locked"),
            safe.file_lock(lock, timeout_s=0.1, what="x"),
        ):
            pass

    def test_the_lock_is_released_on_exit(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        with safe.file_lock(lock, timeout_s=1, what="x"):
            pass
        with safe.file_lock(lock, timeout_s=0.1, what="x"):
            pass

    def test_a_planted_symlink_is_not_followed(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere"
        link = tmp_path / "x.lock"
        link.symlink_to(target)

        with pytest.raises(OSError), safe.file_lock(link, timeout_s=0.1, what="x"):
            pass
        assert not target.exists()


def names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def no_hard_links(src: object, dst: object) -> None:
    raise OSError(errno.ENOTSUP, "Operation not supported")


class TestWriteNewFile:
    def test_writes_a_private_file(self, tmp_path: Path) -> None:
        path = safe.write_new_file(tmp_path, "note.md", "hi\n")

        assert path == tmp_path / "note.md"
        assert path.read_text() == "hi\n"
        assert path.stat().st_mode & 0o777 == 0o600

    def test_a_taken_name_gets_a_number_and_existing_files_are_untouched(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "note.md").write_text("mine")
        (tmp_path / "note 2.md").write_text("also mine")

        path = safe.write_new_file(tmp_path, "note.md", "new")

        assert path.name == "note 3.md"
        assert path.read_text() == "new"
        assert (tmp_path / "note.md").read_text() == "mine"
        assert (tmp_path / "note 2.md").read_text() == "also mine"
        assert names(tmp_path) == ["note 2.md", "note 3.md", "note.md"]

    def test_without_hard_links_it_still_never_replaces_a_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "link", no_hard_links)
        (tmp_path / "note.md").write_text("mine")

        path = safe.write_new_file(tmp_path, "note.md", "new")

        assert path.name == "note 2.md"
        assert path.read_text() == "new"
        assert (tmp_path / "note.md").read_text() == "mine"
        assert names(tmp_path) == ["note 2.md", "note.md"]

    def test_a_failed_write_leaves_nothing_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(fd: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", fail)

        with pytest.raises(OSError, match="disk full"):
            safe.write_new_file(tmp_path, "note.md", "new")
        assert names(tmp_path) == []

    def test_it_gives_up_when_every_numbered_name_is_taken(self, tmp_path: Path) -> None:
        for name in safe.numbered_names("note.md"):
            (tmp_path / name).touch()

        with pytest.raises(FileExistsError):
            safe.write_new_file(tmp_path, "note.md", "new")
        assert not [name for name in names(tmp_path) if name.startswith(".")]


class TestMoveNoReplace:
    @pytest.fixture
    def source(self, tmp_path: Path) -> Path:
        path = tmp_path / "inbox" / "memo.m4a"
        path.parent.mkdir()
        path.write_bytes(b"audio bytes")
        os.utime(path, (1_700_000_000, 1_700_000_000))
        return path

    @pytest.fixture
    def archive(self, tmp_path: Path) -> Path:
        path = tmp_path / "archive"
        path.mkdir()
        return path

    def test_moves_the_file(self, source: Path, archive: Path) -> None:
        moved = safe.move_no_replace(source, archive)

        assert moved == archive / "memo.m4a"
        assert moved.read_bytes() == b"audio bytes"
        assert not source.exists()

    def test_a_taken_name_gets_a_number(self, source: Path, archive: Path) -> None:
        (archive / "memo.m4a").write_bytes(b"older")

        moved = safe.move_no_replace(source, archive)

        assert moved.name == "memo 2.m4a"
        assert (archive / "memo.m4a").read_bytes() == b"older"
        assert not source.exists()

    def test_without_hard_links_it_renames_without_replacing(
        self, source: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "link", no_hard_links)
        (archive / "memo.m4a").write_bytes(b"older")

        moved = safe.move_no_replace(source, archive)

        assert moved.name == "memo 2.m4a"
        assert moved.read_bytes() == b"audio bytes"
        assert (archive / "memo.m4a").read_bytes() == b"older"
        assert not source.exists()

    def test_across_filesystems_it_copies_then_removes_the_source(
        self, source: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_link = os.link

        def cross_device(src: object, dst: object) -> None:
            if Path(str(src)) == source:
                raise OSError(errno.EXDEV, "Cross-device link")
            real_link(src, dst)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "link", cross_device)

        moved = safe.move_no_replace(source, archive)

        assert moved == archive / "memo.m4a"
        assert moved.read_bytes() == b"audio bytes"
        assert moved.stat().st_mtime == 1_700_000_000
        assert not source.exists()
        assert names(archive) == ["memo.m4a"]

    def test_a_failed_copy_keeps_the_source(
        self, source: Path, archive: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def cross_device(src: object, dst: object) -> None:
            raise OSError(errno.EXDEV, "Cross-device link")

        def fail(fd: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "link", cross_device)
        monkeypatch.setattr(os, "fsync", fail)

        with pytest.raises(OSError, match="disk full"):
            safe.move_no_replace(source, archive)
        assert source.read_bytes() == b"audio bytes"
        assert names(archive) == []


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
