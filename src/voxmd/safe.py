"""Shared safety primitives.

Everything in voxmd that spawns a process or accepts a caller-supplied file
path goes through this module, so the policy lives in one auditable place
instead of being re-derived (and eventually gotten wrong) at each call site.

Two rules are enforced here and nowhere else:

* **No shell, ever.** Subprocesses take a list argv. A memo named
  ``; rm -rf ~ .m4a`` is then just a file with an odd name, because the string
  never reaches a shell that could parse it.
* **No unbounded wait.** Every subprocess carries a timeout. A wedged decoder
  on a corrupt file must not hang the CLI, and later must not wedge the
  watcher's single worker thread.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import IO

from .errors import AudioError, DependencyError, InputError, ToolFailure, ToolTimeout


def human_bytes(n: int) -> str:
    """Format a byte count for a human-facing error message."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - unreachable, loop always returns


def resolve_tool(name: str, *, hint: str) -> Path:
    """Locate an external binary, or fail with a message that says what to do.

    Accepts either a bare command name to look up on PATH, or an explicit path
    from config for a non-standard install location.
    """
    candidate = Path(name).expanduser()
    if candidate.is_absolute() or os.sep in name:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise DependencyError(f"{name} is not an executable file.\n{hint}")

    found = shutil.which(name)
    if found is None:
        raise DependencyError(f"{name!r} was not found on PATH.\n{hint}")
    return Path(found)


def run(
    argv: Sequence[str | Path],
    *,
    timeout: float,
    what: str,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess under voxmd's fixed safety policy.

    ``what`` is a human label used in timeout messages, e.g. ``"whisper-cli"``.

    Returns the completed process without checking the return code — callers
    decide what a non-zero exit means for them. It is guaranteed to have
    exited: ``subprocess.run`` waits and reaps, which is what lets the pipeline
    later guarantee whisper is gone before Ollama is asked to load a model.
    """
    args = [str(a) for a in argv]
    try:
        # S603/S607: this is the one sanctioned subprocess call in the codebase.
        # argv is a list, shell is explicitly False, stdin is closed so nothing
        # can block on input we will never send, and timeout is mandatory.
        return subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeout(f"{what} exceeded its {timeout:.0f}s timeout and was killed.") from exc
    except FileNotFoundError as exc:  # pragma: no cover - resolve_tool runs first
        raise DependencyError(f"{what} could not be executed: {exc}") from exc
    except OSError as exc:
        raise ToolFailure(f"{what} could not be started: {exc}") from exc


def resolve_input_file(
    path: Path | str,
    *,
    allowed_suffixes: Iterable[str],
    max_bytes: int,
    error: type[InputError] = AudioError,
    limit_setting: str = "limits.max_audio_mb",
) -> Path:
    """Validate a caller-supplied input file and return its resolved path.

    ``error`` and ``limit_setting`` let non-audio inputs (transcripts) reuse the
    same checks with their own error type and config key.

    Checks are ordered cheapest-first so an obviously wrong input fails before
    anything expensive happens.

    The suffix allowlist is a pre-filter, not a security control — an extension
    costs nothing to fake. ``ffprobe`` downstream is the real gate on whether a
    file is actually decodable audio. What this function genuinely protects
    against is handing a directory, FIFO, socket, or device node to ffmpeg: a
    FIFO in particular would block a read forever, and the timeout in ``run``
    is a backstop, not a substitute for not doing it.
    """
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        # RuntimeError covers a symlink loop.
        raise error(f"No such file: {candidate}") from exc

    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise error(f"Not a regular file: {resolved}")

    suffixes = {s.lower() for s in allowed_suffixes}
    if resolved.suffix.lower() not in suffixes:
        supported = ", ".join(sorted(suffixes))
        raise error(
            f"Unsupported file type {resolved.suffix or '(none)'!r}: {resolved.name}\n"
            f"Supported: {supported}"
        )

    if info.st_size == 0:
        raise error(f"File is empty: {resolved}")

    if info.st_size > max_bytes:
        raise error(
            f"{resolved.name} is {human_bytes(info.st_size)}, over the "
            f"{human_bytes(max_bytes)} limit.\n"
            f"Raise {limit_setting} in your config if this is expected."
        )

    if not os.access(resolved, os.R_OK):
        raise error(f"Not readable: {resolved}")

    return resolved


def read_text_input(
    source: Path | str | None,
    *,
    allowed_suffixes: Iterable[str],
    max_bytes: int,
    stdin: IO[str] | None,
    what: str,
    limit_setting: str,
    no_input_message: str,
) -> str:
    """Read UTF-8 text from a file, or from stdin when ``source`` is None or ``-``.

    ``what`` names the input in messages ("transcript", "extraction"). Reads are
    bounded even after the size check, because a file can grow in between and
    a pipe has no size to check.
    """
    if source is None or str(source) == "-":
        data = _read_stdin(stdin, max_bytes, no_input_message)
        label = "stdin"
    else:
        path = resolve_input_file(
            source,
            allowed_suffixes=allowed_suffixes,
            max_bytes=max_bytes,
            error=InputError,
            limit_setting=limit_setting,
        )
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        label = path.name

    if len(data) > max_bytes:
        raise InputError(
            f"{label} is over the {human_bytes(max_bytes)} {what} limit.\n"
            f"Raise {limit_setting} in your config if this is expected."
        )
    if b"\x00" in data:
        raise InputError(f"{label} looks like binary data, not a {what}.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError(f"{label} is not UTF-8 text.") from exc


def _read_stdin(stdin: IO[str] | None, max_bytes: int, no_input_message: str) -> bytes:
    if stdin is None or stdin.isatty():
        # Reading a terminal would sit waiting for input the user doesn't know
        # is expected.
        raise InputError(no_input_message)
    buffer = getattr(stdin, "buffer", None)
    if buffer is not None:
        return buffer.read(max_bytes + 1)
    return stdin.read(max_bytes + 1).encode("utf-8")


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Replace ``path`` so readers see the old file or the new one, never half of either.

    The temp file sits beside the target, so the rename stays on one filesystem
    and is atomic. It is flushed to disk before the rename, and the directory
    after it, so a crash can't leave an empty file behind. Mode defaults to
    0600: voxmd's files hold names and memo content.
    """
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(mode)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


MAX_NAME_ATTEMPTS = 100
COPY_CHUNK_BYTES = 1024 * 1024
# Filesystems without hard links (exFAT, some network shares) report these.
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK})


def numbered_names(name: str, *, limit: int = MAX_NAME_ATTEMPTS) -> Iterator[str]:
    """``name``, then ``stem 2.suffix``, ``stem 3.suffix``, up to ``limit`` names."""
    path = Path(name)
    yield name
    for number in range(2, limit + 1):
        yield f"{path.stem} {number}{path.suffix}"


def write_new_file(directory: Path, name: str, text: str, *, mode: int = 0o600) -> Path:
    """Write ``text`` to a new file named ``name`` or a numbered variant. Never replaces one.

    The content is written and synced under a hidden temp name first, so
    anything watching the directory (Obsidian) never sees a half-written file.
    Returns the path it landed at.
    """
    fd, temp_name = tempfile.mkstemp(prefix=".voxmd-", suffix=".tmp", dir=directory)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(mode)
        target = _place(temp, directory, name)
    finally:
        temp.unlink(missing_ok=True)
    _fsync_dir(directory)
    return target


def move_no_replace(source: Path, directory: Path) -> Path:
    """Move ``source`` into ``directory`` under its own name or a numbered variant.

    Never replaces an existing file, and the source is removed only once the
    file is in place. Across filesystems it is copied, synced, and size-checked
    first. Returns the new path.
    """
    try:
        if source.stat().st_dev == directory.stat().st_dev:
            target = _place(source, directory, source.name)
            source.unlink(missing_ok=True)
            _fsync_dir(directory)
            _fsync_dir(source.parent)
            return target
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    fd, temp_name = tempfile.mkstemp(prefix=".voxmd-", suffix=".tmp", dir=directory)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as copy, source.open("rb") as original:
            shutil.copyfileobj(original, copy, COPY_CHUNK_BYTES)
            copy.flush()
            os.fsync(copy.fileno())
        shutil.copystat(source, temp)
        if temp.stat().st_size != source.stat().st_size:
            raise OSError(errno.EIO, "the copy is incomplete", str(temp))
        target = _place(temp, directory, source.name)
    finally:
        temp.unlink(missing_ok=True)
    _fsync_dir(directory)
    source.unlink()
    return target


def _place(file: Path, directory: Path, name: str) -> Path:
    """Give ``file`` a name in ``directory`` that no other file has. Never replaces one.

    A hard link fails when the name is taken, so claiming a name is atomic; the
    caller then removes ``file``. Where hard links aren't supported, the name is
    reserved by an exclusive create and ``file`` is renamed over that
    reservation, so the only thing ever replaced is voxmd's own empty placeholder.
    """
    for candidate in numbered_names(name):
        target = directory / candidate
        try:
            os.link(file, target)
            return target
        except FileExistsError:
            continue
        except OSError as exc:
            if exc.errno not in _NO_HARD_LINKS:
                raise
        try:
            reserved = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            continue
        os.close(reserved)
        file.replace(target)
        return target
    raise FileExistsError(
        errno.EEXIST,
        f"{MAX_NAME_ATTEMPTS} files with this name already exist",
        str(directory / name),
    )


def _fsync_dir(directory: Path) -> None:
    """Flush a directory entry change to disk. Best effort: not every filesystem allows it."""
    with contextlib.suppress(OSError):
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


@contextlib.contextmanager
def file_lock(path: Path, *, timeout_s: float, what: str) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``path``, created 0600 if missing.

    Waits at most ``timeout_s``: the holder is another short voxmd run, and one
    that hung must not hang this one too. The retry loop only runs while the
    lock is actually contended. ``O_NOFOLLOW`` stops a planted symlink from
    making voxmd create a file somewhere else.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ToolTimeout(
                        f"{what} is locked by another voxmd process; "
                        f"gave up after {timeout_s:.0f}s."
                    ) from None
                time.sleep(0.05)
        yield
    finally:
        os.close(fd)  # Closing the descriptor releases the lock.
