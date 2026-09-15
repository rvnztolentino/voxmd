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

import os
import shutil
import stat
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

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
