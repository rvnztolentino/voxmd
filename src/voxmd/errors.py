"""Error types.

Every failure a user can plausibly cause or fix gets a type here with a message
written for them, not for a stack trace. ``cli.main`` catches ``VoxmdError`` and
prints just the message; anything else escapes with its traceback intact,
because an unexpected exception is a bug and hiding it helps nobody.

Exit codes are distinct so shell callers can branch on the failure kind.
"""

from __future__ import annotations


class VoxmdError(Exception):
    """Base for every expected, user-actionable failure."""

    exit_code = 1


class ConfigError(VoxmdError):
    """Config file is missing, malformed, or holds an invalid value."""

    exit_code = 2


class DependencyError(VoxmdError):
    """A required external tool or model file is missing or unusable.

    Raised for things the setup guide is supposed to have installed: ffmpeg,
    ffprobe, whisper-cli, the ggml model weights, a running Ollama and its model.
    """

    exit_code = 3


class InputError(VoxmdError):
    """The input is missing, the wrong kind of file, unreadable, or over a limit."""

    exit_code = 4


class AudioError(InputError):
    """An audio input is missing, not really audio, unreadable, or over a limit."""

    exit_code = 4


class ToolFailure(VoxmdError):
    """An external tool exited non-zero, or produced nothing usable."""

    exit_code = 5


class ToolTimeout(ToolFailure):
    """An external tool exceeded its timeout and was killed."""

    exit_code = 6


class OutputError(VoxmdError):
    """A note, ledger, or state file could not be written. The recording is untouched."""

    exit_code = 7


class PartialFailure(VoxmdError):
    """The note was written, but a later step (ledger, entities, archive) failed."""

    exit_code = 8
