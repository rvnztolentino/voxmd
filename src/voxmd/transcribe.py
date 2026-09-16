"""Stage 1: audio file in, transcript text out.

Three subprocesses, strictly in sequence:

1. ``ffprobe`` reads the real container and stream layout. This is the actual
   gate on whether a file is audio — the extension allowlist upstream is only a
   cheap pre-filter, since an extension costs nothing to fake.
2. ``ffmpeg`` normalizes to 16 kHz mono signed-16-bit PCM, which is what
   whisper.cpp consumes. **Skipped entirely** when the input already matches,
   so an already-normalized WAV pays nothing.
3. ``whisper-cli`` transcribes and writes plain text to stdout.

All three go through :func:`safe.run`: list argv, no shell, and a timeout
scaled to the audio's real duration rather than a fixed guess.

The intermediate WAV goes in a private temp directory, never beside the source.
That matters later: writing a new audio file into a watched folder would
re-trigger the watcher and process it again.

On resource discipline: :func:`transcribe` returns only after whisper has
exited and been reaped, because ``subprocess.run`` waits. Callers can therefore
rely on the whisper model being out of memory before they load anything else.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import safe
from .config import LimitsConfig, WhisperConfig
from .errors import AudioError, DependencyError, ToolFailure

AUDIO_SUFFIXES = frozenset(
    {
        ".m4a",
        ".mp3",
        ".wav",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".aiff",
        ".aif",
        ".wma",
        ".mp4",
        ".m4v",
        ".mov",
        ".webm",
        ".mkv",
    }
)

WHISPER_SAMPLE_RATE = 16_000
WHISPER_CHANNELS = 1
WHISPER_CODEC = "pcm_s16le"

FFMPEG_HINT = "ffmpeg provides both ffmpeg and ffprobe. Install it with:\n  brew install ffmpeg"
WHISPER_HINT = (
    "whisper.cpp provides the whisper-cli binary. Install it with:\n"
    "  brew install whisper-cpp\n"
    "See setup.md for the model weights, which are a separate download."
)

# whisper.cpp emits one line per segment. With -nt those are bare text, but the
# flag has been renamed across releases, so strip a timestamp prefix if one
# shows up rather than trusting it and leaking "[00:00:00.000 --> ...]" into a
# note. Keeping the text and dropping only the prefix is the safe direction.
_TIMESTAMP_PREFIX = re.compile(
    r"^\s*\[\s*\d+:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d+:\d{2}:\d{2}[.,]\d{3}\s*\]\s*"
)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Non-speech markers whisper emits for silence, music and noise. Matched only
# when the line is *entirely* an uppercase bracketed token, so ordinary speech
# that happens to contain brackets survives.
_NON_SPEECH = re.compile(r"^\[[A-Z_ ]+\]$")
_WHITESPACE = re.compile(r"\s+")

# Containers whose creation_time was never set carry 1904 or 1970 epochs.
EARLIEST_RECORDING = datetime(2000, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class AudioProbe:
    """What ffprobe found in the file."""

    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    created: datetime | None = None
    """When the recording was made, per the container's ``creation_time`` tag,
    as local time to the minute. None when absent or implausible."""

    @property
    def is_whisper_ready(self) -> bool:
        """True when whisper can read the file directly, skipping ffmpeg."""
        return (
            self.sample_rate == WHISPER_SAMPLE_RATE
            and self.channels == WHISPER_CHANNELS
            and self.codec == WHISPER_CODEC
        )


@dataclass(frozen=True)
class TranscriptionResult:
    """A finished transcription, plus what it cost to produce."""

    text: str
    source: Path
    probe: AudioProbe
    converted: bool
    """Whether ffmpeg had to run, or the input was already whisper-ready."""
    whisper_seconds: float


@functools.cache
def default_threads() -> int:
    """Thread count for whisper when config doesn't specify one.

    whisper.cpp runs best on performance cores alone. On Apple Silicon
    ``os.cpu_count()`` counts efficiency cores too, and handing those to whisper
    measurably slows the job down rather than speeding it up — work gets
    scheduled onto cores that finish late and the fast cores wait. This machine
    reports 8 logical CPUs but only 4 performance cores, so the difference is
    not academic.

    Cached because it shells out, and the answer cannot change mid-run.
    """
    if sys.platform == "darwin":
        try:
            result = safe.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                timeout=5,
                what="sysctl",
            )
            if result.returncode == 0:
                count = int(result.stdout.strip())
                if count > 0:
                    return count
        except (ValueError, DependencyError, ToolFailure):
            pass  # Fall through to the portable heuristic.

    total = os.cpu_count() or 4
    return max(1, total // 2)


def probe_audio(source: Path, *, ffprobe: Path, timeout: float) -> AudioProbe:
    """Read the first audio stream's real format.

    Raises AudioError when the file has no decodable audio stream, which is how
    a renamed non-audio file gets caught regardless of its extension.
    """
    result = safe.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "a:0",
            source,
        ],
        timeout=timeout,
        what="ffprobe",
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe could not read {source.name}:\n{_tail(result.stderr)}")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ToolFailure(f"ffprobe returned output that isn't JSON: {exc}") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise AudioError(
            f"{source.name} has no audio stream. "
            "The extension may not match the actual file contents."
        )

    stream = streams[0]
    # Duration lives on the stream for most containers and only on the format
    # for some; either can be absent for a stream-copied or truncated file.
    duration = _as_float(stream.get("duration"))
    container = payload.get("format") or {}
    if duration is None:
        duration = _as_float(container.get("duration"))

    return AudioProbe(
        duration_s=duration,
        sample_rate=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
        codec=stream.get("codec_name"),
        created=_creation_time(container) or _creation_time(stream),
    )


def transcribe(
    audio: Path | str,
    *,
    whisper: WhisperConfig,
    limits: LimitsConfig,
) -> TranscriptionResult:
    """Transcribe an audio file.

    Validation happens before any expensive work: size, real format, and
    duration are all checked before whisper is allowed to start, so a 4-hour
    recording fails in milliseconds instead of after ten minutes of CPU.
    """
    source = safe.resolve_input_file(
        audio,
        allowed_suffixes=AUDIO_SUFFIXES,
        max_bytes=limits.max_audio_bytes,
    )

    ffprobe_bin = safe.resolve_tool("ffprobe", hint=FFMPEG_HINT)
    probe = probe_audio(source, ffprobe=ffprobe_bin, timeout=limits.ffprobe_timeout_s)

    if probe.duration_s is not None and probe.duration_s > limits.max_duration_s:
        raise AudioError(
            f"{source.name} is {probe.duration_s / 60:.0f} minutes, over the "
            f"{limits.max_duration_min} minute limit.\n"
            "Raise limits.max_duration_min in your config if this is expected."
        )

    model = resolve_model(whisper)
    whisper_bin = safe.resolve_tool(whisper.binary, hint=WHISPER_HINT)
    threads = whisper.threads or default_threads()

    with tempfile.TemporaryDirectory(prefix="voxmd-") as tmpdir:
        if probe.is_whisper_ready:
            wav = source
            converted = False
        else:
            wav = Path(tmpdir) / "normalized.wav"
            _convert(source, wav, limits=limits)
            converted = True

        started = time.monotonic()
        raw = _run_whisper(
            wav,
            whisper_bin=whisper_bin,
            model=model,
            language=whisper.language,
            threads=threads,
            duration_s=probe.duration_s,
            limits=limits,
        )
        whisper_seconds = time.monotonic() - started
        # whisper has exited and been reaped by this point; the temp WAV is
        # removed on the way out of this block.

    text = clean_transcript(raw)
    if not text:
        raise ToolFailure(
            f"whisper produced no text for {source.name}. "
            "The recording may be silent or contain no speech."
        )

    return TranscriptionResult(
        text=text,
        source=source,
        probe=probe,
        converted=converted,
        whisper_seconds=whisper_seconds,
    )


def clean_transcript(raw: str) -> str:
    """Normalize whisper's stdout into one flowing paragraph.

    Segment breaks are joined rather than preserved: they fall wherever whisper
    decided to cut, which is not where a sentence or a thought ends, so keeping
    them would only add noise for both a human reader and the extraction stage.
    """
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = _ANSI.sub("", line).strip()
        stripped = _TIMESTAMP_PREFIX.sub("", stripped).strip()
        if not stripped or _NON_SPEECH.match(stripped):
            continue
        lines.append(stripped)
    return _WHITESPACE.sub(" ", " ".join(lines)).strip()


def resolve_model(whisper: WhisperConfig) -> Path:
    """Check the ggml weights exist before spending time on conversion."""
    if whisper.model is None:
        raise DependencyError(
            "No whisper model configured.\n"
            "Pass --model /path/to/ggml-large-v3-turbo.bin, or set whisper.model "
            "in your config. See setup.md for the download."
        )

    model = whisper.model.expanduser()
    if not model.is_file():
        raise DependencyError(f"Whisper model not found: {model}\nSee setup.md for the download.")
    if not os.access(model, os.R_OK):
        raise DependencyError(f"Whisper model is not readable: {model}")
    return model


def _convert(source: Path, destination: Path, *, limits: LimitsConfig) -> None:
    """Normalize to the one format whisper.cpp accepts.

    ``-nostdin`` matters: without it ffmpeg can try to read the terminal and
    swallow keystrokes, which is unpleasant in a CLI and actively wrong in a
    background worker.
    """
    ffmpeg_bin = safe.resolve_tool("ffmpeg", hint=FFMPEG_HINT)
    result = safe.run(
        [
            ffmpeg_bin,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            source,
            "-vn",  # drop cover art / video tracks
            "-map",
            "a:0",  # first audio stream only
            "-ac",
            str(WHISPER_CHANNELS),
            "-ar",
            str(WHISPER_SAMPLE_RATE),
            "-c:a",
            WHISPER_CODEC,
            "-f",
            "wav",
            destination,
        ],
        timeout=limits.ffmpeg_timeout_s,
        what="ffmpeg",
    )
    if result.returncode != 0:
        raise ToolFailure(f"ffmpeg could not convert {source.name}:\n{_tail(result.stderr)}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise ToolFailure(f"ffmpeg produced no output for {source.name}.")


def _run_whisper(
    wav: Path,
    *,
    whisper_bin: Path,
    model: Path,
    language: str,
    threads: int,
    duration_s: float | None,
    limits: LimitsConfig,
) -> str:
    """Run whisper-cli and return its raw stdout."""
    result = safe.run(
        [
            whisper_bin,
            "-m",
            model,
            "-f",
            wav,
            "-t",
            str(threads),
            "-l",
            language,
            "-nt",  # no timestamps
            "-np",  # no progress/system-info prints
        ],
        timeout=_whisper_timeout(duration_s, limits),
        what="whisper-cli",
    )
    if result.returncode != 0:
        raise ToolFailure(f"whisper-cli failed:\n{_tail(result.stderr)}")
    return result.stdout


def _whisper_timeout(duration_s: float | None, limits: LimitsConfig) -> float:
    """Scale the timeout to the audio, with a floor for short clips.

    A fixed timeout is wrong in both directions — too tight for a long meeting,
    meaningless for a 30-second memo. When ffprobe couldn't determine duration,
    fall back to the ceiling implied by max_duration_min rather than guessing
    small and killing a legitimate job.
    """
    if duration_s is None:
        return max(
            limits.whisper_timeout_floor_s, limits.max_duration_s * limits.whisper_timeout_factor
        )
    return max(limits.whisper_timeout_floor_s, duration_s * limits.whisper_timeout_factor)


def _tail(text: str, *, limit: int = 500) -> str:
    """Last chunk of a tool's stderr, for an error message."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "(no error output)"
    return cleaned if len(cleaned) <= limit else "..." + cleaned[-limit:]


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _creation_time(section: object) -> datetime | None:
    """A ``creation_time`` tag as local time, or None if absent or implausible.

    ffmpeg writes it in UTC; a value without an offset is read as UTC too.
    Anything before 2000 or more than a day in the future is ignored, rather
    than dating a memo decades back.
    """
    tags = section.get("tags") if isinstance(section, dict) else None
    value = tags.get("creation_time") if isinstance(tags, dict) else None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if not EARLIEST_RECORDING <= parsed <= datetime.now(UTC) + timedelta(days=1):
        return None
    return parsed.astimezone().replace(tzinfo=None, second=0, microsecond=0)


def _as_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
