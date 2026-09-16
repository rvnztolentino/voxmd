"""Configuration.

One YAML file, read with ``yaml.safe_load`` and validated by pydantic.
Resolution order, first hit wins:

1. ``$VOXMD_CONFIG``
2. ``./voxmd.yaml``
3. ``~/.config/voxmd/config.yaml``

Config is **optional** for the single-stage commands: ``voxmd transcribe``,
``voxmd extract`` and ``voxmd render`` run from CLI flags and defaults alone, so
trying a stage on a real memo doesn't require writing a config file first.
``voxmd process`` needs a vault, from ``vault.path`` or ``--vault``. Sections
are added by the stages that need them rather than being declared up front, so
this file grows alongside the pipeline.

``safe_load`` rather than ``load`` is not a stylistic choice: full-fat YAML can
construct arbitrary Python objects, which would turn "edit your config" into
"execute this file".
"""

from __future__ import annotations

import ipaddress
import os
import re
import urllib.parse
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import ConfigError

CONFIG_ENV_VAR = "VOXMD_CONFIG"
LOCAL_CONFIG_NAME = "voxmd.yaml"
USER_CONFIG_PATH = Path("~/.config/voxmd/config.yaml")

OLLAMA_DEFAULT_PORT = 11434

# "auto", or an ISO 639 code with an optional region: en, yue, pt-br.
_LANGUAGE_CODE = re.compile(r"auto|[a-z]{2,3}(-[a-z]{2,4})?")
# Ollama model references: qwen3:8b, library/llama3.2:3b, hf.co/user/repo:Q4_K_M.
_OLLAMA_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-/:]{0,199}")

# Room the context window must leave for the instructions and the transcript,
# beyond the reply itself.
_MIN_PROMPT_ROOM_TOKENS = 2048

_Section = TypeVar("_Section", bound=BaseModel)


class WhisperConfig(BaseModel):
    """whisper.cpp invocation settings."""

    model_config = ConfigDict(extra="forbid")

    binary: str = "whisper-cli"
    """Command name or absolute path. Homebrew's whisper-cpp installs
    ``whisper-cli``; older builds called it ``main``."""

    model: Path | None = None
    """Path to the ggml weights, e.g. ``ggml-large-v3-turbo.bin``. No default —
    the file is a multi-gigabyte manual download, so guessing a path would only
    produce a confusing error later."""

    language: str = "auto"
    """ISO code such as ``en``, or ``auto`` to let whisper detect it."""

    threads: int | None = Field(default=None, ge=1)
    """``None`` means detect performance cores. See ``transcribe.default_threads``."""

    @field_validator("model")
    @classmethod
    def _expand_model(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @field_validator("language")
    @classmethod
    def _clean_language(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("language must not be empty (use 'auto' to detect)")
        # Passed straight to whisper-cli's -l flag. A value starting with "-"
        # would be parsed by whisper as another flag, so accept only the shape
        # of a real language code rather than anything alphabetic.
        if not _LANGUAGE_CODE.fullmatch(cleaned):
            raise ValueError(f"not a language code: {value!r}")
        return cleaned


class OllamaConfig(BaseModel):
    """Ollama extraction settings.

    Two things are deliberately **not** configurable: ``keep_alive`` is always 0
    (the model is never left resident) and thinking is always off. Both are
    constraints, not preferences.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}"
    """Must be loopback. Transcripts are never sent to another machine."""

    model: str = "qwen3:8b"

    temperature: float = Field(default=0.1, ge=0, le=2)
    """Low: extraction should be faithful to the memo, not creative."""

    num_ctx: int = Field(default=16384, ge=4096, le=131072)
    """Context window **ceiling**, in tokens. Each call is sized to its
    transcript, so short memos use less memory than this."""

    num_predict: int = Field(default=2048, ge=256, le=8192)
    """Maximum reply length, in tokens."""

    @field_validator("host")
    @classmethod
    def _require_loopback(cls, value: str) -> str:
        return normalize_loopback_host(value)

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not _OLLAMA_MODEL.fullmatch(cleaned):
            raise ValueError(f"not an Ollama model name: {value!r}")
        return cleaned

    @model_validator(mode="after")
    def _leave_room_for_the_prompt(self) -> OllamaConfig:
        if self.num_ctx < self.num_predict + _MIN_PROMPT_ROOM_TOKENS:
            raise ValueError(
                f"num_ctx ({self.num_ctx}) must be at least num_predict + "
                f"{_MIN_PROMPT_ROOM_TOKENS} ({self.num_predict + _MIN_PROMPT_ROOM_TOKENS}), "
                "or no room is left for the transcript"
            )
        return self


def normalize_loopback_host(value: str) -> str:
    """Validate an Ollama host as loopback and return it as ``scheme://host:port``.

    Refused rather than warned about: voxmd's only permitted network traffic is
    to a local Ollama, and Ollama has no authentication, so a remote host would
    send every transcript to another machine in the clear.
    """
    raw = value.strip()
    if not raw:
        raise ValueError("host must not be empty")

    parts = urllib.parse.urlsplit(raw if "://" in raw else f"http://{raw}")
    if parts.scheme not in {"http", "https"}:
        raise ValueError(f"host must use http or https, got {parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("host must not contain credentials")
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise ValueError("host must not include a path, query, or fragment")
    try:
        port = parts.port or OLLAMA_DEFAULT_PORT
    except ValueError as exc:
        raise ValueError(f"invalid port in host {value!r}") from exc

    hostname = parts.hostname or ""
    if not _is_loopback(hostname):
        raise ValueError(
            f"host must be loopback (127.0.0.1, ::1, or localhost), got {hostname!r}. "
            "voxmd only sends transcripts to a local Ollama."
        )
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parts.scheme}://{netloc}:{port}"


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class RenderConfig(BaseModel):
    """Note rendering settings."""

    model_config = ConfigDict(extra="forbid")

    template: Path | None = None
    """A Jinja2 note template. ``None`` uses the built-in ``note.md.j2``: copy
    that file and point this at the copy to change the note layout."""

    @field_validator("template")
    @classmethod
    def _expand_template(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None


class EntitiesConfig(BaseModel):
    """Known people and topics, used to build [[wikilinks]]."""

    model_config = ConfigDict(extra="forbid")

    file: Path = Field(default=Path("~/.config/voxmd/entities.json"), validate_default=True)
    """A missing file is fine: nothing is linked until names are added."""

    fuzzy_threshold: int = Field(default=90, ge=50, le=100)
    """Similarity (0-100) at which a name counts as a known one. 90 merges
    "Christopher"/"Christophor" but keeps "Marco"/"Marcus" apart."""

    @field_validator("file")
    @classmethod
    def _expand_file(cls, value: Path) -> Path:
        return value.expanduser()


def _absolute(value: Path, setting: str) -> Path:
    """A write destination must not depend on the directory voxmd runs from."""
    expanded = value.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{setting} must be an absolute path or start with ~, got {str(value)!r}")
    return expanded


class VaultConfig(BaseModel):
    """Where ``voxmd process`` writes notes."""

    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    """The Obsidian vault. It must already exist: voxmd never creates it, so a
    typo fails instead of quietly starting a new folder somewhere."""

    folder: Path | None = None
    """Folder inside the vault for new notes, created if missing. ``None`` is
    the vault root."""

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: Path | None) -> Path | None:
        return _absolute(value, "vault.path") if value is not None else None

    @field_validator("folder")
    @classmethod
    def _check_folder(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if value.is_absolute() or str(value).startswith("~") or ".." in value.parts:
            raise ValueError("vault.folder must be a relative path inside the vault, without '..'")
        return value if value.parts else None


class ArchiveConfig(BaseModel):
    """Where processed recordings go."""

    model_config = ConfigDict(extra="forbid")

    dir: Path | None = None
    """Recordings move here once their note is written. ``None`` leaves them
    where they are. Created if missing, but its parent must exist."""

    @field_validator("dir")
    @classmethod
    def _check_dir(cls, value: Path | None) -> Path | None:
        return _absolute(value, "archive.dir") if value is not None else None


class StateConfig(BaseModel):
    """voxmd's own bookkeeping: the ledger of processed recordings."""

    model_config = ConfigDict(extra="forbid")

    dir: Path = Field(default=Path("~/.local/state/voxmd"), validate_default=True)

    @field_validator("dir")
    @classmethod
    def _check_dir(cls, value: Path) -> Path:
        return _absolute(value, "state.dir")


class LimitsConfig(BaseModel):
    """Ceilings and timeouts.

    These exist so a pathological input fails fast instead of quietly occupying
    the machine for an hour. Timeouts scale with audio duration rather than
    being fixed, because a fixed number is either too tight for a long meeting
    or useless for a 30-second memo.
    """

    model_config = ConfigDict(extra="forbid")

    max_audio_mb: int = Field(default=500, ge=1)
    max_duration_min: int = Field(default=180, ge=1)
    ffprobe_timeout_s: float = Field(default=30.0, gt=0)
    ffmpeg_timeout_s: float = Field(default=600.0, gt=0)
    whisper_timeout_floor_s: float = Field(default=300.0, gt=0)
    whisper_timeout_factor: float = Field(default=3.0, gt=0)
    """Multiplier on audio duration. large-v3-turbo runs faster than realtime on
    Apple Silicon, so 3x is generous headroom rather than a tight budget."""

    max_transcript_kb: int = Field(default=512, ge=1)
    """Largest transcript ``voxmd extract`` will read. 512 KB is days of speech."""
    ollama_timeout_s: float = Field(default=600.0, gt=0)
    """Whole request, model load included. keep_alive=0 means every memo loads."""
    ollama_connect_timeout_s: float = Field(default=5.0, gt=0)

    max_extraction_kb: int = Field(default=256, ge=1)
    """Largest extraction JSON ``voxmd render`` will read."""
    max_template_kb: int = Field(default=64, ge=1)
    max_entities_kb: int = Field(default=2048, ge=1)
    """Largest entities.json. 2 MB holds tens of thousands of names."""
    max_ledger_mb: int = Field(default=16, ge=1)
    """Largest ledger. An entry is a few hundred bytes, so 16 MB is decades of memos."""

    @property
    def max_ledger_bytes(self) -> int:
        return self.max_ledger_mb * 1024 * 1024

    @property
    def max_extraction_bytes(self) -> int:
        return self.max_extraction_kb * 1024

    @property
    def max_template_bytes(self) -> int:
        return self.max_template_kb * 1024

    @property
    def max_entities_bytes(self) -> int:
        return self.max_entities_kb * 1024

    @property
    def max_audio_bytes(self) -> int:
        return self.max_audio_mb * 1024 * 1024

    @property
    def max_duration_s(self) -> float:
        return self.max_duration_min * 60

    @property
    def max_transcript_bytes(self) -> int:
        return self.max_transcript_kb * 1024


class Config(BaseModel):
    """The whole config file.

    ``extra="forbid"`` turns a typo'd key into an immediate error instead of a
    setting that silently does nothing.
    """

    model_config = ConfigDict(extra="forbid")

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    entities: EntitiesConfig = Field(default_factory=EntitiesConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)


def config_search_path() -> list[Path]:
    """Candidate config locations, highest precedence first."""
    candidates: list[Path] = []
    from_env = os.environ.get(CONFIG_ENV_VAR)
    if from_env:
        candidates.append(Path(from_env).expanduser())
    candidates.append(Path.cwd() / LOCAL_CONFIG_NAME)
    candidates.append(USER_CONFIG_PATH.expanduser())
    return candidates


def find_config() -> Path | None:
    """First existing config file, or None if the user hasn't written one."""
    for candidate in config_search_path():
        if candidate.is_file():
            return candidate
    return None


def load_config(explicit: Path | str | None = None) -> Config:
    """Load config, falling back to defaults when no file exists.

    An explicitly requested path that doesn't exist is an error — the user
    asked for that file specifically, and silently ignoring it would hide a
    typo behind surprising behaviour.
    """
    if explicit is not None:
        path: Path | None = Path(explicit).expanduser()
        if path is not None and not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
    else:
        path = find_config()

    if path is None:
        return Config()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML:\n{exc}") from exc

    if raw is None:
        return Config()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level, got {type(raw).__name__}."
        )

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config in {path}:\n{_format_errors(exc)}") from exc


def apply_overrides(section: _Section, **overrides: object) -> _Section:
    """Layer CLI flags over a config section, validated exactly as config values are.

    ``model_copy(update=...)`` would skip validation entirely, letting a flag
    such as ``--language=--help`` through checks the same value fails in a
    file. ``None`` means "flag not given" and leaves the config value alone.
    """
    merged = section.model_dump()
    merged.update({key: value for key, value in overrides.items() if value is not None})
    try:
        return type(section).model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid option:\n{_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    """Render pydantic's error list as something readable in a terminal."""
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"  {location}: {err['msg']}")
    return "\n".join(lines)
