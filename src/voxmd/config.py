"""Configuration.

One YAML file, read with ``yaml.safe_load`` and validated by pydantic.
Resolution order, first hit wins:

1. ``$VOXMD_CONFIG``
2. ``./voxmd.yaml``
3. ``~/.config/voxmd/config.yaml``

Config is **optional** at this stage. ``voxmd transcribe`` runs from CLI flags
alone, so trying stage 1 on a real memo doesn't require writing a config file
first. Sections are added by the stages that need them rather than being
declared up front, so this file grows alongside the pipeline.

``safe_load`` rather than ``load`` is not a stylistic choice: full-fat YAML can
construct arbitrary Python objects, which would turn "edit your config" into
"execute this file".
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError

CONFIG_ENV_VAR = "VOXMD_CONFIG"
LOCAL_CONFIG_NAME = "voxmd.yaml"
USER_CONFIG_PATH = Path("~/.config/voxmd/config.yaml")

# "auto", or an ISO 639 code with an optional region: en, yue, pt-br.
_LANGUAGE_CODE = re.compile(r"auto|[a-z]{2,3}(-[a-z]{2,4})?")


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

    @property
    def max_audio_bytes(self) -> int:
        return self.max_audio_mb * 1024 * 1024

    @property
    def max_duration_s(self) -> float:
        return self.max_duration_min * 60


class Config(BaseModel):
    """The whole config file.

    ``extra="forbid"`` turns a typo'd key into an immediate error instead of a
    setting that silently does nothing.
    """

    model_config = ConfigDict(extra="forbid")

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
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


def apply_overrides(whisper: WhisperConfig, **overrides: object) -> WhisperConfig:
    """Layer CLI flags over config, validated exactly as config values are.

    ``model_copy(update=...)`` would skip validation entirely, letting a flag
    such as ``--language=--help`` through checks the same value fails in a
    file. ``None`` means "flag not given" and leaves the config value alone.
    """
    merged = whisper.model_dump()
    merged.update({key: value for key, value in overrides.items() if value is not None})
    try:
        return WhisperConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid option:\n{_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    """Render pydantic's error list as something readable in a terminal."""
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"  {location}: {err['msg']}")
    return "\n".join(lines)
