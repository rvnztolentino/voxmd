"""``voxmd doctor``: check every prerequisite, and change nothing.

Read-only by design. It never installs, pulls, downloads, or creates anything.
It reports what is missing and how to fix it. Its one network request goes to
the configured Ollama host, which config has already confirmed is loopback.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama

from . import extract, safe
from .config import Config, find_config, load_config
from .entities import load_entities
from .errors import ConfigError, VoxmdError
from .ledger import LEDGER_NAME, Ledger
from .pipeline import resolve_archive_dir, resolve_notes_dir
from .render import load_template
from .transcribe import FFMPEG_HINT, WHISPER_HINT, resolve_model

OK = "ok"
WARN = "warn"
FAIL = "FAIL"

OLLAMA_CHECK_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def run_checks(config: Path | None = None, *, client: object = None) -> list[Check]:
    """Every check, in the order a new setup would fix them."""
    try:
        settings = load_config(config)
    except ConfigError as exc:
        return [Check("config", FAIL, str(exc))]
    location = Path(config).expanduser() if config is not None else find_config()
    checks = [Check("config", OK, str(location) if location else "none found; using defaults")]

    checks.append(_guard("ffprobe", lambda: _tool("ffprobe", FFMPEG_HINT)))
    checks.append(_guard("ffmpeg", lambda: _tool("ffmpeg", FFMPEG_HINT)))
    checks.append(_guard("whisper-cli", lambda: _tool(settings.whisper.binary, WHISPER_HINT)))
    checks.append(_guard("whisper model", lambda: _whisper_model(settings)))
    checks.extend(_ollama(settings, client))
    checks.append(_guard("vault", lambda: _vault(settings)))
    checks.append(_guard("archive", lambda: _archive(settings)))
    checks.append(_guard("state", lambda: _state(settings)))
    checks.append(_guard("template", lambda: _template(settings)))
    checks.append(_guard("entities", lambda: _entities(settings)))
    return checks


def model_installed(wanted: str, installed: set[str]) -> bool:
    """Whether ``wanted`` is pulled. ``llama3.2`` means ``llama3.2:latest``."""
    if wanted in installed:
        return True
    return ":" not in wanted.rsplit("/", 1)[-1] and f"{wanted}:latest" in installed


def _guard(name: str, check: Callable[[], tuple[str, str]]) -> Check:
    try:
        status, detail = check()
    except VoxmdError as exc:
        return Check(name, FAIL, str(exc))
    return Check(name, status, detail)


def _tool(binary: str, hint: str) -> tuple[str, str]:
    return OK, str(safe.resolve_tool(binary, hint=hint))


def _whisper_model(settings: Config) -> tuple[str, str]:
    model = resolve_model(settings.whisper)
    return OK, f"{model} ({safe.human_bytes(model.stat().st_size)})"


def _ollama(settings: Config, client: object) -> list[Check]:
    cfg = settings.ollama
    owns_client = client is None
    if client is None:
        client = extract.make_client(
            cfg.host,
            timeout_s=OLLAMA_CHECK_TIMEOUT_S,
            connect_timeout_s=settings.limits.ollama_connect_timeout_s,
        )
    try:
        installed = client.list()  # type: ignore[attr-defined]
        loaded = client.ps()  # type: ignore[attr-defined]
    except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
        return [
            Check(
                "ollama",
                FAIL,
                f"not reachable at {cfg.host} ({type(exc).__name__}).\n{extract.OLLAMA_HINT}",
            )
        ]
    finally:
        if owns_client:
            client.close()  # type: ignore[attr-defined]

    checks = [Check("ollama", OK, f"reachable at {cfg.host}")]
    names = {model.model for model in installed.models if model.model}
    if model_installed(cfg.model, names):
        checks.append(Check("ollama model", OK, cfg.model))
    else:
        checks.append(
            Check(
                "ollama model",
                FAIL,
                f"{cfg.model} is not pulled. Pull it with:\n  ollama pull {cfg.model}",
            )
        )
    if loaded.models:
        checks.append(
            Check(
                "ollama memory",
                WARN,
                f"{len(loaded.models)} model(s) loaded right now. voxmd unloads its own after "
                "each memo; these belong to something else.",
            )
        )
    else:
        checks.append(Check("ollama memory", OK, "no model loaded"))
    return checks


def _vault(settings: Config) -> tuple[str, str]:
    notes, root = resolve_notes_dir(settings.vault)
    if not os.access(root, os.W_OK):
        return FAIL, f"{root} is not writable"
    if not notes.exists():
        return OK, f"{notes} (folder is created on first process)"
    if not notes.is_dir() or not os.access(notes, os.W_OK):
        return FAIL, f"{notes} is not a writable folder"
    return OK, str(notes)


def _archive(settings: Config) -> tuple[str, str]:
    directory = resolve_archive_dir(settings.archive)
    if directory is None:
        return OK, "not set; recordings stay where they are"
    target = directory if directory.exists() else directory.parent
    if not os.access(target, os.W_OK):
        return FAIL, f"{target} is not writable"
    suffix = "" if directory.exists() else " (created on first process)"
    return OK, f"{directory}{suffix}"


def _state(settings: Config) -> tuple[str, str]:
    directory = settings.state.dir
    if not directory.exists():
        existing = next((p for p in directory.parents if p.exists()), None)
        if existing is None or not os.access(existing, os.W_OK):
            return FAIL, f"{directory} can't be created: {existing} is not writable"
        return OK, f"{directory} (created on first process)"
    if not directory.is_dir() or not os.access(directory, os.W_OK):
        return FAIL, f"{directory} is not a writable folder"
    ledger = Ledger.load(directory / LEDGER_NAME, max_bytes=settings.limits.max_ledger_bytes)
    return OK, f"{directory} ({len(ledger)} recordings processed)"


def _template(settings: Config) -> tuple[str, str]:
    _, label = load_template(settings.render.template, max_bytes=settings.limits.max_template_bytes)
    return OK, label


def _entities(settings: Config) -> tuple[str, str]:
    known = load_entities(
        settings.entities.file,
        max_bytes=settings.limits.max_entities_bytes,
        threshold=settings.entities.fuzzy_threshold,
    )
    if not known.exists:
        return OK, f"{known.path} (not found; created on first process)"
    return OK, f"{known.path} ({len(known.people)} people, {len(known.topics)} topics)"
