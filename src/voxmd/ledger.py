"""Which recordings have been processed, so one memo never makes two notes.

Entries are keyed by the SHA-256 of the audio bytes, not the file's name or
path. A memo that is renamed, moved, or synced again is still recognized, and
two different memos that share a name are never confused. Hashing streams the
file in chunks, so a long recording is never read into memory.

The ledger holds metadata only: where the recording was, where its note went,
when, and how long it was. Never transcript text or model output. It is
written atomically at 0600. A corrupt ledger is refused rather than reset,
because an empty ledger would reprocess, and duplicate, every recording.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import safe
from .errors import ConfigError, InputError, OutputError
from .schema import describe_errors

LEDGER_NAME = "ledger.json"
HASH_CHUNK_BYTES = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")


class LedgerEntry(BaseModel):
    """One processed recording."""

    model_config = ConfigDict(extra="forbid")

    source: str
    """Where the recording was when it was processed."""
    size: int = Field(ge=0)
    processed_at: datetime
    note: str
    archived: str | None = None
    duration_s: float | None = None


class LedgerFile(BaseModel):
    """The shape of ``ledger.json``."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    entries: dict[str, LedgerEntry] = Field(default_factory=dict)

    @field_validator("entries")
    @classmethod
    def _check_keys(cls, value: dict[str, LedgerEntry]) -> dict[str, LedgerEntry]:
        if any(not _DIGEST.fullmatch(key) for key in value):
            raise ValueError("keys must be lowercase SHA-256 hex digests")
        return value


def hash_file(path: Path, *, max_bytes: int) -> str:
    """SHA-256 of a file, read in chunks and never past ``max_bytes``."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            total += len(chunk)
            if total > max_bytes:
                raise InputError(
                    f"{path.name} grew past the {safe.human_bytes(max_bytes)} limit while "
                    "being read. If it is still syncing, wait and run again."
                )
            digest.update(chunk)
    return digest.hexdigest()


class Ledger:
    """The ledger file, loaded. Callers serialize access with the pipeline lock."""

    def __init__(self, path: Path, document: LedgerFile, *, max_bytes: int) -> None:
        self.path = path
        self._document = document
        self._max_bytes = max_bytes

    @classmethod
    def load(cls, path: Path, *, max_bytes: int) -> Ledger:
        """Read and validate the ledger. A missing file is an empty ledger."""
        return cls(path, _read(path, max_bytes), max_bytes=max_bytes)

    def __len__(self) -> int:
        return len(self._document.entries)

    def get(self, digest: str) -> LedgerEntry | None:
        return self._document.entries.get(digest)

    def record(self, digest: str, entry: LedgerEntry) -> None:
        """Add or replace one entry, and write the whole ledger atomically."""
        if not _DIGEST.fullmatch(digest):
            raise ValueError("digest must be a lowercase SHA-256 hex digest")
        self._document.entries[digest] = entry
        text = self._document.model_dump_json(indent=2) + "\n"
        if len(text.encode("utf-8")) > self._max_bytes:
            raise OutputError(
                f"{self.path} would grow past the {safe.human_bytes(self._max_bytes)} limit.\n"
                "Raise limits.max_ledger_mb in your config."
            )
        try:
            safe.atomic_write_text(self.path, text)
        except OSError as exc:
            raise OutputError(f"Could not write {self.path}: {exc.strerror or exc}") from exc


def _read(path: Path, max_bytes: int) -> LedgerFile:
    try:
        info = path.stat()
    except FileNotFoundError:
        return LedgerFile()
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc.strerror or exc}") from exc

    # Checked before opening: opening a FIFO would block.
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError(f"Ledger is not a regular file: {path}")
    if info.st_size > max_bytes:
        raise ConfigError(
            f"{path} is {safe.human_bytes(info.st_size)}, over the "
            f"{safe.human_bytes(max_bytes)} limit.\n"
            "Raise limits.max_ledger_mb in your config if this is expected."
        )
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc.strerror or exc}") from exc

    refuse = "Fix it or move it aside; voxmd won't overwrite it."
    try:
        raw = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path} is not UTF-8 text. {refuse}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}). {refuse}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a JSON object. {refuse}")
    try:
        return LedgerFile.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid ledger {path} ({describe_errors(exc)}). {refuse}") from exc
