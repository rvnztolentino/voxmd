"""Stage 4: known people and topics, and which names become [[wikilinks]].

``entities.json`` is a small file you can edit by hand::

    {"people": ["Marco", "Ana"], "topics": ["release"]}

Matching is exact first, fuzzy second:

* Names are normalized (accents dropped, casefolded, punctuation removed) into a
  dict, so "Marco", "marco" and "Marco." are one O(1) lookup.
* Only on a miss does rapidfuzz compare against the known names, with
  ``fuzz.ratio`` at ``entities.fuzzy_threshold`` (90 by default). That merges a
  spelling slip in a longer name ("Christopher"/"Christophor", 91) but keeps
  short distinct names apart ("Marco"/"Marcus", 73; "Ana"/"Anna", 86). A linear
  fuzzy scan per name is the slow path, so it never runs when the dict hits.

Safety:

* A hand-edited file is validated on load. A malformed one is an error, never
  treated as empty, because the next update would then overwrite the list.
* Names become link targets, which Obsidian resolves as note filenames.
  Characters that would break a wikilink or a filename (``[ ] # ^ | \\ / : * ?
  " < >``) are replaced, so a name can't alias, anchor, or escape its link.
* Updates re-read the file under a lock and write it atomically, so two runs
  can't lose each other's additions and a crash can't leave half a file.
"""

from __future__ import annotations

import json
import re
import stat
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import safe
from .errors import ConfigError
from .schema import clean_text, describe_errors

MAX_ENTITY_CHARS = 120
MAX_ENTITIES = 10_000
LOCK_TIMEOUT_S = 10.0
KINDS = ("people", "topics")

LINK_UNSAFE = re.compile(r'[\[\]#^|\\/:*?"<>]')
_NOT_WORD = re.compile(r"[\W_]+")
_SPACES = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Comparison key for a name: accents dropped, casefolded, punctuation removed."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NOT_WORD.sub(" ", stripped.casefold()).strip()


def link_target(name: str) -> str:
    """A name made safe to put inside ``[[ ]]``, or ``""`` if nothing usable is left."""
    text = LINK_UNSAFE.sub(" ", clean_text(name, limit=MAX_ENTITY_CHARS))
    return _SPACES.sub(" ", text).strip(" .")


class EntitiesFile(BaseModel):
    """The shape of ``entities.json``. Unknown keys are refused, so a typo shows."""

    model_config = ConfigDict(extra="forbid")

    people: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    @field_validator("people", "topics")
    @classmethod
    def _clean(cls, values: list[str]) -> list[str]:
        if len(values) > MAX_ENTITIES:
            raise ValueError(f"has more than {MAX_ENTITIES} entries")
        return values


class EntityIndex:
    """Known names of one kind, looked up exact-first and fuzzy only on a miss."""

    def __init__(self, names: Iterable[str] = (), *, threshold: int) -> None:
        self.threshold = threshold
        self._by_key: dict[str, str] = {}
        for name in names:
            # Entries already in the file are kept as distinct as the user
            # wrote them: only exact duplicates collapse, never fuzzy ones.
            target = link_target(name)
            key = normalize(target)
            if key and key not in self._by_key:
                self._by_key[key] = target

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def names(self) -> list[str]:
        return list(self._by_key.values())

    def find(self, name: str) -> str | None:
        """The known name this one refers to, or None."""
        key = normalize(link_target(name))
        if not key:
            return None
        hit = self._by_key.get(key)
        if hit is not None or not self._by_key:
            return hit
        # Imported on the slow path only: most lookups hit the dict.
        from rapidfuzz import fuzz, process

        best = process.extractOne(
            key, list(self._by_key), scorer=fuzz.ratio, score_cutoff=self.threshold
        )
        return self._by_key[best[0]] if best is not None else None

    def add(self, name: str) -> tuple[str | None, bool]:
        """Record a name. Returns its link target and whether it was new."""
        existing = self.find(name)
        if existing is not None:
            return existing, False
        target = link_target(name)
        key = normalize(target)
        if not key:
            return None, False
        self._by_key[key] = target
        return target, True


@dataclass(frozen=True)
class Resolved:
    """How one note's names of one kind map onto the entities file."""

    links: dict[str, str] = field(default_factory=dict)
    """Extracted name → known link target. Only names already in the file."""
    new: list[str] = field(default_factory=list)
    """Names not in the file, link-safe, de-duplicated, in first-seen order."""


@dataclass(frozen=True)
class Entities:
    """The loaded entities file."""

    path: Path
    exists: bool
    people: EntityIndex
    topics: EntityIndex


def resolve(names: Iterable[str], index: EntityIndex) -> Resolved:
    """Split a note's names into known ones (linked) and new ones (to append)."""
    links: dict[str, str] = {}
    pending = EntityIndex(threshold=index.threshold)
    new: list[str] = []
    for name in names:
        known = index.find(name)
        if known is not None:
            links[name] = known
            continue
        target, created = pending.add(name)
        if created and target is not None:
            new.append(target)
    return Resolved(links=links, new=new)


def load_entities(path: Path | str, *, max_bytes: int, threshold: int) -> Entities:
    """Load and validate the entities file. A missing file is an empty one."""
    target = Path(path).expanduser()
    raw = _read_raw(target, max_bytes)
    document = _validate(raw or {}, target)
    return Entities(
        path=target,
        exists=raw is not None,
        people=EntityIndex(document.people, threshold=threshold),
        topics=EntityIndex(document.topics, threshold=threshold),
    )


def update_entities(
    path: Path | str,
    *,
    people: Iterable[str],
    topics: Iterable[str],
    max_bytes: int,
    threshold: int,
) -> int:
    """Append names the file doesn't know yet. Returns how many were added.

    The file is re-read under the lock rather than trusting what was loaded
    earlier, so an addition made by another run in between is kept. Existing
    entries are written back exactly as they were.
    """
    additions = {"people": list(people), "topics": list(topics)}
    if not any(additions.values()):
        return 0

    target = Path(path).expanduser().resolve()
    lock = target.with_name(f".{target.name}.lock")
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with safe.file_lock(lock, timeout_s=LOCK_TIMEOUT_S, what=str(target)):
            raw = _read_raw(target, max_bytes) or {}
            current = _validate(raw, target)
            document: dict[str, Any] = dict(raw)
            added = 0
            for kind in KINDS:
                index = EntityIndex(getattr(current, kind), threshold=threshold)
                entries = list(getattr(current, kind))
                for name in additions[kind]:
                    link, created = index.add(name)
                    if created and link is not None:
                        entries.append(link)
                        added += 1
                document[kind] = entries
            if not added:
                return 0

            _validate(document, target)
            text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
            if len(text.encode("utf-8")) > max_bytes:
                raise ConfigError(
                    f"{target} would grow past the {safe.human_bytes(max_bytes)} limit.\n"
                    "Raise limits.max_entities_kb in your config, or prune the file."
                )
            safe.atomic_write_text(target, text)
            return added
    except OSError as exc:
        raise ConfigError(f"Could not update {target}: {exc.strerror or exc}") from exc


def _read_raw(target: Path, max_bytes: int) -> dict[str, Any] | None:
    """The file's JSON object, or None when the file doesn't exist."""
    try:
        info = target.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ConfigError(f"Could not read {target}: {exc.strerror or exc}") from exc

    # Checked before opening: opening a FIFO would block.
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError(f"Entities file is not a regular file: {target}")
    if info.st_size > max_bytes:
        raise ConfigError(
            f"{target} is {safe.human_bytes(info.st_size)}, over the "
            f"{safe.human_bytes(max_bytes)} limit.\n"
            "Raise limits.max_entities_kb in your config if this is expected."
        )
    try:
        with target.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ConfigError(f"Could not read {target}: {exc.strerror or exc}") from exc
    if len(data) > max_bytes:
        raise ConfigError(f"{target} is over the {safe.human_bytes(max_bytes)} limit.")

    try:
        raw = json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{target} is not UTF-8 text.") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{target} is not valid JSON (line {exc.lineno}, column {exc.colno}). "
            "Fix it by hand; voxmd won't overwrite it."
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(f'{target} must be a JSON object like {{"people": [], "topics": []}}.')
    return raw


def _validate(raw: dict[str, Any], target: Path) -> EntitiesFile:
    try:
        return EntitiesFile.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid entities file {target} ({describe_errors(exc)}). "
            "Fix it by hand; voxmd won't overwrite it."
        ) from exc
