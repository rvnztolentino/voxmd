"""The extraction schema, shared by the extract and render stages.

Kept apart from :mod:`extract` on purpose: that module imports the ollama client
and its httpx stack (about 125 ms), and ``voxmd render`` needs the schema but
never talks to Ollama. Only pydantic and the standard library are imported here.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MAX_TITLE_CHARS = 120
MAX_SUMMARY_CHARS = 2_000
MAX_ITEM_CHARS = 300
MAX_ITEMS = 50

# Unicode categories removed from every extracted string: control characters,
# format characters (zero-width joiners, bidi overrides that can make a note
# display text in a different order than it is stored) and lone surrogates.
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
# Leading list syntax a model sometimes adds itself: "- ", "* ", "1. ", "[ ] ".
# A bare "-" needs a following space, so "-5 degrees" keeps its sign.
_LIST_MARKER = re.compile(r"^(?:[-*+•]\s+|\d{1,3}[.)]\s+|\[[ xX]?\]\s*)+")
_WHITESPACE = re.compile(r"\s+")


def clean_text(value: str, *, limit: int) -> str:
    """Collapse a model-produced string to one clean, bounded line."""
    text = _WHITESPACE.sub(" ", unicodedata.normalize("NFC", value))
    text = "".join(ch for ch in text if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES)
    text = text.strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _clean_items(values: list[str]) -> list[str]:
    """Clean, de-duplicate case-insensitively, drop empties, and cap a list."""
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        item = clean_text(_LIST_MARKER.sub("", value.strip()), limit=MAX_ITEM_CHARS)
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) == MAX_ITEMS:
            break
    return items


class Extraction(BaseModel):
    """Structured note content. Its JSON schema is also the model's output grammar.

    The validators run on every load, not only on model output, so an
    extraction file edited by hand is cleaned exactly like a fresh one.
    """

    model_config = ConfigDict(extra="ignore")

    title: str = Field(description="Short descriptive title, at most 8 words.")
    summary: str = Field(description="Two to four sentence summary of the memo.")
    decisions: list[str] = Field(description="Decisions that were made. Empty if none.")
    actions: list[str] = Field(
        description="Concrete follow-up tasks, each starting with a verb. Empty if none."
    )
    people: list[str] = Field(description="Names of people mentioned. Empty if none.")
    topics: list[str] = Field(description="One to five short topic names.")

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        cleaned = clean_text(value, limit=MAX_TITLE_CHARS).rstrip(".")
        if not cleaned:
            raise ValueError("title is empty")
        return cleaned

    @field_validator("summary")
    @classmethod
    def _clean_summary(cls, value: str) -> str:
        cleaned = clean_text(value, limit=MAX_SUMMARY_CHARS)
        if not cleaned:
            raise ValueError("summary is empty")
        return cleaned

    @field_validator("decisions", "actions", "people", "topics")
    @classmethod
    def _clean_lists(cls, values: list[str]) -> list[str]:
        return _clean_items(values)


def describe_errors(exc: ValidationError, *, limit: int = 3) -> str:
    """Field locations and messages only. Input values could be memo content."""
    parts = []
    for err in exc.errors()[:limit]:
        location = ".".join(str(part) for part in err["loc"]) or "(root)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)
