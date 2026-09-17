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
MAX_SUMMARY_CHARS = 4_000
"""Room for the ten-sentence summary a long meeting gets, with margin."""
MAX_ITEM_CHARS = 300
MAX_ITEMS = 50

# Words a model lists as "people" that aren't anyone's name. Kept out of the
# note and, more importantly, out of entities.json, where they would turn every
# later "you" into a [[you]] link. Compared case-insensitively, whole item only.
_PRONOUNS = "i me my myself you your yourself he him his she her hers they them their we us our it"
_TAGALOG_PRONOUNS = "ako akin ikaw ka kayo siya sila kami kita tayo"
_PLACEHOLDERS = (
    "someone|somebody|anyone|everyone|everybody|nobody|speaker|the speaker|narrator|"
    "the narrator|host|the host|user|the user|caller|unknown|unnamed|person|a person|"
    "friend|a friend|my friend|boss|my boss|manager|the manager|my manager|client|"
    "the client|team|the team"
)
NOT_NAMES = frozenset([*_PRONOUNS.split(), *_TAGALOG_PRONOUNS.split(), *_PLACEHOLDERS.split("|")])

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


_DETERMINERS = frozenset(
    ["the", "a", "an", "my", "your", "his", "her", "our", "their", "this", "that"]
)


def looks_like_a_name(item: str) -> bool:
    """Whether an extracted "person" plausibly is one.

    A fixed list can't catch every way a model describes someone unnamed ("the
    person", "her mom"), so two general rules back it up. In a script with
    capital letters, a name has at least one; and a determiner followed only by
    lowercase words is a description. Scripts without case (Chinese, Japanese,
    Arabic) pass both rules untouched.
    """
    if item.casefold() in NOT_NAMES or not any(ch.isalpha() for ch in item):
        return False
    cased = [ch for ch in item if ch.isupper() or ch.islower()]
    if cased and not any(ch.isupper() for ch in cased):
        return False
    first, _, rest = item.partition(" ")
    return not (first.casefold() in _DETERMINERS and rest and rest == rest.lower())


def _clean_people(values: list[str]) -> list[str]:
    """Like any list, minus pronouns, roles, and descriptions of unnamed people."""
    # Filtered before de-duplicating, so a lowercase "marco" can't knock out "Marco".
    names = [_clean_items([value]) for value in values]
    return _clean_items([item[0] for item in names if item and looks_like_a_name(item[0])])


def _require_every_field(schema: dict[str, object]) -> None:
    """Make the model's output grammar demand every field.

    ``key_points`` has a default so extraction files written before it existed
    still load, but a default would also make it optional for the model, and an
    optional field is one a model quietly leaves out.
    """
    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties)


class Extraction(BaseModel):
    """Structured note content. Its JSON schema is also the model's output grammar.

    The validators run on every load, not only on model output, so an
    extraction file edited by hand is cleaned exactly like a fresh one.
    """

    model_config = ConfigDict(extra="ignore", json_schema_extra=_require_every_field)

    title: str = Field(description="Short descriptive title, at most 8 words.")
    summary: str = Field(description="Summary of what was said, longer for longer recordings.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Main points, facts, or arguments, one per item. Empty if none.",
    )
    decisions: list[str] = Field(description="Decisions that were made. Empty if none.")
    actions: list[str] = Field(
        description="Concrete follow-up tasks, each starting with a verb. Empty if none."
    )
    people: list[str] = Field(
        description="Proper names of specific people. Never pronouns or roles. Empty if none."
    )
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

    @field_validator("key_points", "decisions", "actions", "topics")
    @classmethod
    def _clean_lists(cls, values: list[str]) -> list[str]:
        return _clean_items(values)

    @field_validator("people")
    @classmethod
    def _clean_names(cls, values: list[str]) -> list[str]:
        return _clean_people(values)


def describe_errors(exc: ValidationError, *, limit: int = 3) -> str:
    """Field locations and messages only. Input values could be memo content."""
    parts = []
    for err in exc.errors()[:limit]:
        location = ".".join(str(part) for part in err["loc"]) or "(root)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)
