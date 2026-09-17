"""Stage 3: an extraction in, a markdown note out.

The note comes from a Jinja2 template: the packaged ``templates/note.md.j2``, or
a copy of it named by ``render.template``, so the layout changes without
touching code.

Everything in an extraction came from a model reading a memo, and a memo can
say anything. So:

* **Values are escaped before the template sees them.** A custom template gets
  no raw model output it could forget to escape. Escaping covers the markdown
  and Obsidian syntax that could fetch something, hide text, or restructure the
  note: images and links, raw HTML (``<img src=...>`` would load a remote URL
  when the note is opened), ``%%`` comments, ``#tags``, highlights, math,
  tables, code spans, and list or quote markers at the start of a line.
* **Wikilinks are built only here**, from entity names already stripped of the
  characters that could alias, anchor, or break out of a link.
* **Frontmatter is a dict dumped by ``yaml.safe_dump``**, never interpolated, so
  no value can add keys or close the ``---`` block.
* **The template runs in Jinja's immutable sandbox with StrictUndefined.** It is
  your own file, but the sandbox keeps it away from Python internals, and a
  misspelt variable fails loudly instead of rendering as nothing.
* Model output is only ever template *data*, never template *source*, so it
  can't inject Jinja syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import IO, Any

import jinja2
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import ValidationError

from . import safe
from .entities import LINK_UNSAFE, Entities, Resolved, resolve
from .errors import ConfigError, InputError
from .schema import Extraction, clean_text, describe_errors

JSON_SUFFIXES = frozenset({".json"})
TEMPLATE_SUFFIXES = frozenset({".j2", ".jinja", ".jinja2", ".md", ".txt"})
PACKAGED_TEMPLATE = "note.md.j2"
MAX_SOURCE_CHARS = 255
MIN_INLINE_NAME_CHARS = 2

# Characters with inline meaning in CommonMark or Obsidian. All are ASCII
# punctuation, so a backslash escape is valid for every one of them.
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~=$^&])")
# "%" means something only doubled: "%%" opens an Obsidian comment. A lone one,
# as in "70%", is left alone so figures read normally in the editor.
_COMMENT_MARK = re.compile(r"%(?=%)|(?<=%)%")
# Block syntax that only counts at the start of a line: bullets, ordered-list
# numbers, and --- / +++ runs. Everything else (#, >, *) is already escaped.
_BLOCK_START = re.compile(r"^(?:(?P<run>[-+])[-+]*|\d{1,9}(?P<num>[.)]))(?=\s|$)")

_ERROR_TYPES = (TypeError, ValueError, AttributeError, KeyError, IndexError, ArithmeticError)


@dataclass(frozen=True)
class NoteMeta:
    """Facts about the recording that don't come from the model."""

    created: datetime
    source: str | None = None
    """The audio file. Only its name is recorded, never the directory."""
    duration_s: float | None = None
    transcript: str | None = None
    """The transcript note's vault-relative path without ``.md``, to link to."""


@dataclass(frozen=True)
class RenderResult:
    markdown: str
    people: Resolved
    topics: Resolved


def read_extraction(
    source: Path | str | None, *, max_bytes: int, stdin: IO[str] | None = None
) -> Extraction:
    """Read an extraction JSON file, or stdin when ``source`` is None or ``-``."""
    text = safe.read_text_input(
        source,
        allowed_suffixes=JSON_SUFFIXES,
        max_bytes=max_bytes,
        stdin=stdin,
        what="extraction",
        limit_setting="limits.max_extraction_kb",
        no_input_message=(
            "No extraction given. Pass a JSON file, or pipe one in:\n"
            "  voxmd extract memo.txt | voxmd render"
        ),
    )
    try:
        return Extraction.model_validate_json(text)
    except ValidationError as exc:
        raise InputError(
            f"Not a valid extraction ({describe_errors(exc)}).\n"
            "render expects the JSON that voxmd extract prints."
        ) from exc


def parse_date(value: str | None) -> datetime:
    """``--date`` as a local, minute-precision datetime. Now when not given."""
    if value is None:
        return datetime.now().replace(second=0, microsecond=0)
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise InputError(f"--date is not an ISO 8601 date or datetime: {value!r}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed.replace(second=0, microsecond=0)


def load_template(path: Path | None, *, max_bytes: int) -> tuple[jinja2.Template, str]:
    """Compile the note template. Returns it with a label for messages."""
    if path is None:
        label = f"built-in {PACKAGED_TEMPLATE}"
        source = (
            resources.files("voxmd")
            .joinpath("templates", PACKAGED_TEMPLATE)
            .read_text(encoding="utf-8")
        )
    else:
        label = Path(path).name
        source = safe.read_text_input(
            path,
            allowed_suffixes=TEMPLATE_SUFFIXES,
            max_bytes=max_bytes,
            stdin=None,
            what="template",
            limit_setting="limits.max_template_kb",
            no_input_message="No template given.",
        )
    try:
        return _environment().from_string(source), label
    except jinja2.TemplateSyntaxError as exc:
        raise ConfigError(
            f"Template {label} has a syntax error on line {exc.lineno}: {exc.message}"
        ) from exc


def render_note(
    extraction: Extraction,
    *,
    meta: NoteMeta,
    entities: Entities,
    template: jinja2.Template,
    template_label: str = "template",
) -> RenderResult:
    """Render one note. Writes nothing; the caller decides where it goes."""
    people = resolve(extraction.people, entities.people)
    topics = resolve(extraction.topics, entities.topics)
    context = build_context(extraction, meta=meta, people=people, topics=topics)
    try:
        markdown = template.render(context)
    except jinja2.TemplateError as exc:
        raise ConfigError(
            f"Template {template_label} could not be rendered: {exc.message or type(exc).__name__}"
        ) from exc
    except _ERROR_TYPES as exc:
        # A bug in a custom template, e.g. arithmetic on a string. The type
        # alone is reported: the message could quote note content.
        raise ConfigError(
            f"Template {template_label} could not be rendered: {type(exc).__name__}"
        ) from exc
    return RenderResult(markdown=markdown.rstrip("\n") + "\n", people=people, topics=topics)


def build_context(
    extraction: Extraction, *, meta: NoteMeta, people: Resolved, topics: Resolved
) -> dict[str, Any]:
    """Template variables. Every string in here is already safe to emit."""
    person_links: dict[str, str] = {}
    for name, target in people.links.items():
        person_links.setdefault(name.casefold(), target)
        person_links.setdefault(target.casefold(), target)
    pattern = _name_pattern(person_links)
    source = _source_name(meta.source)

    return {
        "frontmatter": build_frontmatter(meta),
        "title": escape_markdown(extraction.title),
        "summary": _linkify(extraction.summary, pattern, person_links),
        "key_points": [_linkify(item, pattern, person_links) for item in extraction.key_points],
        "decisions": [_linkify(item, pattern, person_links) for item in extraction.decisions],
        "actions": [_linkify(item, pattern, person_links) for item in extraction.actions],
        "people": _entity_items(extraction.people, people),
        "topics": _entity_items(extraction.topics, topics),
        "date": meta.created.strftime("%Y-%m-%d"),
        "time": meta.created.strftime("%H:%M"),
        "source": escape_markdown(source) if source else "",
        "duration": format_duration(meta.duration_s) or "",
        "transcript": transcript_link(meta.transcript) if meta.transcript else "",
    }


TRANSCRIPT_LABEL = "Full transcript"
SENTENCES_PER_PARAGRAPH = 4
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")
TRANSCRIPT_NOTICE = (
    "*Transcribed automatically, so words and names may be misheard. "
    "The recording is the source of truth.*"
)


def transcript_link(path: str) -> str:
    """``[[folder/name|Full transcript]]`` for a vault-relative note path.

    The path is exact when every part of it is safe inside a link. A folder
    name holding link syntax (``#``, ``|``, ``]``) falls back to the file name
    alone, which Obsidian still resolves, rather than to a broken link.
    """
    parts = [part for part in path.split("/") if part]
    if not parts or any(LINK_UNSAFE.search(part) for part in parts):
        name = LINK_UNSAFE.sub(" ", parts[-1] if parts else "").strip()
        return f"[[{name}|{TRANSCRIPT_LABEL}]]" if name else ""
    return f"[[{'/'.join(parts)}|{TRANSCRIPT_LABEL}]]"


def render_transcript(text: str, *, title: str, meta: NoteMeta, max_bytes: int) -> str:
    """The transcript note: frontmatter, a heading, and the words in short paragraphs.

    Every paragraph is escaped exactly like model output. A transcript is just
    as untrusted: a memo that says "open bracket open bracket" can't make a
    link, and one that says a URL can't make it load.
    """
    words = clean_text(text, limit=max_bytes)
    if len(words.encode("utf-8")) > max_bytes:
        # A character limit alone lets three-byte scripts run to triple the size.
        words = words.encode("utf-8")[: max_bytes - 3].decode("utf-8", errors="ignore") + "…"
    sentences = [part for part in _SENTENCE_END.split(words) if part]
    paragraphs = [
        escape_markdown(" ".join(sentences[i : i + SENTENCES_PER_PARAGRAPH]))
        for i in range(0, len(sentences), SENTENCES_PER_PARAGRAPH)
    ]
    body = "\n\n".join(paragraphs) or "*(nothing was transcribed)*"
    heading = escape_markdown(f"{title} (transcript)")
    return f"---\n{build_frontmatter(meta)}---\n\n# {heading}\n\n{body}\n\n{TRANSCRIPT_NOTICE}\n"


def build_frontmatter(meta: NoteMeta) -> str:
    """YAML frontmatter body (without the ``---`` lines), ending in a newline."""
    data: dict[str, str] = {"date": meta.created.isoformat(timespec="minutes")}
    source = _source_name(meta.source)
    if source:
        data["source"] = source
    duration = format_duration(meta.duration_s)
    if duration:
        data["duration"] = duration
    return yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=False, width=10_000
    )


def escape_markdown(text: str) -> str:
    """Escape a single line so it renders as the literal text it is."""
    escaped = _COMMENT_MARK.sub(r"\\%", _MD_SPECIAL.sub(r"\\\1", text))
    return _escape_block_start(escaped)


def wikilink(target: str, label: str | None = None) -> str:
    """``[[target]]``, or ``[[target|label]]`` when the text used another spelling."""
    if (
        label is None
        or label == target
        or LINK_UNSAFE.search(label)
        or _MD_SPECIAL.search(label)
        or "%" in label
    ):
        return f"[[{target}]]"
    return f"[[{target}|{label}]]"


def format_duration(seconds: float | None) -> str | None:
    """``3:25`` or ``1:02:03``, or None when the duration is unknown."""
    if seconds is None or seconds < 0:
        return None
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _environment() -> ImmutableSandboxedEnvironment:
    # S701: autoescape is HTML escaping, which is wrong for markdown. Values are
    # escaped for markdown in build_context before they reach the template.
    return ImmutableSandboxedEnvironment(  # noqa: S701
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        auto_reload=False,
    )


def _escape_block_start(text: str) -> str:
    match = _BLOCK_START.match(text)
    if match is None:
        return text
    if match.group("run"):
        return "\\" + text
    num_end = match.start("num")
    return f"{text[:num_end]}\\{text[num_end:]}"


def _name_pattern(links: dict[str, str]) -> re.Pattern[str] | None:
    """One regex matching any linked person's name as a whole word."""
    names = sorted(
        (name for name in links if len(name) >= MIN_INLINE_NAME_CHARS), key=len, reverse=True
    )
    if not names:
        return None
    alternatives = "|".join(re.escape(name) for name in names)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _linkify(text: str, pattern: re.Pattern[str] | None, links: dict[str, str]) -> str:
    """Escape text, turning mentions of known people into wikilinks."""
    if pattern is None:
        return escape_markdown(text)
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        target = links.get(match.group(0).casefold())
        if target is None:
            continue
        parts.append(_MD_SPECIAL.sub(r"\\\1", text[last : match.start()]))
        parts.append(wikilink(target, match.group(0)))
        last = match.end()
    parts.append(_MD_SPECIAL.sub(r"\\\1", text[last:]))
    return _escape_block_start("".join(parts))


def _entity_items(names: list[str], resolved: Resolved) -> list[str]:
    """Known names as wikilinks, unknown ones as escaped text, without repeats."""
    items: list[str] = []
    seen: set[str] = set()
    for name in names:
        target = resolved.links.get(name)
        item = wikilink(target) if target is not None else escape_markdown(name)
        if item not in seen:
            seen.add(item)
            items.append(item)
    return items


def _source_name(source: str | None) -> str:
    if not source:
        return ""
    return clean_text(Path(source).name, limit=MAX_SOURCE_CHARS)
