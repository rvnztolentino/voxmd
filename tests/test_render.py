"""Stage 3: extraction JSON in, markdown note out."""

from __future__ import annotations

import io
import json
import re
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from conftest import VALID_EXTRACTION
from voxmd import render as rn
from voxmd.config import RenderConfig
from voxmd.entities import Entities, EntityIndex
from voxmd.errors import ConfigError, InputError
from voxmd.schema import Extraction

WHEN = datetime(2026, 9, 15, 14, 3)
KB = 1024


def known(people=(), topics=()) -> Entities:
    return Entities(
        path=Path("entities.json"),
        exists=True,
        people=EntityIndex(people, threshold=90),
        topics=EntityIndex(topics, threshold=90),
    )


def note(
    data: dict[str, object] | None = None,
    *,
    people=(),
    topics=(),
    template_text: str | None = None,
    tmp_path: Path | None = None,
    **meta: object,
) -> rn.RenderResult:
    extraction = Extraction.model_validate({**VALID_EXTRACTION, **(data or {})})
    template_path = None
    if template_text is not None:
        assert tmp_path is not None
        template_path = tmp_path / "custom.md.j2"
        template_path.write_text(template_text)
    compiled, label = rn.load_template(template_path, max_bytes=64 * KB)
    return rn.render_note(
        extraction,
        meta=rn.NoteMeta(created=WHEN, **meta),  # type: ignore[arg-type]
        entities=known(people, topics),
        template=compiled,
        template_label=label,
    )


def split(markdown: str) -> tuple[dict[str, object], str]:
    assert markdown.startswith("---\n")
    _, frontmatter, body = markdown.split("---\n", 2)
    return yaml.safe_load(frontmatter), body


def test_default_note_layout() -> None:
    result = note(people=["Marco"], source="/Users/me/Memos/memo.m4a", duration_s=205.4)

    assert result.markdown == (
        "---\n"
        "date: 2026-09-15T14:03\n"
        "source: memo.m4a\n"
        "duration: '3:25'\n"
        "---\n"
        "\n"
        "# Ship date\n"
        "\n"
        "[[Marco]] and Ana agreed to ship on Friday.\n"
        "\n"
        "## Decisions\n"
        "\n"
        "- Ship on Friday\n"
        "\n"
        "## Actions\n"
        "\n"
        "- [ ] Email the client\n"
        "\n"
        "## Related\n"
        "\n"
        "- People: [[Marco]], Ana\n"
        "- Topics: release\n"
    )


def test_packaged_template_ships_inside_the_package() -> None:
    assert resources.files("voxmd").joinpath("templates", "note.md.j2").is_file()


class TestFrontmatter:
    def test_is_valid_yaml_holding_only_the_recording_facts(self) -> None:
        frontmatter, _ = split(note(source="memo.m4a", duration_s=65).markdown)

        assert frontmatter == {"date": "2026-09-15T14:03", "source": "memo.m4a", "duration": "1:05"}

    def test_unknown_facts_are_left_out(self) -> None:
        frontmatter, _ = split(note().markdown)

        assert frontmatter == {"date": "2026-09-15T14:03"}

    def test_only_the_source_file_name_is_recorded(self) -> None:
        markdown = note(source="/Users/me/Private Clients/acme/memo.m4a").markdown

        assert "Private Clients" not in markdown

    @pytest.mark.parametrize(
        "source",
        [
            "x\n---\nevil: true",
            "evil: true",
            "--- #",
            "'quoted' \"double\"",
            "[a, b]",
            "&anchor *ref",
        ],
    )
    def test_source_names_cannot_add_keys_or_close_the_block(self, source: str) -> None:
        frontmatter, body = split(note(source=source).markdown)

        assert set(frontmatter) == {"date", "source"}
        assert "evil" not in body


class TestEscaping:
    @pytest.mark.parametrize(
        ("raw", "escaped"),
        [
            ("![x](http://e.test/p.png)", r"!\[x\](http://e.test/p.png)"),
            ("<img src=http://e.test/t.png>", r"\<img src\=http://e.test/t.png\>"),
            ("%%hidden%%", r"\%\%hidden\%\%"),
            ("#tag", r"\#tag"),
            ("[[Evil|Marco]]", r"\[\[Evil\|Marco\]\]"),
            ("$x$ and ==hi==", r"\$x\$ and \=\=hi\=\="),
            ("`code` ~~gone~~ ^block", r"\`code\` \~\~gone\~\~ \^block"),
            ("a | b & c", r"a \| b \& c"),
            ("**bold** _it_", r"\*\*bold\*\* \_it\_"),
            ("> [!note] callout", r"\> \[!note\] callout"),
            ("- item", r"\- item"),
            ("+ item", r"\+ item"),
            ("---", r"\---"),
            ("12. item", r"12\. item"),
            ("3) item", r"3\) item"),
            ("back\\slash", "back\\\\slash"),
            ("-5 degrees", "-5 degrees"),
            ("2026 plans", "2026 plans"),
            ("Plain words, (with) punctuation.", "Plain words, (with) punctuation."),
        ],
    )
    def test_escape_markdown(self, raw: str, escaped: str) -> None:
        assert rn.escape_markdown(raw) == escaped

    def test_model_output_cannot_create_links_html_embeds_or_comments(self) -> None:
        hostile = "[[Evil]] ![[secret.png]] <img src=x> %%hide%% [link](http://e.test)"
        result = note(
            {
                "title": hostile,
                "summary": hostile,
                "decisions": [hostile],
                "actions": [hostile],
                "people": [hostile],
                "topics": [hostile],
            }
        )
        _, body = split(result.markdown)

        assert re.search(r"(?<!\\)\[\[", body) is None
        assert re.search(r"(?<!\\)[<%]", body) is None
        assert re.search(r"(?<!\\)\]\(", body) is None


class TestLinking:
    def test_known_people_are_linked_inline_keeping_the_spoken_spelling(self) -> None:
        result = note({"summary": "marco met MARCO's team."}, people=["Marco"])

        assert "[[Marco|marco]] met [[Marco|MARCO]]'s team." in result.markdown

    def test_names_inside_other_words_are_not_linked(self) -> None:
        result = note({"summary": "Always call Al.", "people": ["Al"]}, people=["Al"])

        assert "Always call [[Al]]." in result.markdown

    def test_unknown_people_are_plain_text_and_reported_as_new(self) -> None:
        result = note(people=["Marco"])

        assert "[[Ana]]" not in result.markdown
        assert result.people.new == ["Ana"]

    def test_topics_link_in_the_list_but_not_inline(self) -> None:
        result = note({"summary": "Talked about the release."}, topics=["Release"])

        assert "Talked about the release." in result.markdown
        assert "- Topics: [[Release]]" in result.markdown

    def test_a_fuzzy_match_links_to_the_known_spelling(self) -> None:
        result = note({"people": ["Christophor"]}, people=["Christopher"])

        assert "- People: [[Christopher]]" in result.markdown
        assert result.people.new == []

    def test_variants_of_one_person_appear_once(self) -> None:
        result = note({"people": ["marco", "Marco."]}, people=["Marco"])

        assert "- People: [[Marco]]\n" in result.markdown

    def test_inline_link_inside_hostile_text_stays_contained(self) -> None:
        result = note({"summary": "[[Evil|Marco]] and Marco"}, people=["Marco"])

        assert r"\[\[Evil\|[[Marco]]\]\] and [[Marco]]" in result.markdown


class TestSections:
    def test_empty_sections_are_left_out(self) -> None:
        markdown = note({"decisions": [], "actions": [], "people": [], "topics": []}).markdown

        assert "## Decisions" not in markdown
        assert "## Actions" not in markdown
        assert "## Related" not in markdown

    def test_actions_are_checkboxes(self) -> None:
        markdown = note({"actions": ["Email Ana", "Book the room"]}).markdown

        assert "- [ ] Email Ana\n- [ ] Book the room\n" in markdown


class TestTemplates:
    def test_a_custom_template_is_used_with_escaped_values(self, tmp_path: Path) -> None:
        result = note(
            {"title": "#1 plan"}, template_text="{{ title }} on {{ date }}", tmp_path=tmp_path
        )

        assert result.markdown == r"\#1 plan on 2026-09-15" + "\n"

    def test_syntax_errors_name_the_line(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="line 2"):
            note(template_text="ok\n{% if %}", tmp_path=tmp_path)

    def test_misspelt_variables_fail_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="undefined"):
            note(template_text="{{ titel }}", tmp_path=tmp_path)

    @pytest.mark.parametrize(
        "template_text",
        [
            "{{ title.__class__.__mro__ }}",
            "{{ cycler.__init__.__globals__ }}",
            "{{ decisions.append('x') }}",
        ],
    )
    def test_the_sandbox_blocks_internals_and_mutation(
        self, tmp_path: Path, template_text: str
    ) -> None:
        with pytest.raises(ConfigError, match="could not be rendered"):
            note(template_text=template_text, tmp_path=tmp_path)

    def test_template_bugs_do_not_echo_note_content(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as info:
            note(template_text="{{ summary + 1 }}", tmp_path=tmp_path)

        assert "TypeError" in str(info.value)
        assert "Ana" not in str(info.value)

    def test_template_files_are_checked_like_other_inputs(self, tmp_path: Path) -> None:
        wrong = tmp_path / "note.py"
        wrong.write_text("{{ title }}")
        with pytest.raises(InputError, match="Unsupported file type"):
            rn.load_template(wrong, max_bytes=64 * KB)

        big = tmp_path / "note.md.j2"
        big.write_text("x" * 100)
        with pytest.raises(InputError, match="limit"):
            rn.load_template(big, max_bytes=50)

    def test_render_config_expands_the_template_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        assert RenderConfig(template="~/note.md.j2").template == tmp_path / "note.md.j2"
        with pytest.raises(ValidationError):
            RenderConfig.model_validate({"templat": "x"})


class TestReadExtraction:
    def test_reads_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "memo.json"
        path.write_text(json.dumps(VALID_EXTRACTION))

        assert rn.read_extraction(path, max_bytes=KB).title == "Ship date"

    def test_reads_stdin(self) -> None:
        stdin = io.TextIOWrapper(io.BytesIO(json.dumps(VALID_EXTRACTION).encode()))

        assert rn.read_extraction(None, max_bytes=KB, stdin=stdin).people == ["Marco", "Ana"]

    def test_hand_edited_values_are_cleaned_again(self, tmp_path: Path) -> None:
        path = tmp_path / "memo.json"
        path.write_text(
            json.dumps({**VALID_EXTRACTION, "title": "Ship‮ date.", "actions": ["- Email", "email"]})
        )

        extraction = rn.read_extraction(path, max_bytes=KB)

        assert extraction.title == "Ship date"
        assert extraction.actions == ["Email"]

    @pytest.mark.parametrize(
        "content", ['{"title": "Secret Client Name"}', "not json at all", "[]"]
    )
    def test_invalid_extractions_fail_without_echoing_content(
        self, tmp_path: Path, content: str
    ) -> None:
        path = tmp_path / "memo.json"
        path.write_text(content)

        with pytest.raises(InputError, match="Not a valid extraction") as info:
            rn.read_extraction(path, max_bytes=KB)
        assert "Secret Client Name" not in str(info.value)

    def test_only_json_files_are_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "memo.txt"
        path.write_text(json.dumps(VALID_EXTRACTION))

        with pytest.raises(InputError, match="Unsupported file type"):
            rn.read_extraction(path, max_bytes=KB)


class TestDates:
    def test_defaults_to_now_at_minute_precision(self) -> None:
        parsed = rn.parse_date(None)

        assert parsed.second == 0
        assert parsed.microsecond == 0
        assert abs((datetime.now() - parsed).total_seconds()) < 120

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2026-09-15", datetime(2026, 9, 15, 0, 0)),
            ("2026-09-15T14:03:59", datetime(2026, 9, 15, 14, 3)),
        ],
    )
    def test_parses_iso_dates(self, value: str, expected: datetime) -> None:
        assert rn.parse_date(value) == expected

    def test_timezones_convert_to_local_time(self) -> None:
        aware = datetime(2026, 9, 15, 6, 3, tzinfo=UTC)

        assert rn.parse_date(aware.isoformat()) == aware.astimezone().replace(tzinfo=None)

    def test_rejects_non_dates(self) -> None:
        with pytest.raises(InputError, match="ISO 8601"):
            rn.parse_date("next tuesday")

    @pytest.mark.parametrize(
        ("seconds", "text"),
        [(None, None), (0, "0:00"), (59.6, "1:00"), (205.4, "3:25"), (3723, "1:02:03")],
    )
    def test_format_duration(self, seconds: float | None, text: str | None) -> None:
        assert rn.format_duration(seconds) == text
