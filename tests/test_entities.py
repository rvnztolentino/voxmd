"""Stage 4: entities.json matching, validation, and safe updates."""

from __future__ import annotations

import json
import stat
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from voxmd import entities as ent
from voxmd import safe
from voxmd.config import EntitiesConfig
from voxmd.errors import ConfigError, ToolTimeout

MAX = 1 << 20


def index(*names: str, threshold: int = 90) -> ent.EntityIndex:
    return ent.EntityIndex(names, threshold=threshold)


def update(path: Path, *, people=(), topics=(), max_bytes: int = MAX) -> int:
    return ent.update_entities(
        path, people=people, topics=topics, max_bytes=max_bytes, threshold=90
    )


class TestMatching:
    @pytest.mark.parametrize("variant", ["Marco", "marco", "MARCO", "Marco.", "  marco "])
    def test_case_and_punctuation_variants_are_one_entity(self, variant: str) -> None:
        assert index("Marco").find(variant) == "Marco"

    def test_accents_are_ignored(self) -> None:
        assert index("José").find("Jose") == "José"

    @pytest.mark.parametrize(("known", "other"), [("Marco", "Marcus"), ("Ana", "Anna")])
    def test_distinct_short_names_stay_apart(self, known: str, other: str) -> None:
        assert index(known).find(other) is None

    def test_a_spelling_slip_in_a_longer_name_merges(self) -> None:
        assert index("Christopher").find("Christophor") == "Christopher"

    def test_threshold_is_configurable(self) -> None:
        assert index("Christopher", threshold=95).find("Christophor") is None

    def test_exact_hits_never_run_the_fuzzy_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rapidfuzz import process

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("fuzzy scan ran on an exact hit")

        monkeypatch.setattr(process, "extractOne", boom)
        assert index("Marco", "Ana").find("marco") == "Marco"

    def test_near_duplicates_already_in_the_file_stay_distinct(self) -> None:
        # The user wrote both; loading must not merge (and later drop) one.
        assert len(index("Christopher", "Christophor")) == 2

    def test_names_are_made_link_safe(self) -> None:
        assert ent.link_target("Ana]]|alias#h^b/x") == "Ana alias h b x"
        assert ent.link_target("  [[]]  ") == ""

    def test_a_name_with_nothing_usable_is_ignored(self) -> None:
        assert index().add("[[]]") == (None, False)


def test_resolve_links_known_names_and_collects_new_ones_once() -> None:
    resolved = ent.resolve(["marco", "Ana", "ana", "Bob"], index("Marco"))

    assert resolved.links == {"marco": "Marco"}
    assert resolved.new == ["Ana", "Bob"]


class TestLoad:
    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        loaded = ent.load_entities(tmp_path / "none.json", max_bytes=MAX, threshold=90)

        assert not loaded.exists
        assert len(loaded.people) == 0

    def test_loads_people_and_topics(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"
        path.write_text('{"people": ["Marco"], "topics": ["release"]}')

        loaded = ent.load_entities(path, max_bytes=MAX, threshold=90)

        assert loaded.exists
        assert loaded.people.names == ["Marco"]
        assert loaded.topics.find("Release") == "release"

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("{nope", "not valid JSON"),
            ("[]", "JSON object"),
            ('{"people": "Marco"}', "Invalid entities file"),
            ('{"persons": []}', "Invalid entities file"),
            ('{"people": [1]}', "Invalid entities file"),
        ],
    )
    def test_malformed_files_are_refused(self, tmp_path: Path, content: str, message: str) -> None:
        path = tmp_path / "entities.json"
        path.write_text(content)

        with pytest.raises(ConfigError, match=message):
            ent.load_entities(path, max_bytes=MAX, threshold=90)

    def test_errors_never_echo_file_contents(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"
        path.write_text('{"people": [42], "secret": "Marco Rossi"}')

        with pytest.raises(ConfigError) as info:
            ent.load_entities(path, max_bytes=MAX, threshold=90)

        assert "Marco Rossi" not in str(info.value)

    def test_directory_and_oversized_files_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not a regular file"):
            ent.load_entities(tmp_path, max_bytes=MAX, threshold=90)

        big = tmp_path / "entities.json"
        big.write_text(json.dumps({"people": ["x" * 100]}))
        with pytest.raises(ConfigError, match="limit"):
            ent.load_entities(big, max_bytes=50, threshold=90)


class TestUpdate:
    def test_creates_a_private_file_and_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "config" / "entities.json"

        added = update(path, people=["Marco"], topics=["release"])

        assert added == 2
        assert json.loads(path.read_text()) == {"people": ["Marco"], "topics": ["release"]}
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_appends_only_new_names_and_keeps_entries_verbatim(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"
        path.write_text('{"people": ["marco rossi", "Christophor"], "topics": []}')

        added = update(path, people=["Marco Rossi", "Christopher", "Ana"])

        assert added == 1
        assert json.loads(path.read_text())["people"] == ["marco rossi", "Christophor", "Ana"]

    def test_nothing_new_leaves_the_file_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"
        original = '{ "people": ["Marco"] }'
        path.write_text(original)

        assert update(path, people=["marco"]) == 0
        assert path.read_text() == original

    def test_a_corrupt_file_is_never_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"
        path.write_text("{broken")

        with pytest.raises(ConfigError, match="won't overwrite"):
            update(path, people=["Marco"])
        assert path.read_text() == "{broken"

    def test_additions_are_link_safe(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"

        update(path, people=["Ana]]#x", "[[]]"])

        assert json.loads(path.read_text())["people"] == ["Ana x"]

    def test_a_symlinked_file_updates_its_target(self, tmp_path: Path) -> None:
        real = tmp_path / "dotfiles" / "entities.json"
        real.parent.mkdir()
        real.write_text('{"people": []}')
        link = tmp_path / "entities.json"
        link.symlink_to(real)

        update(link, people=["Marco"])

        assert link.is_symlink()
        assert json.loads(real.read_text())["people"] == ["Marco"]

    def test_growth_past_the_limit_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"
        path.write_text('{"people": []}')

        with pytest.raises(ConfigError, match="would grow"):
            update(path, people=[f"Person {i}" for i in range(20)], max_bytes=64)
        assert path.read_text() == '{"people": []}'

    def test_a_held_lock_times_out_instead_of_hanging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ent, "LOCK_TIMEOUT_S", 0.2)
        path = tmp_path / "entities.json"
        lock = path.resolve().with_name(".entities.json.lock")

        with (
            safe.file_lock(lock, timeout_s=1, what="test"),
            pytest.raises(ToolTimeout, match="locked"),
        ):
            update(path, people=["Marco"])

    def test_concurrent_updates_keep_every_addition(self, tmp_path: Path) -> None:
        path = tmp_path / "entities.json"
        names = [f"Person {chr(65 + i)}{chr(75 + i)}" for i in range(8)]
        threads = [
            threading.Thread(target=update, args=(path,), kwargs={"people": [n]}) for n in names
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(json.loads(path.read_text())["people"]) == sorted(names)


class TestConfig:
    def test_default_file_is_under_the_user_config_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        assert EntitiesConfig().file == tmp_path / ".config" / "voxmd" / "entities.json"

    @pytest.mark.parametrize("threshold", [49, 101])
    def test_threshold_is_bounded(self, threshold: int) -> None:
        with pytest.raises(ValidationError):
            EntitiesConfig(fuzzy_threshold=threshold)


def test_a_link_target_never_holds_a_comment_marker() -> None:
    from voxmd.entities import link_target

    assert link_target("Ana%%%x 10%") == "Ana%x 10%"
