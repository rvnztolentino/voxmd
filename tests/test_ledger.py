"""The ledger of processed recordings."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from voxmd.errors import ConfigError, InputError, OutputError
from voxmd.ledger import Ledger, LedgerEntry, hash_file

DIGEST = "a" * 64


def entry(**overrides: object) -> LedgerEntry:
    values: dict[str, object] = {
        "source": "/inbox/memo.m4a",
        "size": 4096,
        "processed_at": datetime(2026, 9, 15, 14, 3).astimezone(),
        "note": "/vault/2026-09-15 Ship date.md",
        "duration_s": 125.0,
    }
    values.update(overrides)
    return LedgerEntry.model_validate(values)


class TestHashFile:
    def test_matches_sha256_of_the_whole_file(self, tmp_path: Path) -> None:
        path = tmp_path / "memo.m4a"
        data = bytes(range(256)) * 10_000  # spans several chunks
        path.write_bytes(data)

        assert hash_file(path, max_bytes=len(data)) == hashlib.sha256(data).hexdigest()

    def test_a_file_past_the_limit_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "memo.m4a"
        path.write_bytes(b"x" * 100)

        with pytest.raises(InputError, match="still syncing"):
            hash_file(path, max_bytes=99)


class TestLedger:
    def test_a_missing_ledger_is_empty_and_is_not_created_by_loading(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"

        ledger = Ledger.load(path, max_bytes=1 << 20)

        assert len(ledger) == 0
        assert ledger.get(DIGEST) is None
        assert not path.exists()

    def test_recorded_entries_survive_a_reload_and_the_file_is_private(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ledger.json"
        Ledger.load(path, max_bytes=1 << 20).record(DIGEST, entry())

        reloaded = Ledger.load(path, max_bytes=1 << 20)

        assert reloaded.get(DIGEST) == entry()
        assert path.stat().st_mode & 0o777 == 0o600

    def test_recording_the_same_digest_replaces_its_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        ledger = Ledger.load(path, max_bytes=1 << 20)
        ledger.record(DIGEST, entry())
        ledger.record(DIGEST, entry(archived="/archive/memo.m4a"))

        reloaded = Ledger.load(path, max_bytes=1 << 20)
        assert len(reloaded) == 1
        assert reloaded.get(DIGEST).archived == "/archive/memo.m4a"  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("{", "not valid JSON"),
            ("[]", "must be a JSON object"),
            ('{"version": 2, "entries": {}}', "Invalid ledger"),
            ('{"version": 1, "entries": {"not-a-digest": {}}}', "Invalid ledger"),
            ('{"version": 1, "entries": {}, "extra": 1}', "Invalid ledger"),
        ],
    )
    def test_a_damaged_ledger_is_refused_and_never_overwritten(
        self, tmp_path: Path, content: str, message: str
    ) -> None:
        path = tmp_path / "ledger.json"
        path.write_text(content)

        with pytest.raises(ConfigError, match=message):
            Ledger.load(path, max_bytes=1 << 20)
        assert path.read_text() == content

    def test_invalid_utf8_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        path.write_bytes(b"\xff\xfe{}")

        with pytest.raises(ConfigError, match="UTF-8"):
            Ledger.load(path, max_bytes=1 << 20)

    def test_a_directory_in_its_place_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        path.mkdir()

        with pytest.raises(ConfigError, match="not a regular file"):
            Ledger.load(path, max_bytes=1 << 20)

    def test_an_oversized_ledger_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        path.write_text('{"version": 1, "entries": {}}' + " " * 100)

        with pytest.raises(ConfigError, match="max_ledger_mb"):
            Ledger.load(path, max_bytes=50)

    def test_a_write_that_would_pass_the_limit_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.json"
        ledger = Ledger.load(path, max_bytes=100)

        with pytest.raises(OutputError, match="max_ledger_mb"):
            ledger.record(DIGEST, entry())
        assert not path.exists()

    def test_a_malformed_digest_is_a_programming_error(self, tmp_path: Path) -> None:
        ledger = Ledger.load(tmp_path / "ledger.json", max_bytes=1 << 20)

        with pytest.raises(ValueError, match="SHA-256"):
            ledger.record("../../etc", entry())


class TestProcessedOn:
    """Where `voxmd status` gets "files processed today" from."""

    def entry(self, when: datetime) -> LedgerEntry:
        return LedgerEntry(source="/memo.m4a", size=1, processed_at=when, note="/note.md")

    def test_it_counts_only_the_day_asked_for(self, tmp_path: Path) -> None:
        ledger = Ledger.load(tmp_path / "ledger.json", max_bytes=1 << 20)
        ledger.record("a" * 64, self.entry(datetime(2026, 9, 15, 0, 1).astimezone()))
        ledger.record("b" * 64, self.entry(datetime(2026, 9, 15, 23, 59).astimezone()))
        ledger.record("c" * 64, self.entry(datetime(2026, 9, 16, 9, 0).astimezone()))

        assert ledger.processed_on(date(2026, 9, 15)) == 2
        assert ledger.processed_on(date(2026, 9, 16)) == 1
        assert ledger.processed_on(date(2026, 9, 17)) == 0

    def test_an_entry_written_in_another_timezone_counts_in_this_one(self, tmp_path: Path) -> None:
        ledger = Ledger.load(tmp_path / "ledger.json", max_bytes=1 << 20)
        moment = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
        ledger.record("d" * 64, self.entry(moment))

        assert ledger.processed_on(moment.astimezone().date()) == 1

    def test_an_empty_ledger_counts_nothing(self, tmp_path: Path) -> None:
        ledger = Ledger.load(tmp_path / "ledger.json", max_bytes=1 << 20)
        assert ledger.processed_on(date.today()) == 0
