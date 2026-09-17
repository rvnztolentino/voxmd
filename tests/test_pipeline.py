"""Stage 5: the whole pipeline, with every external tool faked."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import VALID_EXTRACTION, FakeClient, FakeRunner, chat_reply, probe_json
from voxmd import pipeline, safe
from voxmd.config import Config
from voxmd.errors import (
    ConfigError,
    DependencyError,
    InputError,
    OutputError,
    PartialFailure,
    ToolFailure,
    ToolTimeout,
)
from voxmd.ledger import LEDGER_NAME, Ledger
from voxmd.pipeline import FALLBACK_TITLE, ProcessResult, note_filename, process

TRANSCRIPT = "Marco and Ana agreed to ship on Friday."
RECORDED = datetime(2026, 9, 15, 14, 3)
NOTE_NAME = "2026-09-15 Ship date.md"


@dataclass
class Env:
    root: Path
    settings: Config
    audio: Path
    data: bytes
    client: FakeClient
    runner: FakeRunner

    @property
    def notes(self) -> Path:
        return self.root / "vault" / "Memos"

    @property
    def archive(self) -> Path:
        return self.root / "archive"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def entities(self) -> Path:
        return self.root / "entities.json"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def run(self, **kwargs: object) -> ProcessResult:
        return process(self.audio, settings=self.settings, client=self.client, **kwargs)

    def ledger(self) -> Ledger:
        return Ledger.load(self.state / LEDGER_NAME, max_bytes=1 << 20)

    def restore_audio(self) -> None:
        self.audio.write_bytes(self.data)
        stamp = RECORDED.timestamp()
        os.utime(self.audio, (stamp, stamp))

    def settings_with(self, **sections: dict[str, object]) -> Config:
        data = self.settings.model_dump(mode="json")
        for key, values in sections.items():
            data[key] = {**data[key], **values}
        return Config.model_validate(data)


@pytest.fixture
def env(tmp_path: Path, fake_run: FakeRunner, fake_tools: None, model_file: Path) -> Env:
    (tmp_path / "vault").mkdir()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    fake_run.set("ffprobe", stdout=probe_json())
    fake_run.set("whisper-cli", stdout=TRANSCRIPT)
    settings = Config.model_validate(
        {
            "whisper": {"model": str(model_file)},
            "vault": {"path": str(tmp_path / "vault"), "folder": "Memos"},
            "archive": {"dir": str(tmp_path / "archive")},
            "state": {"dir": str(tmp_path / "state")},
            "entities": {"file": str(tmp_path / "entities.json")},
        }
    )
    environment = Env(
        root=tmp_path,
        settings=settings,
        audio=inbox / "memo.m4a",
        data=os.urandom(4096),
        client=FakeClient([chat_reply(), chat_reply()]),
        runner=fake_run,
    )
    environment.restore_audio()
    return environment


def only_note(env: Env) -> Path:
    (note,) = env.notes.glob("*.md")
    return note


def only_transcript(env: Env) -> Path:
    (transcript,) = (env.notes / "Transcripts").glob("*.md")
    return transcript


# --- the happy path ---------------------------------------------------------


def test_a_recording_becomes_a_note_and_then_moves_to_the_archive(env: Env) -> None:
    result = env.run()

    assert result.note == env.notes / NOTE_NAME
    text = result.note.read_text()
    assert text.startswith(
        "---\ndate: 2026-09-15T14:03\nsource: memo.m4a\nduration: '2:05'\n---\n\n# Ship date\n"
    )
    assert "- [ ] Email the client" in text
    assert result.note.stat().st_mode & 0o777 == 0o600

    assert result.archived == env.archive / "memo.m4a"
    assert result.archived.read_bytes() == env.data
    assert not env.audio.exists()

    recorded = env.ledger().get(env.digest)
    assert recorded is not None
    assert recorded.note == str(result.note)
    assert recorded.archived == str(result.archived)
    assert recorded.duration_s == 125.0

    assert result.entities_added == 3
    assert json.loads(env.entities.read_text()) == {
        "people": ["Marco", "Ana"],
        "topics": ["release"],
    }
    assert result.problems == ()


def test_stages_are_reported_in_order(env: Env) -> None:
    stages: list[str] = []

    env.run(on_stage=stages.append)

    assert stages == ["hashing", "transcribing", "extracting", "writing"]


def test_no_folder_configured_writes_to_the_vault_root(env: Env) -> None:
    settings = env.settings_with(vault={"folder": None})

    result = process(env.audio, settings=settings, client=env.client)

    assert result.note == (env.root / "vault" / NOTE_NAME)


def test_the_ledger_holds_no_transcript_or_extracted_content(env: Env) -> None:
    env.run()

    ledger = (env.state / LEDGER_NAME).read_text()
    for private in (TRANSCRIPT, VALID_EXTRACTION["summary"], "Marco", "Email the client"):
        assert private not in ledger


# --- never both models resident (C5) ----------------------------------------


def test_whisper_has_exited_and_been_checked_before_ollama_is_called(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_transcribe = pipeline.transcribe
    real_check = pipeline.assert_no_child_processes
    real_chat = env.client.chat

    def transcribe(*args: object, **kwargs: object) -> object:
        events.append("transcribe")
        return real_transcribe(*args, **kwargs)  # type: ignore[arg-type]

    def check() -> None:
        events.append("no children")
        real_check()

    def chat(**kwargs: object) -> object:
        events.append("ollama")
        return real_chat(**kwargs)

    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(pipeline, "assert_no_child_processes", check)
    monkeypatch.setattr(env.client, "chat", chat)

    env.run()

    assert events == ["transcribe", "no children", "ollama"]
    assert env.client.chats[0]["keep_alive"] == 0


def test_a_child_process_still_running_stops_the_run_before_ollama(env: Env) -> None:
    child = subprocess.Popen(  # noqa: S603 - fixed argv, test only
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    try:
        with pytest.raises(ToolFailure, match="never holds both models"):
            env.run()
    finally:
        child.kill()
        child.wait()

    assert env.client.chats == []
    assert env.audio.exists()
    assert not env.notes.exists()


def test_the_child_process_check_is_real_on_this_platform() -> None:
    """macOS gained os.waitid in Python 3.13, which is why voxmd requires 3.13."""
    assert hasattr(os, "waitid")


def test_the_child_process_check_passes_with_no_children() -> None:
    pipeline.assert_no_child_processes()


# --- idempotency ------------------------------------------------------------


def test_the_same_audio_uploaded_again_gets_a_second_note(env: Env) -> None:
    """A re-upload is a request for a note, even of audio seen before."""
    first = env.run()
    original = first.note.read_text()
    env.restore_audio()

    second = env.run()

    assert not second.skipped
    assert second.note == env.notes / "2026-09-15 Ship date 2.md"
    assert second.transcript is not None and second.transcript.name.endswith("(transcript) 2.md")
    assert first.note.read_text() == original
    assert second.archived == env.archive / "memo 2.m4a"


def test_the_same_audio_under_another_name_also_gets_a_note(env: Env) -> None:
    first = env.run()
    again = env.audio.with_name("renamed.m4a")
    again.write_bytes(env.data)

    result = process(again, settings=env.settings, client=env.client)

    assert not result.skipped
    assert result.note != first.note
    assert len(env.client.chats) == 2


def test_a_file_left_untouched_where_it_was_processed_is_not_redone(env: Env) -> None:
    """Without an archive, a watcher restart must not duplicate its own leftovers."""
    env.settings = env.settings_with(archive={"dir": None})
    first = env.run()
    calls = len(env.runner.calls)

    result = env.run()

    assert result.skipped
    assert result.note == first.note
    assert result.processed_at is not None
    assert len(env.runner.calls) == calls
    assert len(env.client.chats) == 1
    assert len(list(env.notes.glob("*.md"))) == 1


def test_a_file_replaced_in_place_counts_as_a_new_upload(env: Env) -> None:
    env.settings = env.settings_with(archive={"dir": None})
    env.run()
    stamp = env.audio.stat().st_mtime + 60
    os.utime(env.audio, (stamp, stamp))

    result = env.run()

    assert not result.skipped
    assert result.note.name == "2026-09-15 Ship date 2.md"


def test_an_entry_from_before_mtimes_were_recorded_still_protects_leftovers(env: Env) -> None:
    env.settings = env.settings_with(archive={"dir": None})
    env.run()
    ledger = env.ledger()
    old = ledger.get(env.digest)
    assert old is not None
    ledger.record(env.digest, old.model_copy(update={"mtime_ns": None}))

    assert env.run().skipped


def test_skip_mode_brings_back_skipping_every_repeat(env: Env) -> None:
    env.settings = env.settings_with(vault={"duplicates": "skip"})
    first = env.run()
    again = env.audio.with_name("renamed.m4a")
    again.write_bytes(env.data)
    calls = len(env.runner.calls)

    result = process(again, settings=env.settings, client=env.client)

    assert result.skipped
    assert result.note == first.note
    assert len(env.runner.calls) == calls
    assert again.exists()


def test_every_copy_counts_toward_processed_today(env: Env) -> None:
    env.run()
    env.restore_audio()
    env.run()
    env.restore_audio()
    env.client.replies.append(chat_reply())
    env.run()

    entry = env.ledger().get(env.digest)
    assert entry is not None and len(entry.earlier) == 2
    assert env.ledger().processed_on(entry.processed_at.astimezone().date()) == 3


def test_force_writes_a_second_note_and_never_replaces_the_first(env: Env) -> None:
    first = env.run()
    original = first.note.read_text()
    env.restore_audio()

    second = env.run(force=True)

    assert second.note == env.notes / "2026-09-15 Ship date 2.md"
    assert first.note.read_text() == original
    assert second.archived == env.archive / "memo 2.m4a"


def test_an_existing_note_with_the_same_name_is_never_replaced(env: Env) -> None:
    env.notes.mkdir(parents=True)
    mine = env.notes / NOTE_NAME
    mine.write_text("my own note")

    result = env.run()

    assert mine.read_text() == "my own note"
    assert result.note.name == "2026-09-15 Ship date 2.md"


# --- failures leave the recording where it was ------------------------------


def test_an_ollama_failure_leaves_the_recording_in_place(env: Env) -> None:
    env.client.replies[:] = [ConnectionError("refused")]

    with pytest.raises(DependencyError):
        env.run()

    assert env.audio.exists()
    assert not env.notes.exists()
    assert len(env.ledger()) == 0


def test_a_failed_note_write_leaves_the_recording_and_ledger_untouched(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(*args: object, **kwargs: object) -> Path:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(safe, "write_new_file", disk_full)

    with pytest.raises(OutputError, match="No space left"):
        env.run()

    assert env.audio.exists()
    assert len(env.ledger()) == 0
    assert not env.entities.exists()


def test_a_failed_ledger_write_is_a_partial_failure_that_keeps_the_recording(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(self: Ledger, digest: str, entry: object) -> None:
        raise OutputError("disk full")

    monkeypatch.setattr(Ledger, "record", fail)

    with pytest.raises(PartialFailure, match="left in place"):
        env.run()

    assert env.audio.exists()
    assert only_note(env).name == NOTE_NAME


def test_a_failed_archive_is_reported_and_the_note_stands(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*args: object, **kwargs: object) -> Path:
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(safe, "move_no_replace", denied)

    result = env.run()

    assert result.problems == ("The recording was not archived: Permission denied",)
    assert result.note.exists()
    assert env.audio.exists()
    recorded = env.ledger().get(env.digest)
    assert recorded is not None
    assert recorded.archived is None


def test_a_failed_entities_update_is_reported_and_archiving_still_happens(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def locked(*args: object, **kwargs: object) -> int:
        raise ConfigError("entities are locked")

    monkeypatch.setattr(pipeline, "update_entities", locked)

    result = env.run()

    assert result.problems == ("The entities file was not updated: entities are locked",)
    assert result.archived is not None


def test_a_recording_that_changes_after_its_note_is_written_is_not_archived(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_update = pipeline.update_entities

    def sync_rewrites_it(*args: object, **kwargs: object) -> int:
        with env.audio.open("ab") as handle:
            handle.write(b"more audio")
        return real_update(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "update_entities", sync_rewrites_it)

    result = env.run()

    assert result.archived is None
    assert "changed while" in result.problems[0]
    assert env.audio.exists()
    assert result.note.exists()


def test_a_recording_that_changes_during_processing_is_refused(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_transcribe = pipeline.transcribe

    def still_syncing(source: Path, **kwargs: object) -> object:
        result = real_transcribe(source, **kwargs)  # type: ignore[arg-type]
        with env.audio.open("ab") as handle:
            handle.write(b"more audio")
        return result

    monkeypatch.setattr(pipeline, "transcribe", still_syncing)

    with pytest.raises(InputError, match="changed while"):
        env.run()

    assert env.client.chats == []
    assert env.audio.exists()


# --- archive options --------------------------------------------------------


def test_no_archive_leaves_the_recording_in_place(env: Env) -> None:
    result = env.run(archive=False)

    assert result.archived is None
    assert env.audio.exists()
    assert not env.archive.exists()


def test_without_an_archive_dir_the_recording_stays(env: Env) -> None:
    settings = env.settings_with(archive={"dir": None})

    result = process(env.audio, settings=settings, client=env.client)

    assert result.archived is None
    assert env.audio.exists()


def test_a_symlinked_recording_is_not_archived(env: Env) -> None:
    real = env.root / "elsewhere" / "memo.m4a"
    real.parent.mkdir()
    env.audio.rename(real)
    env.audio.symlink_to(real)

    result = env.run()

    assert "symlink" in result.problems[0]
    assert real.read_bytes() == env.data
    assert env.audio.is_symlink()
    assert result.note.exists()


# --- cheap checks run before any work ---------------------------------------


def test_a_missing_vault_fails_before_anything_runs_or_is_created(env: Env) -> None:
    settings = env.settings_with(vault={"path": str(env.root / "nope")})

    with pytest.raises(ConfigError, match="Vault not found"):
        process(env.audio, settings=settings, client=env.client)

    assert env.runner.calls == []
    assert not (env.root / "nope").exists()
    assert not env.state.exists()


def test_no_vault_configured_says_how_to_set_one(env: Env) -> None:
    settings = env.settings_with(vault={"path": None})

    with pytest.raises(ConfigError, match="--vault"):
        process(env.audio, settings=settings, client=env.client)
    assert env.runner.calls == []


def test_a_notes_folder_symlinked_out_of_the_vault_is_refused(env: Env) -> None:
    outside = env.root / "outside"
    outside.mkdir()
    env.notes.symlink_to(outside)

    with pytest.raises(ConfigError, match="outside the vault"):
        env.run()

    assert env.runner.calls == []
    assert list(outside.iterdir()) == []


def test_a_corrupt_ledger_is_refused_before_transcribing(env: Env) -> None:
    env.state.mkdir()
    ledger = env.state / LEDGER_NAME
    ledger.write_text("{")

    with pytest.raises(ConfigError, match="not valid JSON"):
        env.run()

    assert ledger.read_text() == "{"
    assert env.runner.calls == []


def test_a_second_concurrent_run_is_refused(env: Env) -> None:
    env.state.mkdir()

    with (
        safe.file_lock(env.state / pipeline.LOCK_NAME, timeout_s=1, what="test"),
        pytest.raises(ToolTimeout, match="Another voxmd process"),
    ):
        env.run(lock_timeout_s=0.1)

    assert env.runner.calls == []


def test_a_person_at_a_prompt_is_told_within_a_second(env: Env) -> None:
    """The default is short on purpose: the watcher passes a longer one."""
    assert pipeline.LOCK_TIMEOUT_S == 1.0

    env.state.mkdir()
    started = time.monotonic()
    with (
        safe.file_lock(env.state / pipeline.LOCK_NAME, timeout_s=1, what="test"),
        pytest.raises(ToolTimeout),
    ):
        env.run()

    assert time.monotonic() - started < 2.0


def test_a_caller_that_can_wait_gets_the_lock_when_it_is_released(env: Env) -> None:
    """How `voxmd watch` shares the machine with a manual run instead of failing."""
    env.state.mkdir()
    lock = safe.file_lock(env.state / pipeline.LOCK_NAME, timeout_s=1, what="test")
    lock.__enter__()
    threading.Timer(0.15, lambda: lock.__exit__(None, None, None)).start()

    result = env.run(lock_timeout_s=10)

    assert result.note.exists()


# --- recording time ---------------------------------------------------------


def test_the_containers_creation_time_dates_the_note(env: Env) -> None:
    payload = json.loads(probe_json())
    payload["format"]["tags"] = {"creation_time": "2026-03-01T12:00:00.000000Z"}
    env.runner.set("ffprobe", stdout=json.dumps(payload))
    local = datetime(2026, 3, 1, 12, tzinfo=UTC).astimezone()

    result = env.run()

    assert result.note.name == f"{local:%Y-%m-%d} Ship date.md"
    assert f"date: {local:%Y-%m-%dT%H:%M}\n" in result.note.read_text()


def test_an_explicit_date_wins(env: Env) -> None:
    result = env.run(created=datetime(2025, 1, 2, 9, 30))

    assert result.note.name == "2025-01-02 Ship date.md"


# --- note file names --------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Ship date", "Ship date"),
        ("../../etc/passwd", "etc passwd"),
        ("..", FALLBACK_TITLE),
        ("", FALLBACK_TITLE),
        (".hidden", "hidden"),
        ('a/b\\c:d*e?f"g<h>i|j', "a b c d e f g h i j"),
        ("[[Evil]] #tag ^block", "Evil tag block"),
        ("Café ☕", "Café ☕"),
    ],
)
def test_note_filenames_are_safe_in_a_vault(title: str, expected: str) -> None:
    assert note_filename(title, RECORDED) == f"2026-09-15 {expected}.md"


def test_long_titles_are_cut_on_a_character_boundary() -> None:
    name = note_filename("é" * 200, RECORDED)

    assert len(name.encode("utf-8")) <= len("2026-09-15 .md") + pipeline.MAX_TITLE_BYTES
    assert "�" not in name
    assert name.startswith("2026-09-15 éé")


def test_a_hostile_title_still_lands_inside_the_notes_folder(env: Env) -> None:
    hostile = {**VALID_EXTRACTION, "title": "../../../../tmp/escape"}
    env.client.replies[:] = [chat_reply(hostile)]

    result = env.run()

    assert result.note.parent == env.notes.resolve()
    assert result.note.name == "2026-09-15 tmp escape.md"


# --- transcripts ------------------------------------------------------------


def test_the_transcript_is_saved_as_its_own_note_and_linked(env: Env) -> None:
    result = env.run()

    transcript = only_transcript(env)
    assert result.transcript == transcript
    assert transcript.name == "2026-09-15 Ship date (transcript).md"
    assert oct(transcript.stat().st_mode & 0o777) == "0o600"
    text = transcript.read_text()
    assert text.startswith("---\ndate: 2026-09-15T14:03\nsource: memo.m4a\n")
    assert "# Ship date (transcript)" in text
    assert TRANSCRIPT in text
    assert "may be misheard" in text
    assert (
        "- [[Memos/Transcripts/2026-09-15 Ship date (transcript)|Full transcript]]"
        in only_note(env).read_text()
    )
    recorded = env.ledger().get(env.digest)
    assert recorded is not None and recorded.transcript == str(transcript)


def test_the_link_follows_the_name_the_transcript_actually_got(env: Env) -> None:
    """With --force the second transcript is numbered; its note must point at it."""
    env.run()
    env.restore_audio()
    env.client.replies.append(chat_reply())

    second = env.run(force=True)

    assert second.transcript is not None
    assert second.transcript.name == "2026-09-15 Ship date (transcript) 2.md"
    assert "Ship date (transcript) 2|Full transcript]]" in second.note.read_text()


def test_transcripts_can_be_switched_off(env: Env) -> None:
    env.settings = env.settings_with(vault={"transcripts": False})

    result = env.run()

    assert result.transcript is None
    assert not (env.notes / "Transcripts").exists()
    assert "Full transcript" not in result.note.read_text()


def test_the_transcripts_folder_is_configurable(env: Env) -> None:
    env.settings = env.settings_with(vault={"transcripts_folder": "Raw/Text"})

    result = env.run()

    assert result.transcript == env.notes / "Raw" / "Text" / result.transcript.name
    assert "[[Memos/Raw/Text/" in result.note.read_text()


def test_a_transcript_that_cannot_be_saved_costs_the_link_not_the_note(env: Env) -> None:
    (env.notes).mkdir(parents=True)
    (env.notes / "Transcripts").write_text("a file where the folder should be")

    result = env.run()

    assert result.note.exists()
    assert result.transcript is None
    assert "Full transcript" not in result.note.read_text()
    assert len(result.problems) == 1
    assert result.problems[0].startswith("The transcript was not saved: could not write into")
    assert not env.audio.exists(), "a missing transcript doesn't stop archiving"


def test_a_note_that_fails_takes_its_new_transcript_with_it(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the retry leaves an orphan behind and writes a numbered second one."""
    real = safe.write_new_file
    calls: list[str] = []

    def fail_the_note(directory: Path, name: str, text: str, **kwargs: object) -> Path:
        calls.append(name)
        if "(transcript)" in name:
            return real(directory, name, text, **kwargs)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(safe, "write_new_file", fail_the_note)

    with pytest.raises(OutputError, match="No space left"):
        env.run()

    assert len(calls) == 2
    assert list((env.notes / "Transcripts").iterdir()) == []
    assert env.audio.exists()
    assert len(env.ledger()) == 0


def test_a_transcript_is_escaped_like_model_output(env: Env) -> None:
    """A recording is just as untrusted as the model: it can't make links or embeds."""
    hostile = "Open [[Secret]] now. ![[x.png]] <img src=http://e.test/p> %%hide%% #tag. Done."
    env.runner.set("whisper-cli", stdout=hostile)

    env.run()

    body = only_transcript(env).read_text().split("---\n", 2)[2]
    assert re.search(r"(?<!\\)\[\[", body) is None
    assert re.search(r"(?<!\\)<", body) is None
    assert "%%" not in body.replace("\\%", "")
    assert "\\#tag" in body


def test_a_transcripts_folder_symlinked_out_of_the_vault_is_refused(env: Env) -> None:
    outside = env.root / "outside"
    outside.mkdir()
    env.notes.mkdir(parents=True)
    (env.notes / "Transcripts").symlink_to(outside)

    result = env.run()

    assert list(outside.iterdir()) == []
    assert result.transcript is None
    assert "outside the vault" in result.problems[0]


def test_the_transcript_is_split_into_readable_paragraphs(env: Env) -> None:
    sentences = " ".join(f"Sentence number {n}." for n in range(1, 10))
    env.runner.set("whisper-cli", stdout=sentences)

    env.run()

    body = only_transcript(env).read_text().split("\n# ", 1)[1]
    paragraphs = [p for p in body.split("\n\n")[1:] if p.startswith("Sentence")]
    assert len(paragraphs) == 3
    assert paragraphs[0] == " ".join(f"Sentence number {n}." for n in range(1, 5))


def test_a_title_cannot_smuggle_a_comment_marker_into_the_transcript_link(env: Env) -> None:
    """File names and links can't be escaped, so "%%" is reduced to "%" at the source."""
    env.client.replies[:] = [chat_reply({**VALID_EXTRACTION, "title": "Plan %%hidden 50%"})]

    result = env.run()

    assert result.transcript is not None
    assert "%%" not in result.note.name
    assert "%%" not in result.transcript.name
    assert "%%" not in result.note.read_text().replace("\\%", "")
