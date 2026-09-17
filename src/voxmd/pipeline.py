"""Stage 5: one recording in, one note in the vault.

``voxmd process`` runs the four stages in a fixed order, and the order is what
keeps it safe:

1. **Cheap checks first.** The vault exists, the template compiles, the
   entities file is valid, the audio is acceptable. A typo fails in
   milliseconds, not after a minute of transcription.
2. **Know what's done.** The recording is hashed and looked up in the ledger.
   New audio, or a new upload of audio seen before, gets a note (``Title 2.md``
   for a repeat). The same file left untouched where it was processed is
   skipped unless forced, so a watcher restart doesn't duplicate its own
   leftovers. ``vault.duplicates: skip`` skips every repeat instead.
3. **Transcribe.** whisper-cli has exited and been reaped when this returns,
   and that is then *checked*: voxmd must have no child process left before
   Ollama is asked to load a model. The two models are never resident together.
4. **Extract**, with ``keep_alive=0``, so Ollama unloads as soon as it answers.
5. **Write the transcript, then the note** that links to it, each under a name
   no other file has, never replacing one, and read back to verify.
6. **Only then** record it in the ledger, append new entities, and move the
   recording to the archive. Any failure before this point leaves the
   recording exactly where it was, so the next run retries it.

One run at a time: a lock in the state directory makes a second concurrent
``voxmd process`` fail at once, instead of fighting the first for every core.
"""

from __future__ import annotations

import contextlib
import os
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import extract as extract_stage
from . import safe
from .config import ArchiveConfig, Config, VaultConfig
from .entities import link_target, load_entities, update_entities
from .errors import (
    ConfigError,
    InputError,
    OutputError,
    PartialFailure,
    ToolFailure,
    ToolTimeout,
    VoxmdError,
)
from .extract import ExtractionResult
from .ledger import LEDGER_NAME, Ledger, LedgerEntry, hash_file
from .render import NoteMeta, load_template, render_note, render_transcript
from .transcribe import AUDIO_SUFFIXES, TranscriptionResult, transcribe

LOCK_NAME = "process.lock"
LOCK_TIMEOUT_S = 1.0
MAX_TITLE_BYTES = 120
FALLBACK_TITLE = "Voice memo"
TRANSCRIPT_SUFFIX = " (transcript)"


@dataclass(frozen=True)
class ProcessResult:
    """What one ``process`` run did."""

    source: Path
    note: Path
    skipped: bool = False
    """Already in the ledger; nothing was run or written."""
    processed_at: datetime | None = None
    """For a skipped recording, when it was first processed."""
    transcript: Path | None = None
    archived: Path | None = None
    entities_added: int = 0
    transcription: TranscriptionResult | None = None
    extraction: ExtractionResult | None = None
    seconds: float = 0.0
    problems: tuple[str, ...] = ()
    """Steps after the note was written that failed. The note itself is fine."""


def process(
    audio: Path | str,
    *,
    settings: Config,
    created: datetime | None = None,
    force: bool = False,
    archive: bool = True,
    lock_timeout_s: float = LOCK_TIMEOUT_S,
    on_stage: Callable[[str], None] | None = None,
    client: Any = None,
) -> ProcessResult:
    """Turn one recording into a note in the vault.

    ``created`` overrides the recording time, which otherwise comes from the
    container's creation_time tag, then the file's modification time.
    ``lock_timeout_s`` is how long to wait for another voxmd run to finish: a
    person at a prompt wants to be told straight away, so the default is a
    second, while ``voxmd watch`` passes minutes because it has nothing better
    to do than wait. ``client`` is an injectable Ollama client for tests.
    """
    started = time.monotonic()
    stage = on_stage or (lambda _name: None)
    limits = settings.limits

    notes_dir, vault_root = resolve_notes_dir(settings.vault)
    archive_dir = resolve_archive_dir(settings.archive) if archive else None
    template, template_label = load_template(
        settings.render.template, max_bytes=limits.max_template_bytes
    )
    known = load_entities(
        settings.entities.file,
        max_bytes=limits.max_entities_bytes,
        threshold=settings.entities.fuzzy_threshold,
    )
    given = Path(audio).expanduser()
    source = safe.resolve_input_file(
        given, allowed_suffixes=AUDIO_SUFFIXES, max_bytes=limits.max_audio_bytes
    )
    state_dir = prepare_state_dir(settings.state.dir)

    with contextlib.ExitStack() as stack:
        _lock_run(stack, state_dir, lock_timeout_s)
        ledger = Ledger.load(state_dir / LEDGER_NAME, max_bytes=limits.max_ledger_bytes)

        before = source.stat()
        stage("hashing")
        digest = hash_file(source, max_bytes=limits.max_audio_bytes)
        seen = ledger.get(digest)
        if seen is not None and not force and should_skip(seen, source, before, settings):
            return ProcessResult(
                source=source,
                note=Path(seen.note),
                skipped=True,
                processed_at=seen.processed_at,
                seconds=time.monotonic() - started,
            )

        stage("transcribing")
        transcription = transcribe(source, whisper=settings.whisper, limits=limits)
        assert_no_child_processes()
        check_unchanged(source, before)

        stage("extracting")
        extraction = extract_stage.extract(
            transcription.text, ollama_cfg=settings.ollama, limits=limits, client=client
        )

        stage("writing")
        probe = transcription.probe
        when = created or probe.created or _modified(before)
        meta = NoteMeta(created=when, source=source.name, duration_s=probe.duration_s)
        name = note_filename(extraction.extraction.title, when)
        problems: list[str] = []

        # The transcript goes first, so the note can link to the name it
        # actually got. It is secondary: failing to save it costs the link,
        # not the note.
        transcript = None
        if settings.vault.transcripts:
            try:
                transcript = write_transcript(
                    notes_dir / settings.vault.transcripts_folder,
                    vault_root,
                    name,
                    render_transcript(
                        transcription.text,
                        title=extraction.extraction.title,
                        meta=meta,
                        max_bytes=limits.max_transcript_bytes,
                    ),
                )
            except VoxmdError as exc:
                problems.append(f"The transcript was not saved: {exc}")

        try:
            link = transcript.relative_to(vault_root).with_suffix("") if transcript else None
            rendered = render_note(
                extraction.extraction,
                meta=replace(meta, transcript=link.as_posix() if link else None),
                entities=known,
                template=template,
                template_label=template_label,
            )
            note = write_note(notes_dir, vault_root, name, rendered.markdown)
        except BaseException:
            # A transcript with no note would be an orphan, and the retry would
            # write a second one. It is voxmd's own file from a moment ago.
            if transcript is not None:
                transcript.unlink(missing_ok=True)
            raise

        entry = LedgerEntry(
            source=str(source),
            size=before.st_size,
            processed_at=datetime.now().astimezone(),
            note=str(note),
            transcript=str(transcript) if transcript else None,
            duration_s=probe.duration_s,
            mtime_ns=before.st_mtime_ns,
            earlier=[*seen.earlier, seen.processed_at] if seen is not None else [],
        )
        try:
            ledger.record(digest, entry)
        except VoxmdError as exc:
            raise PartialFailure(
                f"The note was written to {note}, but the ledger was not updated: {exc}\n"
                "The recording was left in place. Processing it again writes a second note."
            ) from exc

        added = 0
        try:
            added = update_entities(
                settings.entities.file,
                people=rendered.people.new,
                topics=rendered.topics.new,
                max_bytes=limits.max_entities_bytes,
                threshold=settings.entities.fuzzy_threshold,
            )
        except VoxmdError as exc:
            problems.append(f"The entities file was not updated: {exc}")

        archived = None
        if archive_dir is not None:
            try:
                # A sync client may have rewritten it since it was hashed; moving
                # it now would archive audio the ledger never saw.
                check_unchanged(source, before)
                archived = archive_recording(given, source, archive_dir)
            except (VoxmdError, OSError) as exc:
                problems.append(f"The recording was not archived: {_reason(exc)}")
            if archived is not None:
                try:
                    ledger.record(digest, entry.model_copy(update={"archived": str(archived)}))
                except VoxmdError as exc:
                    problems.append(
                        f"The recording was archived to {archived}, but the ledger doesn't "
                        f"say so: {exc}"
                    )

    return ProcessResult(
        source=source,
        note=note,
        transcript=transcript,
        archived=archived,
        entities_added=added,
        transcription=transcription,
        extraction=extraction,
        seconds=time.monotonic() - started,
        problems=tuple(problems),
    )


def should_skip(seen: LedgerEntry, source: Path, info: os.stat_result, settings: Config) -> bool:
    """Whether a recording whose audio is already in the ledger is left alone.

    With ``vault.duplicates: skip``, always. With ``copy``, only when it is the
    very same file, untouched where it was processed, and never archived: that
    is a watcher restart finding its own leftovers, not a new upload.
    """
    if settings.vault.duplicates == "skip":
        return True
    same_place = seen.source == str(source) and seen.archived is None
    untouched = seen.mtime_ns is None or seen.mtime_ns == info.st_mtime_ns
    return same_place and untouched


def resolve_notes_dir(vault: VaultConfig) -> tuple[Path, Path]:
    """The folder notes go in and the vault root, both resolved. Creates nothing."""
    if vault.path is None:
        raise ConfigError(
            "No vault configured.\n"
            "Set vault.path in your config, or pass --vault /path/to/your/vault."
        )
    try:
        root = vault.path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"Vault not found: {vault.path}\nvoxmd never creates the vault; check vault.path."
        ) from exc
    if not root.is_dir():
        raise ConfigError(f"Vault is not a directory: {root}")
    notes = root / vault.folder if vault.folder is not None else root
    return contained_in(notes, root), root


def contained_in(path: Path, root: Path) -> Path:
    """``path`` resolved, refused if it leads outside ``root`` (a symlinked folder, say)."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ConfigError(f"{path} leads outside the vault ({root}); refusing to write there.")
    return resolved


def resolve_archive_dir(archive: ArchiveConfig) -> Path | None:
    """The archive folder, or None when not configured. Creates nothing."""
    directory = archive.dir
    if directory is None:
        return None
    if directory.exists() and not directory.is_dir():
        raise ConfigError(f"archive.dir is not a directory: {directory}")
    if not directory.exists() and not directory.parent.is_dir():
        raise ConfigError(
            f"archive.dir's parent folder does not exist: {directory.parent}\n"
            "Create it, or fix archive.dir."
        )
    return directory


def prepare_state_dir(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(
            f"Could not create the state directory {path}: {exc.strerror or exc}"
        ) from exc
    return path


def assert_no_child_processes() -> None:
    """Fail unless this process has no children left, i.e. whisper-cli is really gone.

    ``WNOWAIT`` only looks: it doesn't reap anything, so the check has no side
    effects. No children at all raises ChildProcessError, which is the pass case.
    """
    if not hasattr(os, "waitid"):  # pragma: no cover - voxmd targets macOS
        return
    try:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return
    raise ToolFailure(
        "A child process (whisper-cli) is still running, so the Ollama model was not loaded. "
        "voxmd never holds both models in memory at once."
    )


def check_unchanged(source: Path, before: os.stat_result) -> None:
    """Refuse a recording that changed after it was hashed, e.g. one still syncing."""
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise InputError(
            f"{source.name} changed while it was being processed. "
            "If it is still syncing, wait and run again."
        )


def note_filename(title: str, created: datetime) -> str:
    """``2026-09-15 Ship date.md``: the date first, so notes sort by recording time.

    The title passes through the same filter as wikilink targets, so it holds no
    path separator, no leading dot, and nothing Obsidian can't use in a name.
    """
    name = unicodedata.normalize("NFC", link_target(title))
    name = name.encode("utf-8")[:MAX_TITLE_BYTES].decode("utf-8", errors="ignore").strip(" .")
    return f"{created:%Y-%m-%d} {name or FALLBACK_TITLE}.md"


def write_note(directory: Path, vault_root: Path, name: str, markdown: str) -> Path:
    """Write the note as a new file in ``directory`` and verify it reads back intact."""
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Checked again now the folder exists, in case it is a symlink out of the vault.
        directory = contained_in(directory, vault_root)
        note = safe.write_new_file(directory, name, markdown)
        written = note.read_bytes()
    except OSError as exc:
        raise OutputError(
            f"Could not write the note into {directory}: {exc.strerror or exc}\n"
            "The recording was left in place."
        ) from exc
    if written != markdown.encode("utf-8"):
        raise OutputError(
            f"The note at {note} does not match what was written. The recording was left in place."
        )
    return note


def transcript_filename(note_name: str) -> str:
    """``2026-09-15 Ship date (transcript).md`` for ``2026-09-15 Ship date.md``."""
    return f"{Path(note_name).stem}{TRANSCRIPT_SUFFIX}.md"


def write_transcript(directory: Path, vault_root: Path, note_name: str, markdown: str) -> Path:
    """Save the transcript note, with the same containment and no-overwrite rules as a note."""
    try:
        return write_note(directory, vault_root, transcript_filename(note_name), markdown)
    except OutputError as exc:
        reason = _reason(exc.__cause__) if exc.__cause__ else "it did not read back intact"
        raise OutputError(f"could not write into {directory}: {reason}") from exc


def archive_recording(given: Path, source: Path, directory: Path) -> Path | None:
    """Move the recording into the archive. None when it is already there."""
    if given.is_symlink():
        raise InputError(
            f"{given.name} is a symlink, so it was left in place rather than moving "
            "the file it points to."
        )
    directory.mkdir(mode=0o700, exist_ok=True)
    resolved = directory.resolve()
    if source.parent == resolved:
        return None
    return safe.move_no_replace(source, resolved)


def _lock_run(stack: contextlib.ExitStack, state_dir: Path, timeout_s: float) -> None:
    try:
        stack.enter_context(
            safe.file_lock(state_dir / LOCK_NAME, timeout_s=timeout_s, what=str(state_dir))
        )
    except ToolTimeout as exc:
        raise ToolTimeout(
            "Another voxmd process is already running. Let it finish, then try again."
        ) from exc
    except OSError as exc:
        raise OutputError(f"Could not lock {state_dir}: {exc.strerror or exc}") from exc


def _modified(info: os.stat_result) -> datetime:
    return datetime.fromtimestamp(info.st_mtime).replace(second=0, microsecond=0)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return str(exc)
