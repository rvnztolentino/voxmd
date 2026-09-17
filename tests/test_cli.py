"""The CLI surface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import (
    VALID_EXTRACTION,
    FakeClient,
    FakeOllama,
    FakeRunner,
    chat_reply,
    probe_json,
)
from voxmd import __version__, cli, errors
from voxmd import extract as extract_module
from voxmd.config import CONFIG_ENV_VAR

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_real_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_transcript_goes_to_stdout_and_nothing_else(
    fake_run: FakeRunner, fake_tools: None, memo: Path, model_file: Path
) -> None:
    fake_run.set("ffprobe", stdout=probe_json())
    fake_run.set("whisper-cli", stdout="Just the words.")

    result = runner.invoke(cli.app, ["transcribe", str(memo), "--model", str(model_file)])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Just the words.\n"


def test_verbose_details_go_to_stderr_only(
    fake_run: FakeRunner, fake_tools: None, memo: Path, model_file: Path
) -> None:
    fake_run.set("ffprobe", stdout=probe_json())
    fake_run.set("whisper-cli", stdout="Just the words.")

    result = runner.invoke(cli.app, ["transcribe", str(memo), "-m", str(model_file), "-v"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "Just the words.\n"
    assert "whisper:" in result.stderr
    assert "ffmpeg:" in result.stderr


def test_flags_override_config(
    fake_run: FakeRunner, fake_tools: None, memo: Path, model_file: Path, tmp_path: Path
) -> None:
    (tmp_path / "voxmd.yaml").write_text(
        "whisper:\n  language: fr\n  threads: 8\n  model: /nowhere.bin\n"
    )
    fake_run.set("ffprobe", stdout=probe_json())
    fake_run.set("whisper-cli", stdout="ok")

    result = runner.invoke(
        cli.app, ["transcribe", str(memo), "-m", str(model_file), "-l", "EN", "-t", "2"]
    )

    assert result.exit_code == 0, result.output
    (call,) = fake_run.called("whisper-cli")
    assert call.flag("-m") == str(model_file)
    assert call.flag("-l") == "en"
    assert call.flag("-t") == "2"


@pytest.mark.parametrize("bad", ["--help", "-f", "en; rm -rf ~"])
def test_language_flag_is_validated_like_a_config_value(
    fake_run: FakeRunner, fake_tools: None, memo: Path, model_file: Path, bad: str
) -> None:
    # A value starting with "-" would reach whisper-cli as a flag of its own.
    from voxmd.errors import ConfigError

    fake_run.set("ffprobe", stdout=probe_json())
    result = runner.invoke(
        cli.app, ["transcribe", str(memo), "-m", str(model_file), f"--language={bad}"]
    )

    assert isinstance(result.exception, ConfigError)
    assert fake_run.calls == []


def test_expected_errors_print_one_line_and_a_distinct_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["voxmd", "transcribe", str(tmp_path / "missing.m4a")])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 4  # AudioError
    err = capsys.readouterr().err
    assert "No such file" in err
    assert "Traceback" not in err


def test_help_imports_no_stage_modules_or_heavy_dependencies() -> None:
    """Startup cost guard. If this fails, an import escaped a command body."""
    probe = (
        "import sys, voxmd.cli; "
        "heavy = {'voxmd.config', 'voxmd.transcribe', 'voxmd.extract', 'voxmd.schema', "
        "'voxmd.render', 'voxmd.entities', 'voxmd.pipeline', 'voxmd.ledger', 'voxmd.doctor', "
        "'voxmd.watcher', 'voxmd.status', 'voxmd.logging_setup', "
        "'pydantic', 'yaml', 'ollama', 'httpx', 'jinja2', 'rapidfuzz', 'watchdog'}; "
        "loaded = sorted(heavy & set(sys.modules)); "
        "print(','.join(loaded))"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, test only
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=30
    )
    assert result.stdout.strip() == "", f"imported at startup: {result.stdout.strip()}"


def test_version_prints_the_package_version_and_nothing_else() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"voxmd {__version__}\n"


def test_version_matches_the_installed_metadata() -> None:
    """One source of truth: pyproject reads the version from ``__init__``."""
    from importlib.metadata import version

    assert version("voxmd") == __version__


# --- extract ----------------------------------------------------------------


@pytest.fixture
def ollama_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient([chat_reply()])
    monkeypatch.setattr(extract_module, "make_client", lambda *args, **kwargs: client)
    return client


def test_extract_prints_only_json_to_stdout(ollama_client: FakeClient, tmp_path: Path) -> None:
    transcript = tmp_path / "memo.txt"
    transcript.write_text("Marco and Ana agreed to ship Friday.")

    result = runner.invoke(cli.app, ["extract", str(transcript)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == VALID_EXTRACTION
    assert result.stderr == ""


def test_extract_reads_a_piped_transcript(ollama_client: FakeClient) -> None:
    result = runner.invoke(cli.app, ["extract"], input="Marco and Ana agreed.")

    assert result.exit_code == 0, result.output
    assert "Marco and Ana agreed." in ollama_client.chats[0]["messages"][1]["content"]


def test_extract_model_flag_overrides_config(ollama_client: FakeClient, tmp_path: Path) -> None:
    (tmp_path / "voxmd.yaml").write_text("ollama:\n  model: gemma3:4b\n")

    result = runner.invoke(cli.app, ["extract", "-m", "llama3.2:3b"], input="hi there")

    assert result.exit_code == 0, result.output
    assert ollama_client.chats[0]["model"] == "llama3.2:3b"


def test_extract_verbose_details_go_to_stderr_only(ollama_client: FakeClient) -> None:
    result = runner.invoke(cli.app, ["extract", "-v"], input="hi there")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == VALID_EXTRACTION
    assert "attempts: 1" in result.stderr


def test_extract_warns_on_stderr_when_the_transcript_did_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([chat_reply(prompt_tokens=100_000)])
    monkeypatch.setattr(extract_module, "make_client", lambda *args, **kwargs: client)

    result = runner.invoke(cli.app, ["extract"], input="hi there")

    assert result.exit_code == 0, result.output
    assert "truncated" in result.stderr
    assert json.loads(result.stdout) == VALID_EXTRACTION


def test_extract_refuses_a_remote_host_before_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxmd.errors import ConfigError

    (tmp_path / "voxmd.yaml").write_text("ollama:\n  host: http://10.0.0.5:11434\n")
    created: list[object] = []
    monkeypatch.setattr(extract_module, "make_client", lambda *args, **kwargs: created.append(1))

    result = runner.invoke(cli.app, ["extract"], input="hi there")

    assert isinstance(result.exception, ConfigError)
    assert created == []


# --- render -----------------------------------------------------------------


def test_render_imports_neither_ollama_nor_httpx() -> None:
    probe = "import sys, voxmd.render; print(sorted({'ollama', 'httpx'} & set(sys.modules)))"
    result = subprocess.run(  # noqa: S603 - fixed argv, test only
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=30
    )
    assert result.stdout.strip() == "[]"


def test_render_prints_only_the_note_and_writes_nothing(tmp_path: Path) -> None:
    extraction = tmp_path / "memo.json"
    extraction.write_text(json.dumps(VALID_EXTRACTION))

    result = runner.invoke(
        cli.app, ["render", str(extraction), "-s", "memo.m4a", "--date", "2026-09-15T14:03"]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("---\ndate: 2026-09-15T14:03\nsource: memo.m4a\n---\n")
    assert "# Ship date" in result.stdout
    assert result.stderr == ""
    assert not (tmp_path / "home" / ".config").exists()


def test_render_reads_a_piped_extraction() -> None:
    result = runner.invoke(cli.app, ["render"], input=json.dumps(VALID_EXTRACTION))

    assert result.exit_code == 0, result.output
    assert "- [ ] Email the client" in result.stdout


def test_render_links_known_entities_and_saves_new_ones_only_when_asked(tmp_path: Path) -> None:
    entities = tmp_path / "entities.json"
    entities.write_text('{"people": ["Marco"]}')
    payload = json.dumps(VALID_EXTRACTION)

    first = runner.invoke(cli.app, ["render", "-e", str(entities)], input=payload)
    assert "- People: [[Marco]], Ana" in first.stdout
    assert json.loads(entities.read_text()) == {"people": ["Marco"]}

    second = runner.invoke(
        cli.app, ["render", "-e", str(entities), "--update-entities"], input=payload
    )
    assert second.exit_code == 0, second.output
    assert json.loads(entities.read_text()) == {"people": ["Marco", "Ana"], "topics": ["release"]}

    third = runner.invoke(cli.app, ["render", "-e", str(entities)], input=payload)
    assert "- People: [[Marco]], [[Ana]]" in third.stdout


def test_render_verbose_prints_counts_but_no_names(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["render", "-v"], input=json.dumps(VALID_EXTRACTION))

    assert result.exit_code == 0, result.output
    assert "new:      2 people, 1 topics" in result.stderr
    assert "Ana" not in result.stderr
    assert "Marco" not in result.stderr


def test_render_template_flag_overrides_config(tmp_path: Path) -> None:
    (tmp_path / "config.md.j2").write_text("config {{ title }}")
    (tmp_path / "flag.md.j2").write_text("flag {{ title }}")
    (tmp_path / "voxmd.yaml").write_text(f"render:\n  template: {tmp_path / 'config.md.j2'}\n")
    payload = json.dumps(VALID_EXTRACTION)

    from_config = runner.invoke(cli.app, ["render"], input=payload)
    from_flag = runner.invoke(
        cli.app, ["render", "-t", str(tmp_path / "flag.md.j2")], input=payload
    )

    assert from_config.stdout == "config Ship date\n"
    assert from_flag.stdout == "flag Ship date\n"


def test_render_rejects_a_bad_date_before_reading_input() -> None:
    from voxmd.errors import InputError

    result = runner.invoke(cli.app, ["render", "--date", "soon"], input="not json")

    assert isinstance(result.exception, InputError)
    assert "ISO 8601" in str(result.exception)


# --- process ----------------------------------------------------------------


@pytest.fixture
def recording(
    tmp_path: Path,
    fake_run: FakeRunner,
    fake_tools: None,
    model_file: Path,
    ollama_client: FakeClient,
) -> Path:
    (tmp_path / "vault").mkdir()
    (tmp_path / "voxmd.yaml").write_text(
        f"whisper:\n  model: {model_file}\nvault:\n  path: {tmp_path / 'vault'}\n"
    )
    fake_run.set("ffprobe", stdout=probe_json())
    fake_run.set("whisper-cli", stdout="Marco and Ana agreed to ship on Friday.")
    audio = tmp_path / "inbox.m4a"
    audio.write_bytes(b"\x02" * 4096)
    return audio


def test_process_prints_only_the_note_path(recording: Path, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["process", str(recording), "--date", "2026-09-15T14:03"])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{tmp_path / 'vault' / '2026-09-15 Ship date.md'}\n"
    assert result.stderr == ""


def test_process_leaves_the_same_unchanged_file_alone(recording: Path) -> None:
    first = runner.invoke(cli.app, ["process", str(recording)])
    second = runner.invoke(cli.app, ["process", str(recording)])

    assert second.exit_code == 0, second.output
    assert second.stdout == first.stdout
    assert "already processed" in second.stderr


def test_process_verbose_shows_stages_but_no_title_or_names(recording: Path) -> None:
    result = runner.invoke(cli.app, ["process", str(recording), "-v"])

    assert result.exit_code == 0, result.output
    assert "transcribing..." in result.stderr
    assert "whisper:" in result.stderr
    assert "Ship date" not in result.stderr
    assert "Marco" not in result.stderr


def test_process_vault_flag_overrides_config(recording: Path, tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()

    result = runner.invoke(cli.app, ["process", str(recording), "--vault", "other"])

    assert result.exit_code == 0, result.output
    assert Path(result.stdout.strip()).parent == tmp_path / "other"


def test_process_prints_the_note_before_reporting_a_partial_failure(
    recording: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from voxmd import safe
    from voxmd.errors import PartialFailure

    with (tmp_path / "voxmd.yaml").open("a") as config:
        config.write(f"archive:\n  dir: {tmp_path / 'archive'}\n")

    def denied(*args: object, **kwargs: object) -> Path:
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(safe, "move_no_replace", denied)

    result = runner.invoke(cli.app, ["process", str(recording)])

    assert isinstance(result.exception, PartialFailure)
    assert result.exception.exit_code == 8
    assert Path(result.stdout.strip()).is_file()
    assert "not archived" in result.stderr
    assert recording.exists()


def test_process_without_a_vault_fails_before_transcribing(
    recording: Path, tmp_path: Path, fake_run: FakeRunner, model_file: Path
) -> None:
    from voxmd.errors import ConfigError

    (tmp_path / "voxmd.yaml").write_text(f"whisper:\n  model: {model_file}\n")

    result = runner.invoke(cli.app, ["process", str(recording)])

    assert isinstance(result.exception, ConfigError)
    assert fake_run.calls == []


# --- doctor -----------------------------------------------------------------


def test_doctor_lists_every_check_and_fails_when_something_is_missing(
    fake_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxmd.errors import DependencyError

    class Unreachable:
        def list(self) -> None:
            raise ConnectionError("refused")

        def close(self) -> None:
            pass

    monkeypatch.setattr(extract_module, "make_client", lambda *args, **kwargs: Unreachable())

    result = runner.invoke(cli.app, ["doctor"])

    assert isinstance(result.exception, DependencyError)
    assert "whisper model" in result.stdout
    assert "FAIL" in result.stdout
    assert "ollama pull" not in result.stdout


def test_process_model_flag_overrides_config(
    recording: Path, ollama_client: FakeClient, tmp_path: Path
) -> None:
    (tmp_path / "voxmd.yaml").write_text(
        (tmp_path / "voxmd.yaml").read_text() + "ollama:\n  model: qwen3:8b\n"
    )

    result = runner.invoke(cli.app, ["process", str(recording), "-m", "gemma3:4b"])

    assert result.exit_code == 0, result.output
    assert ollama_client.chats[0]["model"] == "gemma3:4b"


# --- models -----------------------------------------------------------------


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    client = FakeOllama(installed=["qwen3:8b", "gemma3:4b"])
    monkeypatch.setattr(extract_module, "make_client", lambda *args, **kwargs: client)
    return client


def test_models_lists_what_is_pulled_and_marks_the_configured_one(catalog: FakeOllama) -> None:
    result = runner.invoke(cli.app, ["models"])

    assert result.exit_code == 0, result.output
    assert "* qwen3:8b" in result.stdout
    assert "  gemma3:4b" in result.stdout
    assert "4.9 GB" in result.stdout
    assert "8.2B Q4_K_M" in result.stdout
    assert "voxmd uses this one" in result.stdout


def test_models_warns_when_the_configured_model_is_not_pulled(
    catalog: FakeOllama, tmp_path: Path
) -> None:
    (tmp_path / "voxmd.yaml").write_text("ollama:\n  model: not-pulled:1b\n")

    result = runner.invoke(cli.app, ["models"])

    assert result.exit_code == 0, result.output
    assert "gemma3:4b" in result.stdout
    assert "ollama pull not-pulled:1b" in result.stderr


def test_models_with_nothing_pulled_says_what_to_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extract_module, "make_client", lambda *args, **kwargs: FakeOllama(installed=[])
    )

    result = runner.invoke(cli.app, ["models"])

    assert result.exit_code == 0, result.output
    assert "No models pulled yet" in result.stdout
    assert "ollama pull qwen3:8b" in result.stdout


def test_models_reports_an_unreachable_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    from voxmd.errors import DependencyError

    monkeypatch.setattr(
        extract_module,
        "make_client",
        lambda *args, **kwargs: FakeOllama(error=ConnectionError("refused")),
    )

    result = runner.invoke(cli.app, ["models"])

    assert isinstance(result.exception, DependencyError)


# --- watch ------------------------------------------------------------------


@pytest.fixture
def watch_setup(tmp_path: Path) -> Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (tmp_path / "vault").mkdir()
    (tmp_path / "voxmd.yaml").write_text(
        f"vault:\n  path: {tmp_path / 'vault'}\n"
        f"watch:\n  dir: {inbox}\n"
        f"state:\n  dir: {tmp_path / 'state'}\n"
    )
    return inbox


def test_watch_runs_the_watcher_and_logs_where_it_says_it_will(
    watch_setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxmd import watcher as watcher_module

    seen: dict[str, object] = {}

    def fake_watch(settings, *, log, **kwargs):  # type: ignore[no-untyped-def]
        seen["watch_dir"] = settings.watch.dir
        seen["log"] = log.path
        log.event("watch.start", pid=1)
        return None

    monkeypatch.setattr(watcher_module, "watch", fake_watch)

    result = runner.invoke(cli.app, ["watch"])

    assert result.exit_code == 0, result.output
    assert seen["watch_dir"] == watch_setup
    assert seen["log"] == tmp_path / "state" / "voxmd.log"
    assert "watch.start" in (tmp_path / "state" / "voxmd.log").read_text()
    # vision.md: "Log to a file, not stdout." The echo goes to stderr so the
    # terminal shows the work, and stdout stays empty.
    assert result.stdout == ""
    assert "watch.start" in result.stderr


def test_watch_takes_the_folder_from_the_flag(
    watch_setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxmd import watcher as watcher_module

    other = tmp_path / "other inbox"
    other.mkdir()
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        watcher_module,
        "watch",
        lambda settings, **kwargs: seen.update(dir=settings.watch.dir, model=settings.ollama.model),
    )

    result = runner.invoke(cli.app, ["watch", "--dir", str(other), "-m", "gemma3:4b"])

    assert result.exit_code == 0, result.output
    assert seen == {"dir": other, "model": "gemma3:4b"}


def test_watch_without_a_folder_says_what_to_set(tmp_path: Path) -> None:
    (tmp_path / "vault").mkdir()
    (tmp_path / "voxmd.yaml").write_text(
        f"vault:\n  path: {tmp_path / 'vault'}\nstate:\n  dir: {tmp_path / 'state'}\n"
    )

    result = runner.invoke(cli.app, ["watch"])

    assert isinstance(result.exception, errors.ConfigError)
    assert "watch.dir" in str(result.exception)


# --- status -----------------------------------------------------------------


def test_status_with_nothing_running(watch_setup: Path) -> None:
    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0, result.output
    assert "not running" in result.stdout
    assert "no watcher has run here" in result.stdout
    assert "0 recording(s) processed" in result.stdout


def test_status_reports_a_live_watcher(
    watch_setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from datetime import datetime

    from voxmd import status as status_module
    from voxmd.status import WatchState, state_path, write_state

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    write_state(
        state_path(state_dir),
        WatchState(
            pid=os.getpid(),
            started_at=datetime(2026, 9, 17, 9, 30).astimezone(),
            watch_dir=str(watch_setup),
            log=str(state_dir / "voxmd.log"),
            observer="FSEventsObserver",
            last_wake=datetime(2026, 9, 17, 10, 15).astimezone(),
            last_wake_trigger="created",
            wakes=4,
            processed=3,
        ),
    )
    monkeypatch.setattr(status_module, "pid_is_voxmd", lambda _pid: True)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0, result.output
    assert f"running (pid {os.getpid()})" in result.stdout
    assert "since 2026-09-17 09:30" in result.stdout
    assert str(watch_setup) in result.stdout
    assert "FSEventsObserver (filesystem events, not polling)" in result.stdout
    assert "2026-09-17 10:15:00 (created)" in result.stdout
    assert "3 processed, 0 failed, 4 wakes" in result.stdout


def test_status_creates_nothing(watch_setup: Path, tmp_path: Path) -> None:
    before = sorted(tmp_path.rglob("*"))

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0, result.output
    assert sorted(tmp_path.rglob("*")) == before
