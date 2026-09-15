"""The CLI surface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import VALID_EXTRACTION, FakeClient, FakeRunner, chat_reply, probe_json
from voxmd import cli
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
        "'voxmd.render', 'voxmd.entities', "
        "'pydantic', 'yaml', 'ollama', 'httpx', 'jinja2', 'rapidfuzz'}; "
        "loaded = sorted(heavy & set(sys.modules)); "
        "print(','.join(loaded))"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, test only
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=30
    )
    assert result.stdout.strip() == "", f"imported at startup: {result.stdout.strip()}"


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
