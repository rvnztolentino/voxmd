"""The CLI surface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FakeRunner, probe_json
from voxmd import cli
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
        "heavy = {'voxmd.config', 'voxmd.transcribe', 'pydantic', 'yaml'}; "
        "loaded = sorted(heavy & set(sys.modules)); "
        "print(','.join(loaded))"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, test only
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=30
    )
    assert result.stdout.strip() == "", f"imported at startup: {result.stdout.strip()}"
