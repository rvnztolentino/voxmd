"""voxmd doctor: read-only prerequisite checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ollama
import pytest

from voxmd import safe
from voxmd.config import CONFIG_ENV_VAR
from voxmd.doctor import FAIL, OK, WARN, Check, model_installed, run_checks
from voxmd.errors import DependencyError


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@dataclass
class FakeOllama:
    installed: list[str] = field(default_factory=lambda: ["qwen3:8b"])
    loaded: list[str] = field(default_factory=list)
    error: BaseException | None = None

    def list(self) -> ollama.ListResponse:
        if self.error is not None:
            raise self.error
        return ollama.ListResponse(
            models=[ollama.ListResponse.Model(model=n) for n in self.installed]
        )

    def ps(self) -> ollama.ProcessResponse:
        return ollama.ProcessResponse(
            models=[ollama.ProcessResponse.Model(model=n) for n in self.loaded]
        )

    def close(self) -> None:
        pass


@pytest.fixture
def configured(tmp_path: Path, fake_tools: None, model_file: Path) -> Path:
    (tmp_path / "vault").mkdir()
    config = tmp_path / "voxmd.yaml"
    config.write_text(
        f"whisper:\n  model: {model_file}\n"
        f"vault:\n  path: {tmp_path / 'vault'}\n  folder: Memos\n"
        f"archive:\n  dir: {tmp_path / 'archive'}\n"
        f"state:\n  dir: {tmp_path / 'state'}\n"
        f"entities:\n  file: {tmp_path / 'entities.json'}\n"
    )
    return config


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


def tree(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*"))


def test_a_complete_setup_passes_and_doctor_creates_nothing(
    configured: Path, tmp_path: Path
) -> None:
    before = tree(tmp_path)

    checks = run_checks(configured, client=FakeOllama())

    assert [check.name for check in checks if check.status != OK] == []
    assert [check.name for check in checks] == [
        "config",
        "ffprobe",
        "ffmpeg",
        "whisper-cli",
        "whisper model",
        "ollama",
        "ollama model",
        "ollama memory",
        "vault",
        "archive",
        "state",
        "template",
        "entities",
    ]
    assert tree(tmp_path) == before


def test_missing_pieces_fail_with_a_way_to_fix_them(tmp_path: Path, fake_tools: None) -> None:
    config = tmp_path / "voxmd.yaml"
    config.write_text(f"vault:\n  path: {tmp_path / 'nope'}\n")

    checks = by_name(run_checks(config, client=FakeOllama(installed=[])))

    assert checks["whisper model"].status == FAIL
    assert "No whisper model configured" in checks["whisper model"].detail
    assert checks["ollama model"].status == FAIL
    assert "ollama pull qwen3:8b" in checks["ollama model"].detail
    assert checks["vault"].status == FAIL
    assert "Vault not found" in checks["vault"].detail
    assert checks["archive"].status == OK


def test_missing_tools_are_reported(configured: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, *, hint: str) -> Path:
        raise DependencyError(f"{name!r} was not found on PATH.\n{hint}")

    monkeypatch.setattr(safe, "resolve_tool", missing)

    checks = by_name(run_checks(configured, client=FakeOllama()))

    for tool in ("ffprobe", "ffmpeg", "whisper-cli"):
        assert checks[tool].status == FAIL
    assert "brew install whisper-cpp" in checks["whisper-cli"].detail


def test_an_unreachable_ollama_is_one_failure_with_a_hint(configured: Path) -> None:
    checks = by_name(run_checks(configured, client=FakeOllama(error=ConnectionError("refused"))))

    assert checks["ollama"].status == FAIL
    assert "ollama serve" in checks["ollama"].detail
    assert "ollama model" not in checks


def test_a_model_loaded_by_something_else_is_a_warning(configured: Path) -> None:
    checks = by_name(run_checks(configured, client=FakeOllama(loaded=["gemma3:4b"])))

    assert checks["ollama memory"].status == WARN


def test_an_invalid_config_is_the_only_check(tmp_path: Path) -> None:
    config = tmp_path / "voxmd.yaml"
    config.write_text("ollama:\n  host: http://10.0.0.5:11434\n")

    checks = run_checks(config, client=FakeOllama())

    assert len(checks) == 1
    assert checks[0].status == FAIL
    assert "loopback" in checks[0].detail


def test_a_corrupt_entities_file_fails_and_is_left_alone(configured: Path, tmp_path: Path) -> None:
    entities = tmp_path / "entities.json"
    entities.write_text('{"people": [')

    checks = by_name(run_checks(configured, client=FakeOllama()))

    assert checks["entities"].status == FAIL
    assert entities.read_text() == '{"people": ['


def test_existing_state_reports_how_many_recordings_were_processed(
    configured: Path, tmp_path: Path
) -> None:
    (tmp_path / "state").mkdir()

    checks = by_name(run_checks(configured, client=FakeOllama()))

    assert checks["state"].status == OK
    assert "0 recordings processed" in checks["state"].detail


@pytest.mark.parametrize(
    ("wanted", "installed", "expected"),
    [
        ("qwen3:8b", {"qwen3:8b"}, True),
        ("llama3.2", {"llama3.2:latest"}, True),
        ("qwen3", {"qwen3:8b"}, False),
        ("qwen3:8b", {"qwen3:14b"}, False),
        ("hf.co/user/repo", {"hf.co/user/repo:latest"}, True),
    ],
)
def test_model_names_match_like_ollama_does(
    wanted: str, installed: set[str], expected: bool
) -> None:
    assert model_installed(wanted, installed) is expected
