"""Config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voxmd.config import (
    CONFIG_ENV_VAR,
    Config,
    LimitsConfig,
    OllamaConfig,
    apply_overrides,
    find_config,
    load_config,
)
from voxmd.errors import ConfigError


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a real config on this machine leak into a test."""
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_no_config_file_means_defaults() -> None:
    config = load_config()
    assert config == Config()
    assert config.whisper.binary == "whisper-cli"
    assert config.whisper.model is None
    assert config.whisper.language == "auto"


def test_empty_file_means_defaults(tmp_path: Path) -> None:
    assert load_config(write(tmp_path / "c.yaml", "")) == Config()


def test_explicit_missing_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path / "c.yaml", "whisper: [unclosed"))


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path / "c.yaml", "- a\n- b\n"))


def test_typo_in_a_key_is_an_error_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="whisper.modle"):
        load_config(write(tmp_path / "c.yaml", "whisper:\n  modle: /x.bin\n"))


def test_python_object_tags_are_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    evil = f"whisper: !!python/object/apply:os.system ['touch {marker}']\n"
    with pytest.raises(ConfigError):
        load_config(write(tmp_path / "c.yaml", evil))
    assert not marker.exists()


def test_model_path_expands_home(tmp_path: Path) -> None:
    config = load_config(write(tmp_path / "c.yaml", "whisper:\n  model: ~/models/w.bin\n"))
    assert config.whisper.model == tmp_path / "home" / "models" / "w.bin"


@pytest.mark.parametrize("bad", ["", "   ", "en; rm -rf ~", "--help"])
def test_language_rejects_values_that_are_not_language_codes(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError, match="language"):
        load_config(write(tmp_path / "c.yaml", f"whisper:\n  language: {bad!r}\n"))


def test_language_is_normalized(tmp_path: Path) -> None:
    config = load_config(write(tmp_path / "c.yaml", "whisper:\n  language: ' EN '\n"))
    assert config.whisper.language == "en"


def test_limits_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="max_audio_mb"):
        load_config(write(tmp_path / "c.yaml", "limits:\n  max_audio_mb: 0\n"))


def test_env_var_takes_precedence_over_local_and_user_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path / "voxmd.yaml", "whisper:\n  language: fr\n")
    write(tmp_path / "home" / ".config" / "voxmd" / "config.yaml", "whisper:\n  language: de\n")
    from_env = write(tmp_path / "env.yaml", "whisper:\n  language: es\n")

    monkeypatch.setenv(CONFIG_ENV_VAR, str(from_env))
    assert find_config() == from_env
    assert load_config().whisper.language == "es"


def test_local_config_beats_user_config(tmp_path: Path) -> None:
    write(tmp_path / "home" / ".config" / "voxmd" / "config.yaml", "whisper:\n  language: de\n")
    write(tmp_path / "voxmd.yaml", "whisper:\n  language: fr\n")
    assert load_config().whisper.language == "fr"


def test_user_config_is_used_when_nothing_else_exists(tmp_path: Path) -> None:
    write(tmp_path / "home" / ".config" / "voxmd" / "config.yaml", "whisper:\n  language: de\n")
    assert load_config().whisper.language == "de"


# --- ollama -----------------------------------------------------------------


def test_ollama_defaults_are_local_and_unloading_is_not_configurable() -> None:
    config = load_config().ollama
    assert config.host == "http://127.0.0.1:11434"
    assert config.model == "qwen3:8b"
    # Constraints, not settings: never exposed to config.
    assert "keep_alive" not in OllamaConfig.model_fields
    assert "think" not in OllamaConfig.model_fields


@pytest.mark.parametrize(
    ("given", "normalized"),
    [
        ("127.0.0.1", "http://127.0.0.1:11434"),
        ("localhost:9999", "http://localhost:9999"),
        ("http://LOCALHOST", "http://localhost:11434"),
        ("https://127.0.0.2:8443/", "https://127.0.0.2:8443"),
        ("[::1]:11434", "http://[::1]:11434"),
        ("http://[::1]", "http://[::1]:11434"),
    ],
)
def test_loopback_hosts_are_accepted_and_normalized(given: str, normalized: str) -> None:
    assert OllamaConfig(host=given).host == normalized


@pytest.mark.parametrize(
    "host",
    [
        "192.168.1.20:11434",
        "10.0.0.1",
        "0.0.0.0",  # noqa: S104 - a value under test, nothing binds to it
        "http://example.com",
        "http://127.0.0.1.evil.example",
        "http://localhost@evil.example",
        "http://user:pw@127.0.0.1",
        "http://127.0.0.1:11434/api",
        "ftp://127.0.0.1",
        "http://127.0.0.1:notaport",
        "",
    ],
)
def test_non_loopback_or_malformed_hosts_are_refused(tmp_path: Path, host: str) -> None:
    with pytest.raises(ConfigError, match="host"):
        load_config(write(tmp_path / "c.yaml", f'ollama:\n  host: "{host}"\n'))


@pytest.mark.parametrize("name", ["qwen3:8b", "library/llama3.2:3b", "hf.co/user/repo:Q4_K_M"])
def test_ollama_model_names_are_accepted(name: str) -> None:
    assert OllamaConfig(model=name).model == name


@pytest.mark.parametrize("name", ["", "-rf", "qwen3 8b", "a;b"])
def test_malformed_ollama_model_names_are_refused(name: str) -> None:
    with pytest.raises(ValidationError):
        OllamaConfig(model=name)


def test_context_must_leave_room_for_the_transcript(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="num_ctx"):
        load_config(write(tmp_path / "c.yaml", "ollama:\n  num_ctx: 4096\n  num_predict: 4096\n"))


def test_overrides_are_validated_for_any_section() -> None:
    with pytest.raises(ConfigError, match="model"):
        apply_overrides(OllamaConfig(), model="--help")
    assert apply_overrides(OllamaConfig(), model=None).model == "qwen3:8b"
    assert apply_overrides(OllamaConfig(), model="gemma3:4b").model == "gemma3:4b"


def test_transcript_limit_is_in_kilobytes() -> None:
    assert LimitsConfig(max_transcript_kb=2).max_transcript_bytes == 2048
