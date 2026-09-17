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


def test_ledger_limit_is_in_megabytes() -> None:
    assert LimitsConfig(max_ledger_mb=2).max_ledger_bytes == 2 * 1024 * 1024


def test_process_settings_default_to_no_vault_no_archive_and_a_private_state_dir(
    tmp_path: Path,
) -> None:
    config = Config()

    assert config.vault.path is None
    assert config.vault.folder is None
    assert config.archive.dir is None
    assert config.state.dir == tmp_path / "home" / ".local" / "state" / "voxmd"


def test_vault_archive_and_state_paths_expand_home(tmp_path: Path) -> None:
    config = load_config(
        write(
            tmp_path / "c.yaml",
            "vault:\n  path: ~/Vault\n  folder: Voice memos/2026\n"
            "archive:\n  dir: ~/archive\nstate:\n  dir: ~/state\n",
        )
    )

    home = tmp_path / "home"
    assert config.vault.path == home / "Vault"
    assert config.vault.folder == Path("Voice memos/2026")
    assert config.archive.dir == home / "archive"
    assert config.state.dir == home / "state"


@pytest.mark.parametrize(
    "setting",
    ["vault:\n  path: relative/vault\n", "archive:\n  dir: archive\n", "state:\n  dir: ./state\n"],
)
def test_write_destinations_must_be_absolute(tmp_path: Path, setting: str) -> None:
    with pytest.raises(ConfigError, match="absolute"):
        load_config(write(tmp_path / "c.yaml", setting))


@pytest.mark.parametrize("folder", ["../outside", "/etc", "~/elsewhere", "a/../../b"])
def test_vault_folder_must_stay_inside_the_vault(tmp_path: Path, folder: str) -> None:
    with pytest.raises(ConfigError, match="inside the vault"):
        load_config(write(tmp_path / "c.yaml", f"vault:\n  folder: '{folder}'\n"))


@pytest.mark.parametrize("folder", ["''", "."])
def test_an_empty_vault_folder_means_the_vault_root(tmp_path: Path, folder: str) -> None:
    config = load_config(write(tmp_path / "c.yaml", f"vault:\n  folder: {folder}\n"))
    assert config.vault.folder is None


class TestWatchAndLog:
    def test_the_defaults_leave_watching_switched_off(self) -> None:
        settings = Config()
        assert settings.watch.dir is None
        assert settings.log.file is None
        assert (settings.watch.stable_seconds, settings.watch.poll_seconds) == (3.0, 0.5)

    def test_a_relative_watch_folder_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="watch.dir must be an absolute path"):
            Config.model_validate({"watch": {"dir": "inbox"}})

    def test_a_relative_log_file_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="log.file must be an absolute path"):
            Config.model_validate({"log": {"file": "voxmd.log"}})

    def test_a_home_relative_watch_folder_is_expanded(self) -> None:
        settings = Config.model_validate({"watch": {"dir": "~/inbox"}})
        assert settings.watch.dir is not None and settings.watch.dir.is_absolute()

    @pytest.mark.parametrize(
        ("section", "values"),
        [
            ("watch", {"stable_seconds": 0.1}),
            ("watch", {"poll_seconds": 0}),
            ("watch", {"settle_timeout_s": 0}),
            ("watch", {"lock_timeout_s": -1}),
            ("log", {"max_mb": 0}),
        ],
    )
    def test_out_of_range_values_are_refused(self, section: str, values: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            Config.model_validate({section: values})

    def test_a_typo_in_the_watch_section_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="stable_second"):
            Config.model_validate({"watch": {"stable_second": 3}})


class TestCommentedOutSections:
    """A section whose every key is commented out reads as null, not as absent."""

    def test_a_null_section_falls_back_to_its_defaults(self, tmp_path: Path) -> None:
        config = tmp_path / "voxmd.yaml"
        config.write_text("archive:\n  # dir: ~/somewhere\nwatch:\n  # dir: ~/inbox\n")

        settings = load_config(config)

        assert settings.archive.dir is None
        assert settings.watch.stable_seconds == 3.0

    def test_the_shipped_example_config_is_valid(self) -> None:
        example = Path(__file__).resolve().parents[1] / "voxmd.example.yaml"
        assert load_config(example).vault.path is not None

    def test_a_null_value_for_an_unknown_key_is_still_an_error(self, tmp_path: Path) -> None:
        config = tmp_path / "voxmd.yaml"
        config.write_text("archve:\n")
        with pytest.raises(ConfigError, match="archve"):
            load_config(config)


class TestTranscriptSettings:
    def test_transcripts_are_on_by_default_in_their_own_folder(self) -> None:
        assert Config().vault.transcripts is True
        assert Config().vault.transcripts_folder == Path("Transcripts")

    @pytest.mark.parametrize("folder", ["/abs", "~/x", "../out", "a/../../b", "", "."])
    def test_the_transcripts_folder_must_stay_inside_the_notes_folder(self, folder: str) -> None:
        with pytest.raises(ValidationError, match="transcripts_folder"):
            Config.model_validate({"vault": {"transcripts_folder": folder}})

    def test_a_nested_transcripts_folder_is_fine(self) -> None:
        settings = Config.model_validate({"vault": {"transcripts_folder": "Raw/Text"}})
        assert settings.vault.transcripts_folder == Path("Raw/Text")


class TestDuplicates:
    def test_repeats_are_copied_by_default(self) -> None:
        assert Config().vault.duplicates == "copy"

    def test_skip_can_be_chosen(self) -> None:
        assert Config.model_validate({"vault": {"duplicates": "skip"}}).vault.duplicates == "skip"

    def test_anything_else_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            Config.model_validate({"vault": {"duplicates": "overwrite"}})
