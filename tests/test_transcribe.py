"""Stage 1: transcribe."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import WHISPER_READY, FakeRunner, probe_json
from voxmd.config import LimitsConfig, WhisperConfig
from voxmd.errors import AudioError, DependencyError, ToolFailure
from voxmd.transcribe import (
    AudioProbe,
    _whisper_timeout,
    clean_transcript,
    transcribe,
)


@pytest.fixture
def whisper(model_file: Path) -> WhisperConfig:
    return WhisperConfig(model=model_file, threads=4)


@pytest.fixture
def limits() -> LimitsConfig:
    return LimitsConfig()


@pytest.fixture
def normal_run(fake_run: FakeRunner, fake_tools: None) -> FakeRunner:
    """A memo that needs converting, then transcribes cleanly."""
    fake_run.set("ffprobe", stdout=probe_json())
    fake_run.set("whisper-cli", stdout=" Pick up milk.\n And call Marco about the launch.\n")
    return fake_run


class TestHappyPath:
    def test_returns_cleaned_text(
        self, normal_run: FakeRunner, memo: Path, whisper, limits
    ) -> None:
        result = transcribe(memo, whisper=whisper, limits=limits)
        assert result.text == "Pick up milk. And call Marco about the launch."
        assert result.source == memo.resolve()
        assert result.converted is True

    def test_stages_run_in_order(self, normal_run: FakeRunner, memo: Path, whisper, limits) -> None:
        transcribe(memo, whisper=whisper, limits=limits)
        assert [call.tool for call in normal_run.calls] == ["ffprobe", "ffmpeg", "whisper-cli"]

    def test_ffmpeg_normalizes_to_16k_mono_pcm(
        self, normal_run: FakeRunner, memo: Path, whisper, limits
    ) -> None:
        transcribe(memo, whisper=whisper, limits=limits)
        (ffmpeg,) = normal_run.called("ffmpeg")
        assert ffmpeg.flag("-ar") == "16000"
        assert ffmpeg.flag("-ac") == "1"
        assert ffmpeg.flag("-c:a") == "pcm_s16le"
        assert "-nostdin" in ffmpeg.argv

    def test_whisper_receives_config(
        self, normal_run: FakeRunner, memo: Path, model_file, limits
    ) -> None:
        config = WhisperConfig(model=model_file, threads=3, language="en")
        transcribe(memo, whisper=config, limits=limits)
        (call,) = normal_run.called("whisper-cli")
        assert call.flag("-m") == str(model_file)
        assert call.flag("-t") == "3"
        assert call.flag("-l") == "en"
        assert "-nt" in call.argv and "-np" in call.argv

    def test_whisper_reads_the_converted_file_not_the_source(
        self, normal_run: FakeRunner, memo: Path, whisper, limits
    ) -> None:
        transcribe(memo, whisper=whisper, limits=limits)
        (ffmpeg,) = normal_run.called("ffmpeg")
        (call,) = normal_run.called("whisper-cli")
        assert call.flag("-f") == ffmpeg.argv[-1]
        assert call.flag("-f") != str(memo)


class TestSkipsWork:
    def test_already_whisper_ready_wav_skips_ffmpeg(
        self, fake_run: FakeRunner, fake_tools: None, tmp_path: Path, whisper, limits
    ) -> None:
        wav = tmp_path / "ready.wav"
        wav.write_bytes(b"\x00" * 64)
        fake_run.set("ffprobe", stdout=probe_json(**WHISPER_READY))
        fake_run.set("whisper-cli", stdout="hello")

        result = transcribe(wav, whisper=whisper, limits=limits)

        assert not fake_run.ran("ffmpeg")
        assert result.converted is False
        (call,) = fake_run.called("whisper-cli")
        assert call.flag("-f") == str(wav.resolve())

    @pytest.mark.parametrize(
        "near_miss",
        [
            {"sample_rate": 44_100, "channels": 1, "codec": "pcm_s16le"},
            {"sample_rate": 16_000, "channels": 2, "codec": "pcm_s16le"},
            {"sample_rate": 16_000, "channels": 1, "codec": "pcm_f32le"},
        ],
    )
    def test_near_miss_formats_still_convert(self, near_miss: dict) -> None:
        probe = AudioProbe(
            duration_s=1.0,
            **{
                "sample_rate": near_miss["sample_rate"],
                "channels": near_miss["channels"],
                "codec": near_miss["codec"],
            },
        )
        assert probe.is_whisper_ready is False


class TestFailsBeforeExpensiveWork:
    """Anything that can be rejected cheaply must be, before whisper starts."""

    def test_no_audio_stream(
        self, fake_run: FakeRunner, fake_tools: None, memo: Path, whisper, limits
    ) -> None:
        fake_run.set("ffprobe", stdout=probe_json(streams=False))
        with pytest.raises(AudioError, match="no audio stream"):
            transcribe(memo, whisper=whisper, limits=limits)
        assert not fake_run.ran("ffmpeg")
        assert not fake_run.ran("whisper-cli")

    def test_ffprobe_cannot_read_file(
        self, fake_run: FakeRunner, fake_tools: None, memo: Path, whisper, limits
    ) -> None:
        fake_run.set("ffprobe", returncode=1, stderr="Invalid data found when processing input")
        with pytest.raises(AudioError, match="Invalid data found"):
            transcribe(memo, whisper=whisper, limits=limits)
        assert not fake_run.ran("whisper-cli")

    def test_over_duration_limit(
        self, fake_run: FakeRunner, fake_tools: None, memo: Path, whisper
    ) -> None:
        fake_run.set("ffprobe", stdout=probe_json(duration=str(61 * 60)))
        with pytest.raises(AudioError, match="max_duration_min"):
            transcribe(memo, whisper=whisper, limits=LimitsConfig(max_duration_min=60))
        assert not fake_run.ran("ffmpeg")
        assert not fake_run.ran("whisper-cli")

    def test_over_size_limit_never_spawns_anything(
        self, fake_run: FakeRunner, fake_tools: None, tmp_path: Path, whisper
    ) -> None:
        big = tmp_path / "big.m4a"
        big.write_bytes(b"\x00" * (1024 * 1024 + 1))
        with pytest.raises(AudioError, match="max_audio_mb"):
            transcribe(big, whisper=whisper, limits=LimitsConfig(max_audio_mb=1))
        assert fake_run.calls == []

    def test_no_model_configured(
        self, fake_run: FakeRunner, fake_tools: None, memo: Path, limits
    ) -> None:
        fake_run.set("ffprobe", stdout=probe_json())
        with pytest.raises(DependencyError, match="--model"):
            transcribe(memo, whisper=WhisperConfig(), limits=limits)
        assert not fake_run.ran("ffmpeg")

    def test_model_file_missing(
        self, fake_run: FakeRunner, fake_tools: None, memo: Path, tmp_path: Path, limits
    ) -> None:
        fake_run.set("ffprobe", stdout=probe_json())
        with pytest.raises(DependencyError, match="not found"):
            transcribe(memo, whisper=WhisperConfig(model=tmp_path / "gone.bin"), limits=limits)
        assert not fake_run.ran("ffmpeg")


class TestToolFailures:
    def test_ffmpeg_failure(self, normal_run: FakeRunner, memo: Path, whisper, limits) -> None:
        normal_run.set("ffmpeg", returncode=1, stderr="Conversion failed!")
        with pytest.raises(ToolFailure, match="Conversion failed"):
            transcribe(memo, whisper=whisper, limits=limits)
        assert not normal_run.ran("whisper-cli")

    def test_whisper_failure(self, normal_run: FakeRunner, memo: Path, whisper, limits) -> None:
        normal_run.set("whisper-cli", returncode=1, stderr="failed to load model")
        with pytest.raises(ToolFailure, match="failed to load model"):
            transcribe(memo, whisper=whisper, limits=limits)

    def test_silence_is_reported_not_returned_as_empty_text(
        self, normal_run: FakeRunner, memo: Path, whisper, limits
    ) -> None:
        normal_run.set("whisper-cli", stdout="[BLANK_AUDIO]\n\n")
        with pytest.raises(ToolFailure, match="no text"):
            transcribe(memo, whisper=whisper, limits=limits)


class TestSecurity:
    def test_hostile_filename_stays_one_argument(
        self, normal_run: FakeRunner, tmp_path: Path, whisper, limits
    ) -> None:
        hostile = tmp_path / "; rm -rf ~ $(whoami) `id`.m4a"
        hostile.write_bytes(b"\x00" * 64)

        transcribe(hostile, whisper=whisper, limits=limits)

        (probe,) = normal_run.called("ffprobe")
        (ffmpeg,) = normal_run.called("ffmpeg")
        assert str(hostile.resolve()) in probe.argv
        assert ffmpeg.flag("-i") == str(hostile.resolve())

    def test_intermediate_wav_is_never_written_beside_the_source(
        self, normal_run: FakeRunner, memo: Path, whisper, limits
    ) -> None:
        # A new audio file in a watched folder would re-trigger the watcher.
        transcribe(memo, whisper=whisper, limits=limits)
        (ffmpeg,) = normal_run.called("ffmpeg")
        destination = Path(ffmpeg.argv[-1])
        assert destination.parent != memo.parent
        assert "voxmd-" in destination.parent.name

    def test_intermediate_wav_is_cleaned_up(
        self, normal_run: FakeRunner, memo: Path, whisper, limits
    ) -> None:
        transcribe(memo, whisper=whisper, limits=limits)
        (ffmpeg,) = normal_run.called("ffmpeg")
        assert not Path(ffmpeg.argv[-1]).parent.exists()

    def test_every_subprocess_has_a_timeout(
        self, normal_run: FakeRunner, memo: Path, whisper, limits
    ) -> None:
        transcribe(memo, whisper=whisper, limits=limits)
        assert all(call.timeout > 0 for call in normal_run.calls)


class TestCleanTranscript:
    def test_joins_segments_into_one_paragraph(self) -> None:
        assert clean_transcript(" First.\n  Second.  \n\nThird.\n") == "First. Second. Third."

    def test_strips_timestamps_if_the_flag_was_ignored(self) -> None:
        raw = "[00:00:00.000 --> 00:00:02.500]  Hello there.\n[00:00:02.500 --> 00:00:04.000]  Bye."
        assert clean_transcript(raw) == "Hello there. Bye."

    def test_drops_non_speech_markers(self) -> None:
        assert clean_transcript("[BLANK_AUDIO]\nReal words.\n[MUSIC]\n") == "Real words."

    def test_keeps_speech_that_contains_brackets(self) -> None:
        assert clean_transcript("Use the [draft] version.") == "Use the [draft] version."

    def test_strips_ansi_colour_codes(self) -> None:
        assert clean_transcript("\x1b[38;5;160mHello\x1b[0m world") == "Hello world"

    def test_empty(self) -> None:
        assert clean_transcript("\n  \n") == ""


class TestWhisperTimeout:
    def test_scales_with_duration(self) -> None:
        limits = LimitsConfig(whisper_timeout_factor=3, whisper_timeout_floor_s=300)
        assert _whisper_timeout(3600, limits) == 10_800

    def test_short_clips_get_the_floor(self) -> None:
        limits = LimitsConfig(whisper_timeout_factor=3, whisper_timeout_floor_s=300)
        assert _whisper_timeout(20, limits) == 300

    def test_unknown_duration_assumes_the_ceiling_not_a_small_guess(self) -> None:
        limits = LimitsConfig(max_duration_min=180, whisper_timeout_factor=3)
        assert _whisper_timeout(None, limits) == 180 * 60 * 3
