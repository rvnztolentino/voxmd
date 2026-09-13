"""Shared test fixtures.

The whole suite runs without ffmpeg, without whisper.cpp, without model
weights, and without a network. External tools are faked at the one place
voxmd spawns them (``safe.run``), which keeps the tests fast and means they
pass on a machine where the setup guide hasn't been followed yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from voxmd import safe


@dataclass
class FakeProc:
    """Stand-in for subprocess.CompletedProcess."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class RecordedCall:
    argv: list[str]
    timeout: float
    what: str

    @property
    def tool(self) -> str:
        return Path(self.argv[0]).name

    def flag(self, name: str) -> str | None:
        """Value following ``name`` in the argv, or None if absent."""
        try:
            return self.argv[self.argv.index(name) + 1]
        except (ValueError, IndexError):
            return None


@dataclass
class FakeRunner:
    """Records every subprocess voxmd would have spawned, and scripts replies."""

    calls: list[RecordedCall] = field(default_factory=list)
    responses: dict[str, FakeProc] = field(default_factory=dict)

    def set(self, tool: str, **kwargs: object) -> None:
        self.responses[tool] = FakeProc(**kwargs)  # type: ignore[arg-type]

    def called(self, tool: str) -> list[RecordedCall]:
        return [call for call in self.calls if call.tool == tool]

    def ran(self, tool: str) -> bool:
        return bool(self.called(tool))

    def __call__(self, argv, *, timeout: float, what: str) -> FakeProc:
        recorded = RecordedCall([str(a) for a in argv], timeout, what)
        self.calls.append(recorded)
        response = self.responses.get(recorded.tool, FakeProc())
        if recorded.tool == "ffmpeg" and response.returncode == 0:
            # Real ffmpeg writes its output file; transcribe checks for it.
            Path(recorded.argv[-1]).write_bytes(b"RIFF fake wav")
        return response


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    """Replace safe.run so no real process is ever spawned."""
    runner = FakeRunner()
    monkeypatch.setattr(safe, "run", runner)
    return runner


@pytest.fixture
def fake_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend ffmpeg, ffprobe and whisper-cli are all installed."""
    monkeypatch.setattr(safe, "resolve_tool", lambda name, *, hint: Path("/usr/local/bin") / name)


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """A stand-in for the ggml weights, which are a 1.6 GB manual download."""
    path = tmp_path / "ggml-large-v3-turbo.bin"
    path.write_bytes(b"not really a model")
    return path


@pytest.fixture
def memo(tmp_path: Path) -> Path:
    """A plausible voice memo. Contents are irrelevant; ffprobe is faked."""
    path = tmp_path / "memo.m4a"
    path.write_bytes(b"\x00" * 2048)
    return path


def probe_json(
    *,
    sample_rate: int | str = 44_100,
    channels: int = 2,
    codec: str = "aac",
    duration: str | None = "125.0",
    streams: bool = True,
) -> str:
    """Build ffprobe JSON output matching what the real tool emits."""
    stream: dict[str, object] = {
        "sample_rate": str(sample_rate),
        "channels": channels,
        "codec_name": codec,
    }
    if duration is not None:
        stream["duration"] = duration
    payload = {
        "streams": [stream] if streams else [],
        "format": {"duration": duration} if duration is not None else {},
    }
    return json.dumps(payload)


WHISPER_READY = {"sample_rate": 16_000, "channels": 1, "codec": "pcm_s16le"}
