# voxmd

Voice memo → structured markdown notes. Fully local. CLI only.

Drop an audio file in, get a linked note in an Obsidian vault. No cloud, no API keys, no cost.

> **Status:** stage 1 of 6. `voxmd transcribe` works; extraction, rendering, the vault, and watch mode are not built yet.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- ffmpeg (`brew install ffmpeg`)
- whisper.cpp (`brew install whisper-cpp`) plus a ggml model file

The full setup walkthrough, including the model download, lives in the project's setup guide.

## Install

```sh
uv sync
uv run voxmd --help
```

## Usage

### Transcribe

```sh
voxmd transcribe memo.m4a --model ~/.local/share/voxmd/models/ggml-large-v3-turbo.bin
```

The transcript goes to stdout and everything else to stderr, so it pipes cleanly:

```sh
voxmd transcribe memo.m4a > memo.txt
```

Add `-v` to see format, timing, and realtime factor on stderr.

A config file is optional for this command. See [`voxmd.example.yaml`](voxmd.example.yaml); voxmd looks in `$VOXMD_CONFIG`, then `./voxmd.yaml`, then `~/.config/voxmd/config.yaml`. CLI flags override config.

## What it does and doesn't do

These are commitments, not aspirations.

**It never auto-starts.** voxmd installs no LaunchAgent, plist, cron entry, or login item. Nothing runs unless you start it, and nothing keeps running after you close the terminal.

**It makes no network calls.** At runtime the only connection voxmd will make is to Ollama on localhost (once extraction exists). No telemetry, no update checks. None of its Python dependencies phone home. Two honest caveats:

- Setup involves downloads you do yourself: the whisper weights from Hugging Face and the Ollama model. voxmd never downloads anything.
- Ollama is a separate program. It may check for its own updates. That's Ollama's behaviour, not voxmd's.

**It never holds two models in memory.** whisper runs as a subprocess and exits before Ollama is asked to load anything, and Ollama is told to unload as soon as it answers. This costs a few seconds of model load per memo in exchange for voxmd occupying no memory between memos.

**Your data stays on disk, unencrypted.** Transcripts and notes are plaintext files. voxmd's logs record filenames, timings, and outcomes, never what was said.

## Resource usage

Idle CPU for watch mode will be measured and recorded here once watch mode exists. Not an estimate; a measurement.

## Development

```sh
uv run pytest
uv run ruff check
```

The test suite needs no ffmpeg, whisper, model weights, or network: external tools are faked at the single point where voxmd spawns them.
