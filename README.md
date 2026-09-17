# voxmd

[![PyPI](https://img.shields.io/pypi/v/voxmd?logo=pypi&logoColor=white)](https://pypi.org/project/voxmd/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/rvnztolentino/voxmd/blob/main/LICENSE)
[![Platform: macOS | Linux](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)](https://github.com/rvnztolentino/voxmd#setup)

Turn voice memos into linked Obsidian notes, entirely on your own machine. voxmd transcribes a recording, pulls out a summary, key points, decisions, actions, people and topics, and writes a note into your vault. Run it on one file, or let it watch the folder your phone syncs into.

## Setup

Works on macOS and Linux (not Windows). The commands use Homebrew; on Linux, install ffmpeg, whisper.cpp and Ollama with your package manager.

1. **Install voxmd and its tools.** voxmd installs with [uv](https://docs.astral.sh/uv/), which also fetches Python 3.13 if you don't have it.

   ```sh
   uv tool install voxmd
   brew install ffmpeg whisper-cpp
   ```

2. **Download the speech model** (1.6 GB).

   ```sh
   mkdir -p ~/.local/share/voxmd/models
   curl -L -o ~/.local/share/voxmd/models/ggml-large-v3-turbo.bin \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
   ```

3. **Pull a language model.** Start Ollama first (open the app, or run `ollama serve`).

   ```sh
   ollama pull qwen3:8b
   ```

4. **Create your config**, then set at least `whisper.model` and `vault.path` (and `watch.dir` to use the watcher).

   ```sh
   mkdir -p ~/.config/voxmd
   curl -L -o ~/.config/voxmd/config.yaml \
     https://raw.githubusercontent.com/rvnztolentino/voxmd/main/voxmd.example.yaml
   ```

5. **Check everything.** Every line should say `ok`; anything that fails tells you how to fix it. It changes nothing.

   ```sh
   voxmd doctor
   ```

## Usage

**One recording:**

```sh
voxmd process memo.m4a
```

Prints the path of the new note. Useful flags: `--model` (another Ollama model), `--vault PATH`, `--date`, `--no-archive` (leave the audio where it is), `--force`, and `-v` for timings.

**A folder, automatically:**

```sh
voxmd watch      # runs until you press Ctrl-C or close the terminal
voxmd status     # in another terminal: running?, last activity, notes made today
```

The watcher waits for each file to finish syncing, handles one memo at a time, picks up anything already in the folder when it starts, and logs everything it does to `~/.local/state/voxmd/voxmd.log`.

**Other commands:** `voxmd models` lists your Ollama models. `transcribe`, `extract` and `render` run one step at a time and pipe into each other:

```sh
voxmd transcribe memo.m4a | voxmd extract | voxmd render --source memo.m4a
```

## Configuration

One YAML file, read from `$VOXMD_CONFIG`, then `./voxmd.yaml`, then `~/.config/voxmd/config.yaml` (first found wins). [`voxmd.example.yaml`](https://github.com/rvnztolentino/voxmd/blob/main/voxmd.example.yaml) documents every setting.

```yaml
whisper:
  model: ~/.local/share/voxmd/models/ggml-large-v3-turbo.bin
  language: auto                    # or a code like en, tl

ollama:
  model: qwen3:8b                   # any model you've pulled

vault:
  path: ~/Documents/Obsidian/Main   # must already exist
  folder: Voice memos
  transcripts: true                 # save each transcript as a linked note
  duplicates: copy                  # same audio again: copy (Title 2.md) or skip

archive:
  dir: ~/Documents/Voice memos archive   # omit to leave recordings in place

watch:
  dir: ~/Documents/Voice memos inbox     # must exist; keep the archive outside it
```

Paths must be absolute or start with `~`, and unknown keys are rejected, so a typo shows up as an error.

## Choosing a model

Pick a language model by your machine's memory, then set `ollama.model` or pass `--model`:

- **8 GB:** `qwen3:4b` or `llama3.2:3b`. Fast, but less accurate.
- **16 GB:** `qwen3:8b` (default) or `qwen2.5:7b`.
- **32 GB+:** `qwen3:14b` or `gemma3:12b`. Better notes, slower to load.

Smaller models make more mistakes and are more easily misled by instructions spoken inside a memo. The model must support Ollama's structured output.

For transcription, `whisper.model` can be any ggml file from the same Hugging Face repo, from `ggml-base.bin` (142 MB, rougher) to the default `ggml-large-v3-turbo.bin` (1.6 GB, most accurate). Whisper understands about 99 languages and detects each memo's language by default.

## What a note looks like

- **File name:** date and title, like `2026-09-17 Launch Plan.md`, dated when the memo was recorded.
- **Contents:** a summary (longer for longer recordings), key points, decisions, and actions as checkboxes.
- **Links:** people and topics in `entities.json` become `[[links]]`, and new ones are added for next time.
- **Transcript:** the full transcript is saved in `Transcripts/` and linked from the note.
- **Repeats:** processing the same audio again creates `Title 2.md` and never changes the first note.
- **Templates:** the layout comes from [`note.md.j2`](https://github.com/rvnztolentino/voxmd/blob/main/src/voxmd/templates/note.md.j2). Copy it and set `render.template` to customise.

## Good to know

- **Local only.** voxmd talks to nothing but Ollama on your own machine.
- **Nothing runs in the background.** The watcher only runs while you have it open, and never starts on its own.
- **Nothing is overwritten or deleted.** A recording is only archived after its note is saved.
- **It can make mistakes.** Every note says so; check names and actions against the transcript or the recording.
- **Transcripts are private.** Notes and transcripts are only readable by your user, but they're stored unencrypted. If your vault syncs, they sync too; set `vault.transcripts: false` to skip them.
- **Updating.** `uv tool upgrade voxmd`. See the [changelog](https://github.com/rvnztolentino/voxmd/blob/main/CHANGELOG.md) for what changed.
- **No speaker labels.** voxmd can't tell who said what in a meeting.
- **Exit codes:** `0` done · `2` config · `3` missing tool · `4` bad input · `5` tool failed · `6` timeout · `7` note not written · `8` note written, a later step failed.

## Tech stack

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for transcription, [Ollama](https://ollama.com) for extraction, ffmpeg for audio
- [typer](https://typer.tiangolo.com), [pydantic](https://docs.pydantic.dev), [Jinja2](https://jinja.palletsprojects.com), [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/), [watchdog](https://github.com/gorakhargosh/watchdog)

## Development

```sh
git clone https://github.com/rvnztolentino/voxmd.git
cd voxmd
uv sync
uv run voxmd --help
uv run pytest            # no models, tools or network needed
uv run ruff check
uv run ruff format --check
```

## License

[MIT](https://github.com/rvnztolentino/voxmd/blob/main/LICENSE) · [Changelog](https://github.com/rvnztolentino/voxmd/blob/main/CHANGELOG.md)
