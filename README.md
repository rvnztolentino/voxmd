# voxmd

Voice memo → structured markdown notes. Fully local. CLI only.

Drop an audio file in, get a linked note in an Obsidian vault. No cloud, no API keys, no cost.

> **Status:** stage 5 of 6, the MVP. `voxmd process` turns a recording into a note in your vault. Watch mode is deliberately not built yet: `process` gets used by hand first.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- ffmpeg (`brew install ffmpeg`)
- whisper.cpp (`brew install whisper-cpp`) plus a ggml model file
- [Ollama](https://ollama.com) running locally, with `qwen3:8b` pulled

The full setup walkthrough, including the model downloads, lives in the project's setup guide.

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

### Extract

```sh
voxmd extract memo.txt
voxmd transcribe memo.m4a | voxmd extract
```

Prints JSON with `title`, `summary`, `decisions`, `actions`, `people`, and `topics`. It uses one Ollama call, constrained to that schema, with one retry if the reply doesn't validate. `-m` picks another model and `-v` prints tokens and timing to stderr. If a transcript is too long for `ollama.num_ctx`, it is cut to fit and a warning is printed.

### Render

```sh
voxmd extract memo.txt | voxmd render --source memo.m4a --duration 205
voxmd render memo.json -v
```

Prints a markdown note: YAML frontmatter (`date`, `source`, `duration`), the title, the summary, decisions, actions as `- [ ]` checkboxes, and related people and topics. `--date` sets when the memo was recorded (ISO 8601; defaults to now). Nothing is written unless you pass `--update-entities`.

**Template.** The layout lives in [`src/voxmd/templates/note.md.j2`](src/voxmd/templates/note.md.j2), which lists the variables it receives. Copy it, edit the copy, and set `render.template` or pass `--template`.

**Entities.** `~/.config/voxmd/entities.json` lists the people and topics you know:

```json
{"people": ["Marco", "Ana"], "topics": ["release"]}
```

Known names become `[[wikilinks]]`: in the People and Topics lists, and for people also wherever they're mentioned in the summary, decisions, and actions. Matching ignores case, accents, and punctuation, and tolerates a small spelling slip in a longer name ("Christophor" links to `[[Christopher]]`) without merging different short names ("Marcus" stays apart from "Marco"). Names that aren't in the file stay plain text; `--update-entities` appends them so they link next time. Existing entries are never rewritten, and a malformed file is refused rather than overwritten.

### Process

```sh
voxmd doctor
voxmd process memo.m4a --vault ~/Documents/Obsidian/Main -v
```

Runs transcribe, extract, and render, writes the note into your vault, and prints its path. Set `vault.path` (and optionally `vault.folder`) in your config to drop the flag.

- **Named by date and title**, e.g. `2026-09-15 Website Launch Update.md`. The date comes from the recording's `creation_time` tag, or the file's modification time if it has none. `--date` overrides both.
- **Never overwrites.** If a note with that name exists, the new one is saved as `... 2.md`.
- **Skips what's done.** A ledger in `~/.local/state/voxmd` records each recording by the SHA-256 of its audio. Processing the same memo again, even renamed, prints the existing note instead of writing another. `--force` processes it anyway.
- **Archives last.** With `archive.dir` set, the recording moves there only after the note is written and read back. If anything fails before that, the recording stays where it is, so running it again retries. `--no-archive` leaves it in place for one run.
- **Learns names.** New people and topics are appended to the entities file, so the next note links them.

Exit codes: 0 done or skipped, 2 config, 3 missing dependency, 4 bad input, 5 tool failure, 6 timeout, 7 note not written (recording untouched), 8 note written but a later step failed (the warning says which).

### Doctor

`voxmd doctor` checks ffmpeg, whisper-cli and its model, Ollama and the configured model, the vault, archive and state folders, the template, and the entities file. It changes nothing: no installs, downloads, model pulls, or new folders.

A config file is optional for `transcribe`, `extract`, and `render`; `process` needs a vault from `vault.path` or `--vault`. See [`voxmd.example.yaml`](voxmd.example.yaml); voxmd looks in `$VOXMD_CONFIG`, then `./voxmd.yaml`, then `~/.config/voxmd/config.yaml`. CLI flags override config.

## What it does and doesn't do

These are commitments, not aspirations.

**It never auto-starts.** voxmd installs no LaunchAgent, plist, cron entry, or login item. Nothing runs unless you start it, and nothing keeps running after you close the terminal.

**The only network connection is to Ollama on localhost.** `ollama.host` must be a loopback address; anything else is refused at config load. The client also ignores `HTTP_PROXY`/`ALL_PROXY` and macOS proxy settings, refuses redirects, and ignores `$OLLAMA_HOST`, so a transcript can't be routed off the machine by environment. No telemetry, no update checks, and none of voxmd's Python dependencies phone home. Honest caveats:

- Setup involves downloads you do yourself: the whisper weights from Hugging Face and the Ollama model. voxmd never downloads anything.
- The `ollama` Python library contains `web_search`/`web_fetch` helpers that call ollama.com. voxmd never calls them.
- Ollama is a separate program. The Ollama app may check for its own updates and may register itself as a login item. That's Ollama's behaviour, not voxmd's.

**It never holds two models in memory.** whisper runs as a subprocess and exits before Ollama is asked to load anything. Every Ollama call passes `keep_alive=0`, so the model unloads as soon as it answers, and a failed request sends an explicit unload. `voxmd process` also checks that it has no child process left before it calls Ollama, rather than assuming whisper is gone. This costs a few seconds of model load per memo in exchange for voxmd occupying no memory between memos.

**It never overwrites or deletes your files.** Notes are written under a new name, never over an existing file, and a notes folder that is a symlink leading out of the vault is refused. Recordings are moved, never deleted, and only once their note is safely on disk and the recording hasn't changed since it was processed. A corrupt ledger or entities file is refused, not reset.

**Model output can't reshape a note.** Everything the model writes is escaped before it reaches the template, so a memo can't inject links, embeds, `%%` comments, tags, or raw HTML (an `<img>` would load a remote URL when the note is opened). Frontmatter is written with `yaml.safe_dump`, wikilinks are built only from entity names, and the template runs in Jinja's sandbox.

**Your data stays on disk, unencrypted.** Transcripts and notes are plaintext files. The ledger records file paths, sizes, durations, and times, never what was said, though a note's path does contain its title. Error messages never include transcript text or model output.

## Resource usage

- `voxmd --help` starts in under 0.1s; Ollama, httpx, pydantic, jinja2, and rapidfuzz are only imported by the commands that use them.
- `voxmd render` runs in about 0.12s and never imports the Ollama client. Name lookups are a dict hit; rapidfuzz only runs when a name isn't an exact match, and such a lookup against 10,000 known names takes about 3 ms.
- `voxmd process` on an 11-second memo took 17.4s end to end: whisper 2.8s, then Ollama 13.8s, of which 4.3s was loading qwen3:8b. Sampled every 0.5s, whisper-cli peaked at 1.9 GB and had exited before the Ollama runner started; the runner peaked at 5.3 GB and exited as the note was written. They never overlapped.
- The ledger hashes audio in 1 MB chunks, at about 2 GB/s with 35 MB peak memory for a 200 MB file.
- `voxmd extract` sizes the context window to the transcript, so a short memo uses a 4k-token window instead of the 16k ceiling. qwen3:8b needs roughly 5–6 GB while loaded, and nothing between memos.
- Idle CPU for watch mode will be measured and recorded here once watch mode exists. Not an estimate; a measurement.

## Development

```sh
uv run pytest
uv run ruff check
```

The test suite needs no ffmpeg, whisper, Ollama, model weights, or network: external tools are faked at the single point where voxmd spawns them, and the Ollama client is replaced by a fake.
