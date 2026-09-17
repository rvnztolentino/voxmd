# Changelog

All notable changes to voxmd are listed here. Versions follow [Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-09-17

First release.

### Added

- `voxmd process`: transcribes a recording with whisper.cpp, pulls out a title, summary, key points, decisions, actions, people and topics with a local Ollama model, and writes a note into your Obsidian vault.
- `voxmd watch`: watches a folder in the foreground and processes each memo once it has finished syncing. `voxmd status` shows whether the watcher is running, when it last woke, and how many notes it made today.
- The full transcript is saved as its own note and linked from the summary note (`vault.transcripts`).
- Known people and topics become `[[links]]`, and new ones are remembered in `entities.json`.
- Processing the same audio again makes a second note (`Title 2.md`) and never changes the first (`vault.duplicates`).
- `voxmd doctor` checks the setup without changing anything. `voxmd models` lists the Ollama models you have.
- `transcribe`, `extract` and `render` run one step at a time and can be piped together.
- A log of every wake and every processed file in `~/.local/state/voxmd/voxmd.log`.
- `voxmd --version`.

### Notes

- Runs on macOS and Linux. Windows is not supported.
- Nothing leaves your machine: the only network connection is to Ollama on localhost.

[0.1.0]: https://github.com/rvnztolentino/voxmd/releases/tag/v0.1.0
