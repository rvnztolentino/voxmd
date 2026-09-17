# voxmd code conventions

## Commands

```sh
uv sync                      # Python >=3.13; .python-version pins 3.14
uv run pytest                # needs no models, tools or network
uv run ruff check
uv run ruff format --check
```

Run all three before calling work done. CI (`.github/workflows/release.yml`) runs them on macOS and Linux, Python 3.13 and 3.14.

## Structure

- `cli.py` imports only typer, the standard library, `errors` and the package version at module scope. Every stage module is imported inside its command. `test_help_imports_no_stage_modules_or_heavy_dependencies` enforces this; keep `voxmd --help` around 0.1s.
- Each stage is a module (`transcribe`, `extract`, `render`, `pipeline`, `watcher`). `pipeline.process` is the only place the stages are joined, and the watcher calls it once per file.
- Config is pydantic in `config.py`, with `extra="forbid"`. A new setting needs:
  - a field with a validator
  - a commented entry in `voxmd.example.yaml`
  - a `doctor` check if the setting refers to a path or an external tool
  - a README mention if users need to know about it

  `test_the_shipped_example_config_is_valid` loads the example file.

## Safety primitives: use `safe.py`, don't reimplement

- Subprocesses go only through `safe.run`: list argv, never `shell=True`, a mandatory timeout. The one `# noqa: S603` lives there.
- Input paths go through `safe.resolve_input_file` or `safe.read_text_input` (suffix allowlist, size caps, regular files only — never a FIFO or device).
- Writes:
  - new files: `safe.write_new_file` (hidden temp file, fsync, hard-link claim, a ` 2` suffix instead of overwriting)
  - rewrites: `safe.atomic_write_text`
  - moves: `safe.move_no_replace`
  - locks: `safe.file_lock`
- File modes are 0600 and directories 0700. voxmd never overwrites or deletes a user file, and never creates the vault or the watch folder.
- Vault paths are re-checked with `pipeline.contained_in` after the folder exists, so a symlink can't lead outside the vault.

## Errors and output

- Expected failures raise a `VoxmdError` subclass from `errors.py`; each carries its exit code: 2 config, 3 dependency, 4 input, 5 tool, 6 timeout, 7 no note written, 8 note written but a later step failed. `cli.main` prints one line and exits with that code. Don't add a traceback path for expected errors.
- stdout carries only the command's product (transcript, JSON, markdown, note path). Diagnostics, `-v` output and the watcher's log echo go to stderr. `voxmd watch` writes nothing to stdout.
- Messages say what to do next and point to the README, since users have no `setup.md`.

## Model output is untrusted

- Everything from Ollama is validated by `schema.Extraction` and escaped in `render.py`: wikilinks, embeds, HTML, `%%` comments and tags are neutralized. Only names already in `entities.json` become links, via `entities.link_target`.
- Treat any change to escaping, file naming (`pipeline.note_filename`) or `NOT_NAMES`/`looks_like_a_name` as security-relevant, and add a hostile-input test.

## Platform and dependencies

- macOS and Linux only (`fcntl`, `os.waitid`, POSIX modes). Don't add Windows branches piecemeal.
- Pin runtime and dev dependencies exactly (`==`), and update `uv.lock` with them. `watchdog` stays out of the startup path.
- The note template is package data (`src/voxmd/templates/note.md.j2`). If packaging changes, check that the wheel still contains it.

## Tests

- Tests use fakes (`conftest.FakeRunner`, `FakeClient`, `FakeOllama`) and `tmp_path`. They never touch the real `~/.config/voxmd`, vault or Ollama. CLI tests reset `HOME` and `VOXMD_CONFIG`.
- Platform-dependent expectations switch on `sys.platform` (see the observer test), so CI passes on Linux.
