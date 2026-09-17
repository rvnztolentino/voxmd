---
description: Measure voxmd's runtime promises (C1, C4, C5, C8) on this machine with a synthetic memo
---

Unit tests can't prove the runtime invariants in `.claude/rules/invariants.md`. This command measures them for real. It needs macOS (`say`), ffmpeg, whisper-cli with its model, and Ollama running. If any is missing, report which one and stop.

Use a scratch directory and a synthetic memo only. Never use the user's real vault, config, ledger, watch folder or recordings.

1. **Setup.**
   - Create `T=$(mktemp -d)` with `vault/`, `inbox/`, `archive/` and `state/` inside.
   - Write `$T/voxmd.yaml`:
     - `whisper.model`: the user's configured model path, read from `voxmd doctor` output (don't copy the rest of their config)
     - `vault.path: $T/vault`, `archive.dir: $T/archive`, `watch.dir: $T/inbox`, `state.dir: $T/state`
     - `entities.file: $T/entities.json`, `log.file: $T/state/voxmd.log`
   - Make a memo of about 10 seconds: `say -o "$T/memo.aiff" "<a short made-up meeting with a decision and an action>"`, then `ffmpeg -loglevel error -i "$T/memo.aiff" "$T/memo.m4a"`.
   - Run every voxmd command from `$T`, so `./voxmd.yaml` wins over the user's config.
   - Record `launchctl list | grep -i voxmd`, `ls ~/Library/LaunchAgents` and `crontab -l`, to compare in step 5.
2. **C5: never both models.**
   - Start a sampler in the background that every 0.5s logs a timestamp plus `ps -axo pid,rss,comm` lines matching `whisper-cli` or `ollama runner`.
   - Run `uv run voxmd process "$T/memo.m4a" -v`.
   - Pass: no sample contains both processes. Report each one's first and last timestamp and its peak RSS, and confirm `ollama ps` is empty afterwards.
   - Exclude the sampler's own `grep` from the matches.
3. **C8: loopback only.**
   - While a second `process` run is in progress (copy the memo back first), sample `lsof -nP -iTCP -iUDP -a -p <voxmd pid>` repeatedly.
   - Pass: the only connections go to `127.0.0.1:11434` or `[::1]:11434`.
4. **C4: idle watcher.**
   - Start `uv run voxmd watch` in the background.
   - Once it has logged `watch.start` and `scan`, read the CPU time with `ps -o time= -p <pid>`. Wait 300 seconds and read it again.
   - Pass: the difference is at most one `ps` tick (0.01s), and RSS stays flat.
   - `voxmd status` should report it as running, with its PID.
5. **C1: dies with its terminal, installs nothing.**
   - Send the watcher `SIGHUP`. Pass: it exits at once and the log ends with `watch.stop reason=SIGHUP`.
   - `voxmd status` must now say not running (stale), not running.
   - Re-run the step 1 persistence checks. Pass: identical to before.
6. **C2: log.**
   - Every wake and every file in `$T/state/voxmd.log` has a timestamped line.
   - `grep` the log for words from the memo's text. Pass: nothing matches, because only metadata is logged.
7. **Report** a table with columns constraint / result / measured numbers. Remove `$T` afterwards. Don't paste memo content or note text into the report.
