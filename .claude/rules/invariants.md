# voxmd invariants

These are product promises, not style preferences. A change that breaks one is a bug even if every test passes. If a task seems to need breaking one, stop and ask.

| # | Promise | Enforced in | Guarded by |
|---|---|---|---|
| C1 | Nothing auto-starts. `voxmd watch` is foreground-only and dies with its terminal. No LaunchAgent, plist, cron, systemd unit, login item, `setsid`, `nohup` or daemonizing. | `watcher.py` (`_stop_on_signal`: SIGTERM/SIGHUP stop it) | `test_no_module_can_install_anything_that_survives_a_reboot` (a banned-word scan of every module, docstrings stripped) |
| C2 | Every wake and every processed file gets one line in the event log. The log holds metadata only, never transcript text, summaries or extracted fields. | `logging_setup.EventLog` (0600, `O_NOFOLLOW`, JSON-quoted values, flushed per line) | `test_logging_setup.py`, `test_watcher.py` |
| C3 | `voxmd status` reports: running or not, PID, files processed today, last wake. A dead or recycled PID reads as stale, never as running. | `status.py` (`pid_alive` and `pid_is_voxmd`; "today" comes from the ledger) | `test_status.py` |
| C4 | Watching is event-driven (FSEvents on macOS, inotify on Linux), never polling. An idle watcher uses ~0% CPU. | `watcher.check_observer` refuses `PollingObserver*`; the main thread blocks on `queue.get()` with no timeout | `test_this_machines_observer_is_event_driven`, `test_a_polling_observer_is_refused_rather_than_quietly_accepted` |
| C5 | whisper and the Ollama model are never in memory together. whisper has exited before Ollama is called, and Ollama is unloaded after every call. | `pipeline.assert_no_child_processes` (`os.waitid`, fails closed); `keep_alive=0` on every `chat` call, retries included | `test_a_child_process_still_running_stops_the_run_before_ollama`, `test_the_child_process_check_is_real_on_this_platform` |
| C8 | No network except Ollama on loopback. | `config.normalize_loopback_host`; `extract.make_client` (`trust_env=False`, no redirects) | `test_non_loopback_or_malformed_hosts_are_refused`, `test_extract_refuses_a_remote_host_before_connecting` |

C6 and C7 were one-off process decisions, not code rules.

## Consequences for code

- Don't add a dependency that makes network calls at import or at runtime. Audit any new package for telemetry and update checks before adding it, and flag anything suspicious.
- Never add `keep_alive`, `think` or a remote Ollama host as a config option; they are left out on purpose.
- `pipeline.process` runs its stages in a fixed order: transcribe → child check → extract → transcript note → summary note → ledger → archive. Don't reorder them. The recording is archived only after the note has been written and read back.
- Unit tests can't prove C4, C5 or C8 at runtime. After touching the watcher, the pipeline order or the Ollama client, run `/verify-constraints`.

## Privacy

- The repository is public. Never commit or paste real memo audio, transcripts, extraction JSON (`memo.txt`, `memo.json`), notes, `entities.json`, or a real `voxmd.yaml`. Test fixtures use invented names and paths only.
- Error messages and `-v` output must not echo memo content or model-written titles, except note paths where the command's job is to print them.
