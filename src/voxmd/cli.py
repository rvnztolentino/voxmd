"""Command line entry point.

This module imports **only** typer and the standard library at module scope.
Every stage module is imported inside the command that needs it.

That is deliberate and load-bearing for startup time: the pipeline's heavier
dependencies (ollama's httpx stack, watchdog's platform observers, jinja2,
rapidfuzz) would otherwise be imported on every invocation, including
``voxmd --help`` and including commands that never touch them. Keep new
imports inside their command.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from .errors import VoxmdError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Voice memo to structured markdown notes. Fully local.",
)


@app.callback()
def _root() -> None:
    """Voice memo to structured markdown notes. Fully local."""
    # With a single command, typer would otherwise make `transcribe` the root
    # command and treat the word "transcribe" as its audio argument.


@app.command()
def transcribe(
    audio: Annotated[
        Path,
        typer.Argument(help="Audio file to transcribe.", show_default=False),
    ],
    model: Annotated[
        Path | None,
        typer.Option(
            "--model",
            "-m",
            help="Path to ggml whisper weights. Overrides whisper.model in config.",
            show_default=False,
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code, or 'auto' to detect. Overrides config.",
            show_default=False,
        ),
    ] = None,
    threads: Annotated[
        int | None,
        typer.Option(
            "--threads",
            "-t",
            min=1,
            help="Threads for whisper. Defaults to the performance-core count.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Config file. Defaults to $VOXMD_CONFIG, ./voxmd.yaml, then "
            "~/.config/voxmd/config.yaml.",
            show_default=False,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print timing and format details to stderr."),
    ] = False,
) -> None:
    """Transcribe an audio file and print the text to stdout.

    Works without a config file — pass --model and go. Text goes to stdout and
    everything else to stderr, so this pipes cleanly into a file or another
    command.
    """
    from .config import load_config
    from .transcribe import default_threads
    from .transcribe import transcribe as run_transcribe

    settings = load_config(config)
    from .config import apply_overrides

    whisper = apply_overrides(settings.whisper, model=model, language=language, threads=threads)

    if verbose:
        _note(f"source:  {audio}")
        _note(f"threads: {whisper.threads or default_threads()}")

    result = run_transcribe(audio, whisper=whisper, limits=settings.limits)

    if verbose:
        probe = result.probe
        duration = f"{probe.duration_s:.1f}s" if probe.duration_s else "unknown"
        _note(f"format:  {probe.codec} {probe.sample_rate}Hz {probe.channels}ch, {duration}")
        _note(f"ffmpeg:  {'converted' if result.converted else 'skipped (already 16kHz mono)'}")
        _note(f"whisper: {result.whisper_seconds:.1f}s")
        if probe.duration_s:
            _note(f"speed:   {probe.duration_s / result.whisper_seconds:.1f}x realtime")

    # Only the transcript goes to stdout.
    print(result.text)


@app.command()
def extract(
    transcript: Annotated[
        Path | None,
        typer.Argument(
            help="Transcript text file. Reads stdin when omitted or '-'.",
            show_default=False,
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Ollama model name. Overrides ollama.model in config.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Config file. Defaults to $VOXMD_CONFIG, ./voxmd.yaml, then "
            "~/.config/voxmd/config.yaml.",
            show_default=False,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print model, token, and timing details to stderr."),
    ] = False,
) -> None:
    """Extract title, summary, decisions, actions, people, and topics as JSON.

    Pipes from transcribe: voxmd transcribe memo.m4a | voxmd extract

    Talks only to Ollama on localhost, and unloads the model when done.
    """
    from .config import apply_overrides, load_config
    from .extract import extract as run_extract
    from .extract import read_transcript

    settings = load_config(config)
    ollama_cfg = apply_overrides(settings.ollama, model=model)
    text = read_transcript(
        transcript, max_bytes=settings.limits.max_transcript_bytes, stdin=sys.stdin
    )

    if verbose:
        _note(f"model:    {ollama_cfg.model} @ {ollama_cfg.host}")

    result = run_extract(text, ollama_cfg=ollama_cfg, limits=settings.limits)

    if result.truncated:
        typer.secho(
            f"voxmd: warning: the transcript may have been truncated to fit "
            f"ollama.num_ctx ({ollama_cfg.num_ctx}); the end of the memo may be missing. "
            "Raise ollama.num_ctx to include it.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    if verbose:
        _note(f"attempts: {result.attempts}")
        _note(
            f"context:  {result.num_ctx} tokens "
            f"({result.prompt_tokens} prompt, {result.output_tokens} output)"
        )
        _note(f"load:     {result.load_seconds:.1f}s")
        _note(f"total:    {result.seconds:.1f}s")

    # Only the JSON goes to stdout.
    print(result.extraction.model_dump_json(indent=2))


def _note(message: str) -> None:
    typer.secho(message, fg=typer.colors.BRIGHT_BLACK, err=True)


def main() -> None:
    """Console-script entry point.

    Expected failures print one clear line and exit with a distinct code.
    Anything else keeps its traceback — an unexpected exception is a bug, and
    swallowing it would only make it harder to fix.
    """
    try:
        app()
    except VoxmdError as exc:
        typer.secho(f"voxmd: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(exc.exit_code) from exc
    except KeyboardInterrupt:
        typer.secho("voxmd: interrupted", fg=typer.colors.YELLOW, err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
