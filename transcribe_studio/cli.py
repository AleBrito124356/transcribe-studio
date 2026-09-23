#!/usr/bin/env python3
"""transcribe-studio command-line interface.

Subcommands:
    transcribe   media -> transcript.txt (+ transcript.json)
    subs         media -> captions.srt / captions.vtt (+ optional translation)
    summarize    media or transcript -> summary.md   (needs NIM)
    chapters     media or transcript.json -> chapters.md
    all          full pipeline; pass a directory for batch mode

Run `python cli.py <command> -h` for per-command options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import chapters as chapters_mod
from . import subtitles as subs_mod
from . import summarize as summarize_mod
from .config import load_env, load_json, save_json
from .diarize import PYANNOTE_NOTE, assign_speakers, to_dialogue
from .nim import MissingApiKey, NimClient, NimError
from .pipeline import PipelineOptions, run, run_batch
from .transcribe import (
    FfmpegNotFound,
    MediaDecodeError,
    WhisperUnavailable,
    segments_to_paragraphs,
    transcribe,
)

# Exit codes (also documented in the README).
EXIT_OK = 0
EXIT_PARTIAL = 1       # batch finished but some files failed; or an unexpected error
EXIT_NO_KEY = 2        # a NIM-only feature was requested without NVIDIA_API_KEY
EXIT_NO_FFMPEG = 3
EXIT_NOT_FOUND = 4
EXIT_NIM = 5           # NIM rejected the key, rate limited us, or was unreachable
EXIT_WHISPER = 6       # faster-whisper missing, or the model could not be loaded
EXIT_MEDIA = 7         # ffmpeg could not decode the input
EXIT_INTERRUPTED = 130


def _print_retry(attempt: int, delay: float, reason: str) -> None:
    print(f"note: NIM {reason}; retrying in {delay:.1f}s (retry {attempt})", file=sys.stderr)


def _nim_client(model: str | None = None) -> NimClient:
    return NimClient.from_env(model=model, on_retry=_print_retry)


def _add_common_transcribe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lang", default=None, help="Force language code (e.g. en, es). Default: auto-detect.")
    p.add_argument("--model", default="base", help="Whisper size: tiny|base|small|medium|large-v3 (default: base).")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda (default: auto).")
    p.add_argument("--compute-type", default="int8", help="int8|int8_float16|float16 (default: int8).")
    p.add_argument("--chunk-length", type=float, default=0.0, help="Chunk seconds for very long files (0 = off).")


def _require_nim(model: str | None):
    try:
        return _nim_client(model)
    except MissingApiKey as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_NO_KEY)


def _load_segments_for_summary(source: str, args):
    """Return (full_text, language). Accepts a .txt, a .json transcript, or media."""
    path = Path(source)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8"), (args.lang or "auto")
    if path.suffix.lower() == ".json":
        tr = load_json(path)
        return tr.text, tr.language
    tr = transcribe(source, model_size=args.model, language=args.lang,
                    device=args.device, compute_type=args.compute_type,
                    chunk_length=args.chunk_length)
    return tr.text, tr.language


def _load_segments_for_chapters(source: str, args):
    path = Path(source)
    if path.suffix.lower() == ".json":
        return load_json(path).segments
    tr = transcribe(source, model_size=args.model, language=args.lang,
                    device=args.device, compute_type=args.compute_type,
                    chunk_length=args.chunk_length)
    return tr.segments


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def cmd_transcribe(args) -> int:
    tr = transcribe(
        args.media,
        model_size=args.model,
        language=args.lang,
        device=args.device,
        compute_type=args.compute_type,
        chunk_length=args.chunk_length,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.diarize:
        labelled = assign_speakers(tr.segments)
        body = PYANNOTE_NOTE + "\n\n" + to_dialogue(labelled)
    else:
        body = segments_to_paragraphs(tr.segments)
    (out_dir / "transcript.txt").write_text(body.strip() + "\n", encoding="utf-8")
    save_json(tr, out_dir / "transcript.json")

    print(f"Language: {tr.language}  |  Duration: {tr.duration:.1f}s  |  Segments: {len(tr.segments)}")
    print(f"Wrote {out_dir / 'transcript.txt'}")
    print(f"Wrote {out_dir / 'transcript.json'}")
    return 0


def cmd_subs(args) -> int:
    tr = transcribe(
        args.media,
        model_size=args.model,
        language=args.lang,
        device=args.device,
        compute_type=args.compute_type,
        chunk_length=args.chunk_length,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    subs_mod.write_srt(tr.segments, out_dir / "captions.srt", max_chars=args.max_chars)
    subs_mod.write_vtt(tr.segments, out_dir / "captions.vtt", max_chars=args.max_chars)
    print(f"Wrote {out_dir / 'captions.srt'}")
    print(f"Wrote {out_dir / 'captions.vtt'}")

    if args.translate:
        client = _require_nim(args.model_nim)
        translated = subs_mod.translate_segments(tr.segments, args.translate, client)
        t_srt = out_dir / f"captions.{args.translate}.srt"
        t_vtt = out_dir / f"captions.{args.translate}.vtt"
        subs_mod.write_srt(translated, t_srt, max_chars=args.max_chars)
        subs_mod.write_vtt(translated, t_vtt, max_chars=args.max_chars)
        print(f"Wrote {t_srt}")
        print(f"Wrote {t_vtt}")
    return 0


def cmd_summarize(args) -> int:
    client = _require_nim(args.model_nim)
    text, language = _load_segments_for_summary(args.source, args)
    lang_phrase = summarize_mod.resolve_language(args.summary_lang, language)
    summary = summarize_mod.summarize_transcript(text, client, language=lang_phrase)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "summary.md"
    out_file.write_text(summarize_mod.to_markdown(summary), encoding="utf-8")
    print(summary.get("tldr", ""))
    print(f"Wrote {out_file}")
    return 0


def cmd_chapters(args) -> int:
    segments = _load_segments_for_chapters(args.source, args)
    client = None
    if not args.local and NimClient.available():
        client = _require_nim(args.model_nim)
    notes: list = []
    chapter_list = chapters_mod.detect_chapters(segments, nim_client=client, warnings=notes)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chapters.md"
    out_file.write_text(chapters_mod.chapters_to_markdown(chapter_list), encoding="utf-8")
    print(chapters_mod.chapters_to_text(chapter_list))
    print(f"Wrote {out_file}")
    return 0


def cmd_all(args) -> int:
    options = PipelineOptions(
        model_size=args.model,
        language=args.lang,
        device=args.device,
        compute_type=args.compute_type,
        chunk_length=args.chunk_length,
        translate=args.translate,
        diarize=args.diarize,
        summary_language=args.summary_lang,
        use_nim=not args.no_nim,
        max_chars=args.max_chars,
    )
    # Build the NIM client here (not inside the pipeline) so retries are
    # reported on stderr while the run waits out a rate limit.
    client = _nim_client() if (options.use_nim and NimClient.available()) else None
    source = Path(args.source)
    if source.is_dir():
        results = run_batch(str(source), args.out, options, nim_client=client)
        ok = sum(1 for r in results if r.ok)
        print(f"Batch complete: {ok}/{len(results)} succeeded. Report in {Path(args.out) / 'batch_report.md'}")
        return 0 if ok == len(results) else 1

    result = run(str(source), args.out, options, nim_client=client)
    for w in result.warnings:
        print(f"note: {w}", file=sys.stderr)
    print(f"Language: {result.language}  |  Duration: {result.duration:.1f}s")
    for out in result.outputs:
        print(f"Wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser(prog: str = "transcribe-studio") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Turn any audio or video into transcripts, subtitles, summaries and chapters.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true",
                        help="Show full tracebacks instead of one-line error messages.")
    sub = parser.add_subparsers(dest="command", required=True)

    # transcribe
    p_t = sub.add_parser("transcribe", help="Transcribe media to transcript.txt/json.")
    p_t.add_argument("media")
    p_t.add_argument("-o", "--out", default="results")
    p_t.add_argument("--diarize", action="store_true", help="Add approximate speaker labels.")
    _add_common_transcribe_args(p_t)
    p_t.set_defaults(func=cmd_transcribe)

    # subs
    p_s = sub.add_parser("subs", help="Generate SRT and VTT subtitles.")
    p_s.add_argument("media")
    p_s.add_argument("-o", "--out", default="results")
    p_s.add_argument("--translate", default=None, help="Also write a translated track (e.g. --translate es).")
    p_s.add_argument("--max-chars", type=int, default=subs_mod.DEFAULT_MAX_CHARS, help="Max characters per subtitle line.")
    p_s.add_argument("--model-nim", default=None, help="Override NIM model for translation.")
    _add_common_transcribe_args(p_s)
    p_s.set_defaults(func=cmd_subs)

    # summarize
    p_sum = sub.add_parser("summarize", help="Summarize media, a .txt, or a .json transcript.")
    p_sum.add_argument("source")
    p_sum.add_argument("-o", "--out", default="results")
    p_sum.add_argument("--summary-lang", default="auto", help="Summary language: auto|en|es|... (default: auto).")
    p_sum.add_argument("--model-nim", default=None, help="Override NIM model.")
    _add_common_transcribe_args(p_sum)
    p_sum.set_defaults(func=cmd_summarize)

    # chapters
    p_c = sub.add_parser("chapters", help="Generate YouTube-style chapters.")
    p_c.add_argument("source", help="Media file or a transcript.json.")
    p_c.add_argument("-o", "--out", default="results")
    p_c.add_argument("--local", action="store_true", help="Force the offline heuristic (no NIM).")
    p_c.add_argument("--model-nim", default=None, help="Override NIM model.")
    _add_common_transcribe_args(p_c)
    p_c.set_defaults(func=cmd_chapters)

    # all
    p_a = sub.add_parser("all", help="Full pipeline. Pass a directory for batch mode.")
    p_a.add_argument("source", help="Media file or a directory of media files.")
    p_a.add_argument("-o", "--out", default="results")
    p_a.add_argument("--translate", default=None, help="Add a translated subtitle track (e.g. es).")
    p_a.add_argument("--summary-lang", default="auto")
    p_a.add_argument("--diarize", action="store_true", help="Add approximate speaker labels.")
    p_a.add_argument("--no-nim", action="store_true", help="Skip all NIM features (offline only).")
    p_a.add_argument("--max-chars", type=int, default=subs_mod.DEFAULT_MAX_CHARS)
    _add_common_transcribe_args(p_a)
    p_a.set_defaults(func=cmd_all)

    return parser


def main(argv=None, prog: str = "transcribe-studio") -> int:
    load_env()
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.debug:
            raise
        code, message = _explain(exc)
        print(message, file=sys.stderr)
        return code


def _explain(exc: Exception) -> tuple:
    """Map an exception to (exit code, friendly message) — never a traceback."""
    if isinstance(exc, FfmpegNotFound):
        return EXIT_NO_FFMPEG, str(exc)
    if isinstance(exc, FileNotFoundError):
        return EXIT_NOT_FOUND, f"error: {exc}"
    if isinstance(exc, NimError):
        return EXIT_NIM, f"error: {exc}"
    if isinstance(exc, MissingApiKey):
        return EXIT_NO_KEY, str(exc)
    if isinstance(exc, WhisperUnavailable):
        return EXIT_WHISPER, f"error: {exc}"
    if isinstance(exc, MediaDecodeError):
        return EXIT_MEDIA, f"error: {exc}"
    return EXIT_PARTIAL, (
        f"error: unexpected {type(exc).__name__}: {exc}\n"
        "Re-run with --debug (before the command name) to see the traceback."
    )


if __name__ == "__main__":
    raise SystemExit(main())
