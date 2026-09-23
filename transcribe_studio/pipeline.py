"""One-command pipeline: media file -> transcript, subtitles, summary, chapters.

`run` processes a single file into an output folder. `run_batch` walks a
directory, giving every file its own subfolder and isolating failures so one bad
file never sinks the whole run.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import chapters as chapters_mod
from . import diarize as diarize_mod
from . import subtitles as subs_mod
from . import summarize as summarize_mod
from .config import MEDIA_EXTENSIONS, TranscriptResult, save_json
from .nim import MissingApiKey, NimClient
from .transcribe import segments_to_paragraphs, transcribe


@dataclass
class PipelineOptions:
    """Everything the pipeline needs to know, with sensible defaults."""

    model_size: str = "base"
    language: Optional[str] = None
    device: str = "auto"
    compute_type: str = "int8"
    chunk_length: float = 0.0
    translate: Optional[str] = None  # target language code for a translated subtitle track
    diarize: bool = False
    diarize_gap: float = 1.2
    summary_language: str = "auto"
    make_transcript: bool = True
    make_subtitles: bool = True
    make_summary: bool = True
    make_chapters: bool = True
    use_nim: bool = True
    max_chars: int = subs_mod.DEFAULT_MAX_CHARS
    save_json_transcript: bool = True


@dataclass
class PipelineResult:
    source: str
    output_dir: str
    outputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    ok: bool = True
    error: Optional[str] = None


def _maybe_nim(options: PipelineOptions, warnings: List[str]) -> Optional[NimClient]:
    """Build a NIM client if the pipeline wants one and a key is available."""
    if not options.use_nim:
        return None
    if not NimClient.available():
        warnings.append(
            "NVIDIA_API_KEY not set — skipping summary/translation and using the "
            "offline chapter heuristic. Get a free key at build.nvidia.com."
        )
        return None
    try:
        return NimClient.from_env()
    except MissingApiKey as exc:  # pragma: no cover - guarded by available()
        warnings.append(str(exc))
        return None


def run(
    media_path: str,
    out_dir: str,
    options: Optional[PipelineOptions] = None,
    model=None,
    nim_client: Optional[NimClient] = None,
) -> PipelineResult:
    """Process one media file into ``out_dir``.

    ``model`` (a preloaded WhisperModel) and ``nim_client`` can be supplied to
    reuse resources across a batch. When ``nim_client`` is ``None`` it is created
    on demand from the environment.
    """
    options = options or PipelineOptions()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result = PipelineResult(source=str(media_path), output_dir=str(out_path))

    if nim_client is None:
        nim_client = _maybe_nim(options, result.warnings)

    # 1. Transcribe (local, always).
    transcript: TranscriptResult = transcribe(
        media_path,
        model_size=options.model_size,
        language=options.language,
        device=options.device,
        compute_type=options.compute_type,
        chunk_length=options.chunk_length,
        model=model,
    )
    result.language = transcript.language
    result.duration = transcript.duration

    segments = transcript.segments
    diarized_segments = None
    if options.diarize:
        diarized_segments = diarize_mod.assign_speakers(
            segments, gap_threshold=options.diarize_gap
        )

    # 2. Transcript text.
    if options.make_transcript:
        transcript_file = out_path / "transcript.txt"
        if diarized_segments is not None:
            body = (
                diarize_mod.PYANNOTE_NOTE
                + "\n\n"
                + diarize_mod.to_dialogue(diarized_segments)
            )
        else:
            body = segments_to_paragraphs(segments)
        transcript_file.write_text(body.strip() + "\n", encoding="utf-8")
        result.outputs.append(str(transcript_file))

    if options.save_json_transcript:
        json_file = out_path / "transcript.json"
        save_json(transcript, json_file)
        result.outputs.append(str(json_file))

    # 3. Subtitles (local).
    if options.make_subtitles:
        srt_file = out_path / "captions.srt"
        vtt_file = out_path / "captions.vtt"
        subs_mod.write_srt(segments, srt_file, max_chars=options.max_chars)
        subs_mod.write_vtt(segments, vtt_file, max_chars=options.max_chars)
        result.outputs.extend([str(srt_file), str(vtt_file)])

        # 3b. Translated subtitle track (needs NIM).
        if options.translate:
            if nim_client is None:
                result.warnings.append(
                    f"Skipped translation to '{options.translate}': no NIM key."
                )
            else:
                translated = subs_mod.translate_segments(
                    segments, options.translate, nim_client
                )
                t_srt = out_path / f"captions.{options.translate}.srt"
                t_vtt = out_path / f"captions.{options.translate}.vtt"
                subs_mod.write_srt(translated, t_srt, max_chars=options.max_chars)
                subs_mod.write_vtt(translated, t_vtt, max_chars=options.max_chars)
                result.outputs.extend([str(t_srt), str(t_vtt)])

    # 4. Summary (needs NIM).
    if options.make_summary:
        if nim_client is None:
            result.warnings.append("Skipped summary.md: no NIM key.")
        else:
            language = summarize_mod.resolve_language(
                options.summary_language, transcript.language
            )
            summary = summarize_mod.summarize_transcript(
                transcript.text, nim_client, language=language
            )
            summary_file = out_path / "summary.md"
            summary_file.write_text(
                summarize_mod.to_markdown(summary), encoding="utf-8"
            )
            result.outputs.append(str(summary_file))

    # 5. Chapters (NIM when available, else offline heuristic).
    if options.make_chapters:
        chapter_list = chapters_mod.detect_chapters(segments, nim_client=nim_client)
        chapters_file = out_path / "chapters.md"
        chapters_file.write_text(
            chapters_mod.chapters_to_markdown(chapter_list), encoding="utf-8"
        )
        result.outputs.append(str(chapters_file))

    return result


def find_media_files(input_dir: str, extensions=MEDIA_EXTENSIONS) -> List[Path]:
    """Return sorted media files directly inside ``input_dir``."""
    root = Path(input_dir)
    files = [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return files


def run_batch(
    input_dir: str,
    out_dir: str,
    options: Optional[PipelineOptions] = None,
) -> List[PipelineResult]:
    """Process every media file in ``input_dir`` with per-file error isolation.

    The whisper model is loaded once and reused across files. A failure on one
    file is recorded and the batch continues.
    """
    options = options or PipelineOptions()
    files = find_media_files(input_dir)
    results: List[PipelineResult] = []

    warnings: List[str] = []
    nim_client = _maybe_nim(options, warnings)

    # Load the model once for the whole batch.
    model = None
    if files:
        from .transcribe import _load_model

        model = _load_model(options.model_size, options.device, options.compute_type)

    for media in files:
        file_out = Path(out_dir) / media.stem
        try:
            res = run(str(media), str(file_out), options, model=model, nim_client=nim_client)
            if warnings:
                res.warnings = warnings + res.warnings
            results.append(res)
        except Exception as exc:  # isolate this file's failure
            results.append(
                PipelineResult(
                    source=str(media),
                    output_dir=str(file_out),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    warnings=[traceback.format_exc(limit=2)],
                )
            )

    _write_batch_report(out_dir, results)
    return results


def _write_batch_report(out_dir: str, results: List[PipelineResult]) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    lines = ["# Batch report", ""]
    ok = sum(1 for r in results if r.ok)
    lines.append(f"Processed {len(results)} file(s): {ok} succeeded, {len(results) - ok} failed.")
    lines.append("")
    for r in results:
        status = "ok" if r.ok else "FAILED"
        lines.append(f"## {Path(r.source).name} — {status}")
        if r.ok:
            lines.append(f"- language: {r.language}")
            lines.append(f"- duration: {r.duration:.1f}s")
            lines.append(f"- outputs: {len(r.outputs)} file(s) in `{r.output_dir}`")
        else:
            lines.append(f"- error: {r.error}")
        for w in r.warnings:
            lines.append(f"- warning: {w.strip().splitlines()[0]}")
        lines.append("")
    (root / "batch_report.md").write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
