"""One-command pipeline: media file -> transcript, subtitles, summary, chapters.

`run` processes a single file into an output folder. `run_batch` walks a
directory, giving every file its own subfolder and isolating failures so one bad
file never sinks the whole run.

NIM is strictly optional. When it is unavailable, rate limited or rejects the
key, the affected step is reported as a warning and every local artifact
(transcript, subtitles, offline chapters) is still written.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import __version__
from . import chapters as chapters_mod
from . import diarize as diarize_mod
from . import subtitles as subs_mod
from . import summarize as summarize_mod
from .config import MEDIA_EXTENSIONS, TranscriptResult, save_json
from .nim import MissingApiKey, NimClient, NimError
from .transcribe import DEFAULT_CHUNK_OVERLAP, segments_to_paragraphs, transcribe

MANIFEST_NAME = "manifest.json"
DIARIZATION_HEURISTIC = "pause-heuristic"

ProgressFn = Callable[[float, Optional[float]], None]


@dataclass
class PipelineOptions:
    """Everything the pipeline needs to know, with sensible defaults."""

    model_size: str = "base"
    language: Optional[str] = None
    device: str = "auto"
    compute_type: str = "int8"
    chunk_length: float = 0.0
    chunk_overlap: float = DEFAULT_CHUNK_OVERLAP
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
    # Set when a NIM call failed during this run. ``nim_fatal`` means retrying
    # will not help (bad key / unknown model), so a batch stops using NIM.
    nim_error: Optional[str] = None
    nim_fatal: bool = False
    # True when --skip-existing found finished outputs and nothing was redone.
    skipped: bool = False


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


def _record_nim_failure(result: PipelineResult, step: str, exc: NimError) -> None:
    if result.nim_error is None:
        result.nim_error = str(exc)
    result.nim_fatal = result.nim_fatal or bool(getattr(exc, "fatal", False))
    result.warnings.append(f"NIM {step} failed: {exc}")


def run(
    media_path: str,
    out_dir: str,
    options: Optional[PipelineOptions] = None,
    model=None,
    nim_client: Optional[NimClient] = None,
    progress: Optional[ProgressFn] = None,
) -> PipelineResult:
    """Process one media file into ``out_dir``.

    ``model`` (a preloaded WhisperModel) and ``nim_client`` can be supplied to
    reuse resources across a batch. When ``nim_client`` is ``None`` it is created
    on demand from the environment. ``progress(done_s, total_s)`` is forwarded
    to the transcriber. A ``manifest.json`` describing the run is written last,
    so its presence means every artifact of this run was written.
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
        chunk_overlap=options.chunk_overlap,
        model=model,
        progress=progress,
    )
    result.language = transcript.language
    result.duration = transcript.duration

    segments = transcript.segments
    diarized_segments = None
    if options.diarize:
        diarized_segments = diarize_mod.assign_speakers(
            segments, gap_threshold=options.diarize_gap
        )
        # Keep the labels in transcript.json too, so anything rebuilt from it
        # later (subs, summaries, chapters) still knows who spoke.
        transcript = replace(
            transcript, segments=diarized_segments, diarization=DIARIZATION_HEURISTIC
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
                    f"Skipped translation to '{options.translate}': NIM is not available."
                )
            else:
                try:
                    translated = subs_mod.translate_segments(
                        segments, options.translate, nim_client
                    )
                except NimError as exc:
                    _record_nim_failure(result, f"translation to '{options.translate}'", exc)
                    nim_client = None  # do not hammer a failing endpoint for the next steps
                else:
                    t_srt = out_path / f"captions.{options.translate}.srt"
                    t_vtt = out_path / f"captions.{options.translate}.vtt"
                    subs_mod.write_srt(translated, t_srt, max_chars=options.max_chars)
                    subs_mod.write_vtt(translated, t_vtt, max_chars=options.max_chars)
                    result.outputs.extend([str(t_srt), str(t_vtt)])

    # 4. Summary (needs NIM).
    if options.make_summary:
        if nim_client is None:
            result.warnings.append("Skipped summary.md: NIM is not available.")
        else:
            language = summarize_mod.resolve_language(
                options.summary_language, transcript.language
            )
            try:
                summary = summarize_mod.summarize_transcript(
                    transcript.text, nim_client, language=language
                )
            except NimError as exc:
                _record_nim_failure(result, "summary", exc)
                nim_client = None
            else:
                summary_file = out_path / "summary.md"
                summary_file.write_text(
                    summarize_mod.to_markdown(summary), encoding="utf-8"
                )
                result.outputs.append(str(summary_file))

    # 5. Chapters (NIM when available, else offline heuristic — never fails).
    if options.make_chapters:
        chapter_list = chapters_mod.detect_chapters(
            segments, nim_client=nim_client, warnings=result.warnings
        )
        chapters_file = out_path / "chapters.md"
        chapters_file.write_text(
            chapters_mod.chapters_to_markdown(chapter_list), encoding="utf-8"
        )
        result.outputs.append(str(chapters_file))

    write_manifest(result, options)
    return result


# ---------------------------------------------------------------------------
# Manifest: what produced this folder (and the resume marker for batches)
# ---------------------------------------------------------------------------
def _source_fingerprint(path: str) -> dict:
    p = Path(path)
    try:
        size: Optional[int] = p.stat().st_size
    except OSError:
        size = None
    return {"name": p.name, "size": size}


def write_manifest(result: PipelineResult, options: PipelineOptions) -> Path:
    """Write ``manifest.json`` into the result folder. Written last on purpose."""
    from dataclasses import asdict

    data = {
        "tool": "transcribe-studio",
        "version": __version__,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": _source_fingerprint(result.source),
        "language": result.language,
        "duration": round(result.duration, 3),
        "outputs": [Path(o).name for o in result.outputs],
        "warnings": [w.strip().splitlines()[0] for w in result.warnings if w.strip()],
        "options": asdict(options),
    }
    path = Path(result.output_dir) / MANIFEST_NAME
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def is_up_to_date(source: str, out_dir: str) -> bool:
    """True if ``out_dir`` holds a finished run of this same source file.

    "Finished" means the manifest (written after every artifact) exists, names
    the same file with the same size, and every output it lists is still there.
    """
    manifest = Path(out_dir) / MANIFEST_NAME
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("source") != _source_fingerprint(source):
        return False
    outputs = data.get("outputs") or []
    return bool(outputs) and all((Path(out_dir) / name).exists() for name in outputs)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def find_media_files(input_dir: str, extensions=MEDIA_EXTENSIONS) -> List[Path]:
    """Return sorted media files directly inside ``input_dir``."""
    root = Path(input_dir)
    files = [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return files


def batch_output_names(files: List[Path]) -> Dict[Path, str]:
    """Give every input its own output folder name.

    The folder is the file's stem; when several inputs share a stem
    (``ep01.mp3`` and ``ep01.wav``) the first keeps it and the others get the
    extension appended (``ep01-wav``). Comparison is case-insensitive because
    Windows and macOS file systems are, and a numeric suffix settles any
    remaining clash. Deterministic for a given sorted file list.
    """
    first_with_stem: Dict[str, Path] = {}
    for f in files:
        first_with_stem.setdefault(f.stem.lower(), f)
    names: Dict[Path, str] = {}
    used: set = set()
    for f in files:
        if first_with_stem[f.stem.lower()] == f:
            name = f.stem
        else:
            name = f"{f.stem}-{f.suffix.lstrip('.').lower()}"
        base, n = name, 2
        while name.lower() in used:
            name = f"{base}-{n}"
            n += 1
        used.add(name.lower())
        names[f] = name
    return names


def run_batch(
    input_dir: str,
    out_dir: str,
    options: Optional[PipelineOptions] = None,
    nim_client: Optional[NimClient] = None,
    skip_existing: bool = False,
    progress: Optional[ProgressFn] = None,
    on_result: Optional[Callable[[int, int, PipelineResult], None]] = None,
) -> List[PipelineResult]:
    """Process every media file in ``input_dir`` with per-file error isolation.

    The whisper model is loaded once (on the first file that needs it) and
    reused. A failure on one file is recorded and the batch continues. If NIM
    rejects the key (or the model does not exist), NIM is switched off for the
    remaining files instead of failing the same way on every one of them.
    With ``skip_existing``, files whose folder already holds a finished run of
    the same input are skipped, so an interrupted batch can simply be re-run.
    ``on_result(index, total, result)`` is called after each file.
    """
    from . import transcribe as transcribe_mod

    options = options or PipelineOptions()
    files = find_media_files(input_dir)
    names = batch_output_names(files)
    results: List[PipelineResult] = []

    warnings: List[str] = []
    if nim_client is None:
        nim_client = _maybe_nim(options, warnings)
    nim_disabled_reason: Optional[str] = None

    model = None
    for index, media in enumerate(files, start=1):
        file_out = Path(out_dir) / names[media]
        if skip_existing and is_up_to_date(str(media), str(file_out)):
            res = PipelineResult(source=str(media), output_dir=str(file_out), skipped=True)
            results.append(res)
            if on_result is not None:
                on_result(index, len(files), res)
            continue
        if model is None:
            # Loaded lazily so a fully up-to-date batch never touches Whisper.
            # A load failure affects every file, so it aborts the batch.
            model = transcribe_mod._load_model(options.model_size, options.device, options.compute_type)
        # With no batch-level client, keep run() from rebuilding one from the
        # environment (and repeating the same "no key" warning per file).
        file_options = options if nim_client is not None else replace(options, use_nim=False)
        try:
            res = run(str(media), str(file_out), file_options, model=model,
                      nim_client=nim_client, progress=progress)
            extra = list(warnings)
            if nim_disabled_reason is not None:
                extra.append(f"NIM disabled for the rest of the batch: {nim_disabled_reason}")
            res.warnings = extra + res.warnings
            if res.nim_fatal and nim_client is not None:
                nim_client = None
                nim_disabled_reason = res.nim_error
        except Exception as exc:  # isolate this file's failure
            res = PipelineResult(
                source=str(media),
                output_dir=str(file_out),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                warnings=[traceback.format_exc(limit=2)],
            )
        results.append(res)
        if on_result is not None:
            on_result(index, len(files), res)

    _write_batch_report(out_dir, results)
    return results


def _write_batch_report(out_dir: str, results: List[PipelineResult]) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    lines = ["# Batch report", ""]
    ok = sum(1 for r in results if r.ok)
    skipped = sum(1 for r in results if r.skipped)
    summary = f"Processed {len(results)} file(s): {ok} succeeded, {len(results) - ok} failed."
    if skipped:
        summary += f" {skipped} already up to date (skipped)."
    lines.append(summary)
    lines.append("")
    for r in results:
        status = "skipped (up to date)" if r.skipped else ("ok" if r.ok else "FAILED")
        lines.append(f"## {Path(r.source).name} — {status}")
        if r.skipped:
            lines.append(f"- outputs kept in `{r.output_dir}`")
        elif r.ok:
            lines.append(f"- language: {r.language}")
            lines.append(f"- duration: {r.duration:.1f}s")
            lines.append(f"- outputs: {len(r.outputs)} file(s) in `{r.output_dir}`")
        else:
            lines.append(f"- error: {r.error}")
        for w in r.warnings:
            lines.append(f"- warning: {w.strip().splitlines()[0]}")
        lines.append("")
    (root / "batch_report.md").write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
