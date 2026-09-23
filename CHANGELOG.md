# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-23

### Added
- **Transcript-first CLI.** `subs`, `summarize`, `chapters` and `all` accept a
  saved `transcript.json` and skip Whisper entirely (faster-whisper need not
  be installed). `all transcript.json` regenerates every artifact.
- **Offline demo:** `examples/sample-podcast.transcript.json`, a 3:40
  bilingual (EN/ES) two-topic episode with word timings.
  `python cli.py all examples/sample-podcast.transcript.json --no-nim -o demo/`
  runs with plain Python and no packages.
- **Offline extractive summary** (`transcribe_studio/extractive.py`): TF-IDF +
  TextRank sentence ranking, bilingual action-item detection. Used when there
  is no key, with `--no-nim`, after a NIM failure, or with `summarize --local`.
  It is labelled as extractive in `summary.md`.
- **Word-timed captions:** cues start on their first word and end on their
  last. Breaks prefer sentence ends, pauses and clause punctuation; cues
  never span a silence of 1 s or more or last longer than 7 s; two-line cues
  are balanced.
- `--speaker-labels` on `subs` and `all`: `[Speaker N]` at each turn in SRT and
  `<v Speaker N>` voice tags in VTT.
- Caption lint (`validate_cues`, `write_captions`): overlaps, reading speed,
  line width. It is printed after writing captions and stored in
  `manifest.json` and `batch_report.md`.
- `normalize_chapters()` applies YouTube's chapter rules (00:00 first, nothing
  past the end, at least 10 s each, unique titles, at least three chapters on
  long enough media) to both NIM and local chapters.
- `--chunk-overlap` (default 2 s), a live transcription progress line on
  terminals, and `--skip-existing` to resume a batch.
- `manifest.json` in every output folder: source name/size, options, outputs,
  warnings and caption lint.
- NIM client retries 408/425/429/5xx, timeouts and dropped connections with
  exponential backoff and jitter, honouring `Retry-After`. `NIM_MAX_RETRIES`
  and `NIM_TIMEOUT` settings.
- CLI `--version`, `--debug`, and distinct exit codes (see the README table).
- `python -m transcribe_studio`.

### Changed
- **Package renamed** from top-level `src` (plus a top-level `cli` module) to
  `transcribe_studio`. The console script is
  `transcribe_studio.cli:main`; the repo-root `cli.py` is kept as a shim, so
  `python cli.py ...` still works. Code importing `src.*` must import
  `transcribe_studio.*` instead.
- Offline chapters use TF-IDF with IDF over the whole recording, accent-folded
  bilingual stopwords and fillers, strongest-first boundary selection, and
  titles from the terms most distinctive of each chapter (original casing,
  never repeated). Detection is linear in the number of segments.
- **Without a NIM key, `summary.md` is now written** (as an offline extractive
  summary) instead of being skipped. Set `PipelineOptions(offline_summary=False)`
  for the old behaviour.
- `summarize` without a key writes the extractive summary (with a note)
  instead of exiting with code 2.
- `transcribe(..., chunk_overlap=...)` now defaults to 2 s and really
  overlaps the windows.
- The language auto-detected in the first chunk is locked for later chunks.
- Batch inputs sharing a stem get separate folders (`ep01`, `ep01-wav`).
- A NIM summary receives `Speaker N:` dialogue when the transcript has speakers.
- WebVTT output escapes `&`, `<` and `>`; SRT turns a literal `-->` into `->`.
- SPDX license metadata; build requires `setuptools>=77`.

### Fixed
- `NIM_BASE_URL` / `NIM_MODEL` in `.env` were ignored, because they were read at
  import time, before the CLI loaded `.env`.
- Any NIM error (bad key, network down, rate limit) aborted `pipeline.run()`
  before `chapters.md` was written, and crashed the CLI with a traceback.
- Chunked transcription with overlap silently dropped real speech (6 of 35
  slots in a 70 s test), because the windows never overlapped.
- Same-stem inputs in a batch overwrote each other's output folder.
- Subtitles ignored word timestamps, so text could appear seconds before it
  was spoken.
- Offline chapters: one recurring word hid real topic changes, and accented
  Spanish stopwords ("también", "está") became titles.
- NIM chapters 3 s apart or past the end of the media were accepted.
- `--diarize` labels were lost from `transcript.json`.
- Invalid WebVTT for text containing `&`, `<` or `-->`.
- The wheel installed generic top-level `src` and `cli` modules.
- Undecodable input, a missing faster-whisper, or a model that could not
  load now give one-line errors with exit codes 7/6 instead of tracebacks.

## [0.1.0] - 2026

Initial release.
