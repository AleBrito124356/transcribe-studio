# examples

## `sample-podcast.transcript.json`: try everything offline

A short (about 3 min 40 s) episode of an imaginary bilingual show, *Build
Notes*, saved in exactly the format `transcribe-studio` writes
(`transcript.json`). Leo and Marta talk about home-server backups in English,
then switch to Spanish for tomatoes on a balcony. The text is hand-written;
the word timings are synthetic (derived from word length and punctuation) but
shaped like faster-whisper output, with turn gaps between the two speakers.

Every command except `transcribe` accepts a transcript, which skips Whisper
entirely. There is no model download, no API key and no network, and
faster-whisper doesn't even need to be installed:

```bash
python cli.py all examples/sample-podcast.transcript.json --no-nim -o demo/
python cli.py all examples/sample-podcast.transcript.json --no-nim --diarize --speaker-labels -o demo-speakers/
python cli.py subs examples/sample-podcast.transcript.json --max-chars 32 -o demo-subs/
python cli.py chapters examples/sample-podcast.transcript.json --local
python cli.py summarize examples/sample-podcast.transcript.json --local
```

What to look at in `demo/`:

- `captions.srt` / `captions.vtt`: cues start on their first word and end on
  their last, break at sentence ends, and come with a lint line
  (`Captions: 46 cues | 0 overlaps | ...`).
- `chapters.md`: the switch to the Spanish segment is found at 02:02 with a
  Spanish title, and there are three chapters because YouTube needs at least
  three.
- `summary.md`: an offline extractive summary, labelled as such, with the
  action items from both languages ("We need to test a restore...", "Hay que
  regar temprano...").

## Your own media

Drop audio or video files here and run the full pipeline:

```bash
python cli.py all examples/my-episode.mp3 -o results/
```

Media files in this folder (`*.mp3`, `*.mp4`, `*.wav`, `*.m4a`, `*.mkv`, `*.mov`)
are gitignored so the repository stays text-only. Put anything you like here
without it showing up in `git status`. The first real transcription downloads
the Whisper model you pick (`base` is about 145 MB) from Hugging Face.
