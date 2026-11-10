"""The demo video's spoken question, made by qwn's own TTS (PLAN.md → Demo video).

    uv run python scripts/demo_video/make_question.py --voice serena

Run it on its own, not while `qwn ui` or `qwn chat` runs (it loads the TTS model). It speaks
the voice shot's query in a Qwen3-TTS preset other than the answers' voice, adds a short silence
at each end (the fake microphone starts with the recording; the VAD needs a tail), checks the
transcript with qwn's own ASR, and writes 16 kHz mono 16-bit WAV. TTS samples, so each run
differs a little: the take picked by ear is committed as docs/video/question.wav.
"""

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import timeline  # noqa: E402

LEAD_S, TAIL_S = 0.5, 1.5


def words(text: str) -> set[str]:
    return set(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--voice", required=True, help="a Qwen3-TTS preset, e.g. serena")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "video" / "question.wav")
    args = p.parse_args()

    from qwn.adapters import mlx_audio_tts
    from qwn.config import Settings
    from qwn.models import Registry
    from qwn.voice import SAMPLE_RATE, Voice, encode_wav

    if args.voice.lower() == mlx_audio_tts.SPEAKER.lower():
        sys.exit(f"{args.voice} is the answers' voice: pick another preset")
    tl = timeline.load(ROOT / "docs" / "video" / "timeline.toml")
    shot = next(s for s in tl.shots if s.kind == "voice")
    queries = (ROOT / "eval" / "hard" / "queries.jsonl").read_text().splitlines()
    text = next(q["query"] for q in map(json.loads, queries) if q["id"] == shot.query)

    settings = Settings()
    registry = Registry(settings)
    mlx_audio_tts.SPEAKER = args.voice  # the adapter reads it per call
    speech = registry.tts().synthesize(text)
    pad = [np.zeros(round(s * SAMPLE_RATE), dtype=np.float32) for s in (LEAD_S, TAIL_S)]
    audio = np.concatenate([pad[0], speech, pad[1]])
    heard = Voice(settings, registry).transcribe(audio).text
    if missing := words(text) - words(heard):
        sys.exit(
            f"ASR heard {heard!r}: missing {sorted(missing)}. Run again or pick another voice."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(encode_wav(audio))
    print(f"{args.out}: {audio.size / SAMPLE_RATE:.2f} s ({args.voice}); ASR heard {heard!r}")


if __name__ == "__main__":
    main()
