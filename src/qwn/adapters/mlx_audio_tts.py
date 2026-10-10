"""TTS: Qwen3-TTS (CustomVoice) through mlx-audio (`mlx_audio.tts`).

Checked against mlx-audio 0.5.8: `model.generate(text, voice=speaker, lang_code=...)` yields
`GenerationResult`s whose `audio` is an mx.array at `model.sample_rate` (24 kHz). Each line of the
answer is synthesized on its own, joined with a short pause, then resampled to qwn's 16 kHz.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qwn.config import Settings
from qwn.models_lock import pinned
from qwn.voice import SAMPLE_RATE, resample

SPEAKER = "Ryan"  # an English preset voice of the CustomVoice model
PAUSE_S = 0.35  # between lines


class MlxAudioTts:
    def __init__(self, model: Any, model_id: str) -> None:
        self.model = model
        self.model_id = model_id

    @classmethod
    def load(cls, settings: Settings) -> "MlxAudioTts":
        from mlx_audio.tts import load

        from qwn.adapters.hub import pinned_snapshot

        repo = settings.tts_model
        model = load(str(pinned_snapshot(repo)))
        return cls(model, f"{repo}@{pinned(repo).revision}")

    def synthesize(self, text: str) -> NDArray[np.float32]:
        rate = int(self.model.sample_rate)
        pause = np.zeros(round(PAUSE_S * rate), dtype=np.float32)
        parts: list[NDArray[np.float32]] = []
        for line in (ln.strip() for ln in text.splitlines()):
            if not line:
                continue
            for result in self.model.generate(line, voice=SPEAKER, lang_code="auto"):
                parts.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            parts.append(pause)
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return resample(np.concatenate(parts[:-1]), rate, SAMPLE_RATE)
