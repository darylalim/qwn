"""ASR: Qwen3-ASR through mlx-audio (`mlx_audio.stt`).

Checked against mlx-audio 0.5.8: `model.generate(audio)` takes a 16 kHz numpy array, decodes
greedily by default (temperature 0) and returns an `STTOutput` whose `text` has the model's
"language X<asr_text>" prefix already removed. The language is auto-detected.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qwn.config import Settings
from qwn.models_lock import pinned

MAX_TOKENS = 1024  # ~2 min of speech is a few hundred tokens


class MlxAudioAsr:
    def __init__(self, model: Any, model_id: str) -> None:
        self.model = model
        self.model_id = model_id

    @classmethod
    def load(cls, settings: Settings) -> "MlxAudioAsr":
        from mlx_audio.stt import load

        from qwn.adapters.hub import pinned_snapshot

        repo = settings.asr_model
        model = load(str(pinned_snapshot(repo)))
        return cls(model, f"{repo}@{pinned(repo).revision}")

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        if audio.size == 0:
            return ""
        out = self.model.generate(
            np.ascontiguousarray(audio, dtype=np.float32),
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            verbose=False,
        )
        return str(out.text).strip()
