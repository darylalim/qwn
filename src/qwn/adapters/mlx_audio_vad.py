"""VAD: Silero VAD through mlx-audio (`mlx_audio.vad`), on 16 kHz numpy audio.

Checked against mlx-audio 0.5.8: `get_speech_timestamps` accepts numpy arrays, takes `threshold`,
`min_speech_duration_ms`, `min_silence_duration_ms` and `speech_pad_ms`, and returns
`[{"start", "end"}]` (seconds with `return_seconds=True`). Padding is qwn.voice's job, so it's 0.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qwn.config import Settings
from qwn.models_lock import pinned

SAMPLE_RATE = 16_000


class MlxAudioVad:
    def __init__(self, model: Any, settings: Settings, model_id: str) -> None:
        self.model = model
        self.settings = settings
        self.model_id = model_id

    @classmethod
    def load(cls, settings: Settings) -> "MlxAudioVad":
        from mlx_audio.vad import load

        from qwn.adapters.hub import pinned_snapshot

        repo = settings.vad_model
        model = load(pinned_snapshot(repo))
        return cls(model, settings, f"{repo}@{pinned(repo).revision}")

    def speech_segments(self, audio: NDArray[np.float32]) -> list[tuple[float, float]]:
        if audio.size == 0:
            return []
        s = self.settings
        found = self.model.get_speech_timestamps(
            np.ascontiguousarray(audio, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            threshold=s.vad_threshold,
            min_speech_duration_ms=s.vad_min_speech_ms,
            min_silence_duration_ms=s.vad_min_silence_ms,
            speech_pad_ms=0,
            return_seconds=True,
        )
        return sorted((float(t["start"]), float(t["end"])) for t in found)
