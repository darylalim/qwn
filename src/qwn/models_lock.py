"""Pinned model revisions (repo → commit SHA). Re-pin only with `qwn models pull --update`.

Eval metadata records `repo@sha`, so baselines are comparable only when these match.
"""

from dataclasses import dataclass
from typing import Literal

Group = Literal["core", "voice"]


@dataclass(frozen=True)
class PinnedModel:
    repo: str
    revision: str
    group: Group
    size_gb: float  # expected weights in memory, used by the Registry's memory check


MODELS: dict[str, PinnedModel] = {
    m.repo: m
    for m in [
        # core
        PinnedModel(
            "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "defcdea7cc7a4b0858fea563cbbce171d328e457",
            "core",
            5.8,
        ),
        PinnedModel(
            "mlx-community/Qwen3-VL-Embedding-2B-8bit",
            "b4c9add3544248e763e515c6dd1d1de3ac7d032d",
            "core",
            2.7,
        ),
        PinnedModel(
            "mlx-community/Qwen3-VL-Reranker-2B-8bit",
            "e9cd8bbefa68882550b30cfe1c9905b882965888",
            "core",
            2.7,
        ),
        PinnedModel(
            "mlx-community/Qwen3Guard-Gen-0.6B-MLX",
            "919a45777b667714648ee3d10d4560538b5b8bc8",
            "core",
            1.2,
        ),
        # voice (phase 4)
        PinnedModel(
            "mlx-community/Qwen3-ASR-1.7B-8bit",
            "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
            "voice",
            2.5,
        ),
        PinnedModel(
            "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
            "049ef77fe8816b536193c0c25f9a214d17921282",
            "voice",
            2.9,
        ),
        PinnedModel(
            "mlx-community/silero-vad",
            "7bc17f22d3c0451bd3a6cd71e759b009271ff49a",
            "voice",
            0.002,
        ),
    ]
}


def pinned(repo: str) -> PinnedModel:
    try:
        return MODELS[repo]
    except KeyError:
        raise KeyError(
            f"{repo} is not pinned in qwn/models_lock.py; pin it with `qwn models pull --update`"
        ) from None
