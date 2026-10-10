"""Generator: Qwen3-VL-8B-Instruct through mlx-vlm."""

import logging
from pathlib import Path
from typing import Any

from qwn.answer import parse_box, parse_citations, plan_sources
from qwn.config import Settings
from qwn.interfaces import Answer, Box, Source
from qwn.models_lock import pinned
from qwn.prompts import (
    LOCATE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    image_label,
    locate_text,
    user_text,
)

logger = logging.getLogger(__name__)

# Qwen3-VL-Instruct's recommended sampling (model card / generation_config.json)
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
LOCATE_MAX_TOKENS = 64  # one small JSON object


def messages(question: str, image_sources: list[Source], sources: list[Source]) -> list[dict]:
    content: list[dict[str, Any]] = []
    for s in image_sources:  # each image follows its label, in source order
        content += [{"type": "text", "text": image_label(s)}, {"type": "image"}]
    content.append({"type": "text", "text": user_text(question, sources)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


class MlxVlmGenerator:
    def __init__(self, model: Any, processor: Any, model_id: str, settings: Settings) -> None:
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.max_images = settings.max_images
        self.max_tokens = settings.max_tokens
        self.last_tps = 0.0  # generation tokens/s of the last answer (smoke test)

    @classmethod
    def load(cls, settings: Settings) -> "MlxVlmGenerator":
        from mlx_vlm import load

        from qwn.adapters.hub import pinned_snapshot

        repo = settings.gen_model
        model, processor = load(str(pinned_snapshot(repo)))
        processor.image_processor.max_pixels = settings.max_pixels
        return cls(model, processor, f"{repo}@{pinned(repo).revision}", settings)

    def answer(self, question: str, sources: list[Source], *, greedy: bool = False) -> Answer:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import get_chat_template

        image_sources, included = plan_sources(sources, self.max_images)
        prompt = get_chat_template(
            self.processor,
            messages(question, image_sources, included),
            add_generation_prompt=True,
        )
        result = generate(
            self.model,
            self.processor,
            prompt,
            image=[str(s.image_path) for s in image_sources] or None,
            max_tokens=self.max_tokens,
            verbose=False,
            temperature=0.0 if greedy else TEMPERATURE,  # 0 = argmax; top_p/top_k then unused
            top_p=TOP_P,
            top_k=TOP_K,
        )
        self.last_tps = result.generation_tps
        return Answer(
            text=result.text,
            cited=parse_citations(result.text, [s.label for s in included]),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.generation_tokens,
        )

    def locate(self, image_path: Path, claim: str) -> Box | None:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import get_chat_template

        msgs = [
            {"role": "system", "content": [{"type": "text", "text": LOCATE_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": locate_text(claim)}],
            },
        ]
        prompt = get_chat_template(self.processor, msgs, add_generation_prompt=True)
        result = generate(
            self.model,
            self.processor,
            prompt,
            image=[str(image_path)],
            max_tokens=LOCATE_MAX_TOKENS,
            verbose=False,
            temperature=0.0,  # greedy: the same page and claim always give the same box
        )
        box = parse_box(result.text)  # Qwen3-VL's 0-1000 scale -> Box (0..1)
        if box is None:
            logger.info("locate: no usable box in a %d-char reply", len(result.text))
        return box
