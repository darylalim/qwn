"""Embedder: Qwen3-VL-Embedding through mlx-vlm, in-process.

mlx-vlm's /v1/embeddings server sends text or an image per input, and raw text without the chat
template. We build the model's own input format instead (system instruction, then image and text
in one user turn, chat template with generation prompt, last-token pooling inside the model), so a
`pdf_page` is embedded as image + text together (PLAN.md → Data flow).
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qwn.config import Settings
from qwn.interfaces import Item
from qwn.models_lock import pinned
from qwn.prompts import EMBED_DOC_INSTRUCTION, EMBED_QUERY_INSTRUCTION


def truncate_and_normalize(vectors: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Matryoshka truncation: keep the first `dim` values, then L2-normalise each row."""
    if vectors.ndim != 2 or vectors.shape[1] < dim:
        raise ValueError(f"cannot truncate shape {vectors.shape} to {dim}")
    out = vectors[:, :dim].astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def messages(item: Item, instruction: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if item.image_path is not None:
        content.append({"type": "image"})
    if item.text is not None:
        content.append({"type": "text", "text": item.text})
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


class MlxVlmEmbedder:
    def __init__(self, model: Any, processor: Any, model_id: str, dim: int) -> None:
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.dim = dim

    @classmethod
    def load(cls, settings: Settings) -> "MlxVlmEmbedder":
        from mlx_vlm.encoder_loader import load_encoder_model
        from mlx_vlm.utils import load_processor

        from qwn.adapters.hub import pinned_snapshot

        repo = settings.embed_model
        path = pinned_snapshot(repo)
        # The checkpoint says model_type "qwen3_vl", and mlx-vlm 0.7.6's embedding loader has no
        # remap for it, so it would build the generation model (logits, no embeddings). Ask for
        # mlx-vlm's Qwen3-VL embedding class (last-token pooling + L2 norm) explicitly.
        model = load_encoder_model(path, model_remapping={"qwen3_vl": "qwen3_vl_embedding"})
        processor = load_processor(path, add_detokenizer=False)
        processor.image_processor.max_pixels = settings.max_pixels
        return cls(model, processor, f"{repo}@{pinned(repo).revision}", settings.embed_dim)

    def embed(self, items: list[Item], *, is_query: bool) -> NDArray[np.float32]:
        instruction = EMBED_QUERY_INSTRUCTION if is_query else EMBED_DOC_INSTRUCTION
        rows = [self._embed_one(item, instruction) for item in items]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return truncate_and_normalize(np.stack(rows), self.dim)

    def _embed_one(self, item: Item, instruction: str) -> NDArray[np.float32]:
        import mlx.core as mx
        from mlx_vlm.prompt_utils import get_chat_template
        from mlx_vlm.utils import load_image, prepare_inputs

        prompt = get_chat_template(
            self.processor, messages(item, instruction), add_generation_prompt=True
        )
        images = [load_image(str(item.image_path))] if item.image_path is not None else None
        # mlx-vlm caches rope position ids on the language model and only resets them for image
        # inputs, so a text input after an image would reuse the image's positions. Reset always.
        self.model.language_model._position_ids = None
        self.model.language_model._rope_deltas = None
        inputs = prepare_inputs(
            self.processor,
            images=images,
            prompts=prompt,
            image_token_index=getattr(self.model.config, "image_token_index", None),
        )
        out = self.model(**inputs)
        mx.eval(out.text_embeds)
        return np.array(out.text_embeds[0].astype(mx.float32), dtype=np.float32)
