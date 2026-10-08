"""Reranker: Qwen3-VL-Reranker through mlx-vlm, in-process.

mlx-vlm has no documented Python API for reranking, so we call the same function its /v1/rerank
handler uses (`score_documents`): one forward pass per batch, score = sigmoid(logit("yes") -
logit("no")) at the last token. No sampling, so it's deterministic.
"""

from typing import Any

from qwn.config import Settings
from qwn.interfaces import Item
from qwn.models_lock import pinned
from qwn.prompts import RERANK_INSTRUCTION


class MlxVlmReranker:
    def __init__(self, model: Any, processor: Any, model_id: str) -> None:
        self.model = model
        self.processor = processor
        self.model_id = model_id

    @classmethod
    def load(cls, settings: Settings) -> "MlxVlmReranker":
        from mlx_vlm.reranker_loader import load_reranker

        from qwn.adapters.hub import pinned_snapshot

        repo = settings.rerank_model
        model, processor = load_reranker(pinned_snapshot(repo))
        processor.image_processor.max_pixels = settings.max_pixels
        return cls(model, processor, f"{repo}@{pinned(repo).revision}")

    def score(self, query: Item, docs: list[Item]) -> list[float]:
        from mlx_vlm.server.reranking import RerankItem, score_documents

        def item(i: Item) -> RerankItem:
            image = str(i.image_path) if i.image_path is not None else None
            return RerankItem(text=i.text, image=image)

        if not docs:
            return []
        scores, _tokens = score_documents(
            self.model,
            self.processor,
            self.model.config,
            item(query),
            [item(d) for d in docs],
            RERANK_INSTRUCTION,
        )
        return scores
