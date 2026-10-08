"""qwn's mlx-vlm embedder vs the official model (scripts/make_embedding_reference.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from qwn.interfaces import Item

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = FIXTURES / "embedding_reference.json"
MIN_COSINE = 0.99


@pytest.mark.skipif(not REFERENCE.exists(), reason="run scripts/make_embedding_reference.py")
def test_matches_official_embeddings(registry):
    ref = json.loads(REFERENCE.read_text())
    embedder = registry.embedder()
    for sample, expected in zip(ref["samples"], ref["vectors"], strict=True):
        image = FIXTURES / sample["image"] if sample["image"] else None
        got = embedder.embed(
            [Item(text=sample["text"], image_path=image)], is_query=sample["is_query"]
        )
        cos = float(got[0] @ np.asarray(expected, dtype=np.float32))
        assert cos >= MIN_COSINE, (sample, cos)
