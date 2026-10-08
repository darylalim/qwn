"""Deterministic stand-ins for the model interfaces. No MLX, no downloads."""

import hashlib
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from qwn.answer import parse_citations
from qwn.interfaces import Answer, Box, Item, Source, Verdict


def _words(text: str | None) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


class FakeEmbedder:
    """Hash-seeded unit vectors: the same item always maps to the same vector."""

    model_id = "fake/embedder@0"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, items: list[Item], *, is_query: bool) -> NDArray[np.float32]:
        rows = []
        for item in items:
            key = f"{item.text}|{item.image_path}".encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)


class FakeReranker:
    """Score = number of query words that appear in the doc text."""

    model_id = "fake/reranker@0"

    def score(self, query: Item, docs: list[Item]) -> list[float]:
        q = _words(query.text)
        return [float(len(q & _words(d.text))) for d in docs]


class FakeGenerator:
    """Answers from S1 and cites it; abstains when there are no sources."""

    model_id = "fake/generator@0"

    def answer(self, question: str, sources: list[Source], *, greedy: bool = False) -> Answer:
        if not sources:
            text = "I couldn't find this in your documents."
        else:
            text = f"{sources[0].text[:80]} [S1]"
        cited = parse_citations(text, [s.label for s in sources])
        return Answer(text=text, cited=cited, prompt_tokens=len(question), completion_tokens=8)

    def locate(self, image_path: Path, claim: str) -> Box | None:
        return None


class FakeGuard:
    """Keyword rules: 'bomb' is Unsafe, 'hack' is Controversial, anything else Safe."""

    model_id = "fake/guard@0"

    def _verdict(self, text: str, refusal: bool) -> Verdict:
        words = _words(text)
        r = "No" if refusal else None
        if "bomb" in words:
            return Verdict("Unsafe", ["Violent"], r)
        if "hack" in words:
            return Verdict("Controversial", ["Non-violent Illegal Acts"], r)
        return Verdict("Safe", [], r)

    def check_prompt(self, text: str) -> Verdict:
        return self._verdict(text, refusal=False)

    def check_response(self, prompt: str, response: str) -> Verdict:
        return self._verdict(response, refusal=True)
