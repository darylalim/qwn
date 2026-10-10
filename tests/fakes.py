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
    """Cites the first source sharing a content word (5+ letters) with the question, quoting it;
    abstains when none does. `extra` is appended verbatim (e.g. an invented "[S9]")."""

    model_id = "fake/generator@0"

    def __init__(self, extra: str = "") -> None:
        self.extra = extra
        self.calls: list[list[Source]] = []

    def answer(self, question: str, sources: list[Source], *, greedy: bool = False) -> Answer:
        self.calls.append(sources)
        q = {w for w in _words(question) if len(w) >= 5}
        match = next((s for s in sources if q & _words(s.text)), None)
        if match is None:
            text = "I couldn't find this in your documents."
        else:
            text = f"{match.text[:80]} [{match.label}]"
        text += self.extra
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


class FakeVad:
    """Energy VAD: 32 ms frames louder than `level` RMS are speech; gaps under 0.3 s are joined."""

    model_id = "fake/vad@0"

    def __init__(self, level: float = 0.05) -> None:
        self.level = level
        self.calls = 0

    def speech_segments(self, audio: NDArray[np.float32]) -> list[tuple[float, float]]:
        self.calls += 1
        frame, rate = 512, 16_000
        segments: list[tuple[float, float]] = []
        for i in range(0, audio.size - frame + 1, frame):
            if np.sqrt(np.mean(audio[i : i + frame] ** 2)) < self.level:
                continue
            start, end = i / rate, (i + frame) / rate
            if segments and start - segments[-1][1] < 0.3:
                segments[-1] = (segments[-1][0], end)
            else:
                segments.append((start, end))
        return segments


class FakeAsr:
    """Returns `text` for every chunk; records each chunk's length in seconds."""

    model_id = "fake/asr@0"

    def __init__(self, text: str = "How much did revenue grow?") -> None:
        self.text = text
        self.chunks: list[float] = []

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        self.chunks.append(audio.size / 16_000)
        return self.text


class FakeTts:
    """A 220 Hz tone, 50 ms per character of text, at 16 kHz."""

    model_id = "fake/tts@0"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, text: str) -> NDArray[np.float32]:
        self.texts.append(text)
        t = np.arange(round(len(text) * 0.05 * 16_000)) / 16_000
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def tone_bursts(*parts: tuple[str, float], rate: int = 16_000) -> NDArray[np.float32]:
    """Synthetic audio: ("speech", seconds) is a 440 Hz tone at 0.5, ("silence", s) is quiet."""
    out = []
    for kind, seconds in parts:
        t = np.arange(round(seconds * rate)) / rate
        out.append(0.5 * np.sin(2 * np.pi * 440 * t) if kind == "speech" else 0 * t)
    return np.concatenate(out).astype(np.float32)


GB = 10**9


class FakeMemory:
    """MLX memory counters without MLX."""

    def __init__(self, active_gb=0.0, recommended_gb=26.8):
        self.active = int(active_gb * GB)
        self.recommended = int(recommended_gb * GB)
        self.cleared = 0

    def active_bytes(self):
        return self.active

    def peak_bytes(self):
        return self.active

    def recommended_bytes(self):
        return self.recommended

    def clear_cache(self):
        self.cleared += 1
