"""Model interfaces. The rest of qwn depends only on these; adapters live in qwn.adapters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Item:  # one thing to embed or rerank
    text: str | None = None
    image_path: Path | None = None  # at least one of text/image_path is set

    def __post_init__(self) -> None:
        if self.text is None and self.image_path is None:
            raise ValueError("Item needs text, image_path or both")


@dataclass(frozen=True)
class Source:  # a retrieved, reranked chunk handed to the generator
    label: str  # "S1".."S5"
    chunk_id: str
    path: str
    page: int | None  # 1-based for PDFs, None otherwise
    text: str
    image_path: Path | None
    score: float


@dataclass(frozen=True)
class Answer:
    text: str  # raw model text containing [S#] markers
    cited: list[str]  # labels actually cited, in order of first appearance
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Verdict:
    label: Literal["Safe", "Controversial", "Unsafe"] | None  # None = unparseable
    categories: list[str] = field(default_factory=list)
    refusal: Literal["Yes", "No"] | None = None  # response checks only


@dataclass(frozen=True)
class Box:  # phase 6; relative coordinates, 0..1, origin top-left
    x0: float
    y0: float
    x1: float
    y1: float


class Embedder(Protocol):
    model_id: str
    dim: int  # output dim after truncation

    def embed(self, items: list[Item], *, is_query: bool) -> NDArray[np.float32]: ...

    # returns shape (len(items), dim), each row L2-normalised


class Reranker(Protocol):
    model_id: str

    def score(self, query: Item, docs: list[Item]) -> list[float]: ...

    # one score per doc; higher = more relevant; deterministic


class Generator(Protocol):
    model_id: str

    def answer(self, question: str, sources: list[Source], *, greedy: bool = False) -> Answer: ...

    # greedy=True: temperature 0, for reproducible eval runs

    def locate(self, image_path: Path, claim: str) -> Box | None: ...

    # phase 6: region of the page that supports `claim`; None if the model can't say (greedy)


class Guard(Protocol):
    model_id: str

    def check_prompt(self, text: str) -> Verdict: ...

    def check_response(self, prompt: str, response: str) -> Verdict: ...


# Phase 4 (voice). Audio is mono float32 at 16 kHz everywhere inside qwn.
class Vad(Protocol):
    model_id: str

    def speech_segments(self, audio: NDArray[np.float32]) -> list[tuple[float, float]]: ...

    # (start_s, end_s) of detected speech, sorted, non-overlapping; [] = no speech
