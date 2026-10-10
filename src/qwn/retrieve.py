"""Hybrid retrieval: vector + FTS5 keyword, fused with RRF, then reranked (PLAN.md → Search)."""

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from qwn.config import Settings
from qwn.index import Chunk, Index, fts_query
from qwn.ingest import embed_input
from qwn.interfaces import Item
from qwn.models import Registry

logger = logging.getLogger(__name__)


class EmptyScope(ValueError):
    """`--in` / "Search in" named paths with nothing indexed under them."""

    def __init__(self, paths: list[str]) -> None:
        super().__init__(f"Nothing indexed under {', '.join(paths)}.")


def scope_choices(paths: list[str]) -> list[str]:
    """What "Search in" offers for these indexed files: folders first, then files, each sorted.

    Folders are every folder holding an indexed file, below the folder all files share (picking
    that one would mean "everything", which an empty choice already does).
    """
    files = sorted(set(paths))
    if not files:
        return []
    parents = [Path(f).parent for f in files]
    common = Path(os.path.commonpath([str(p) for p in parents]))
    folders: set[str] = set()
    for parent in parents:
        for folder in (parent, *parent.parents):
            if folder == common or not folder.is_relative_to(common):
                break
            folders.add(str(folder))
    return sorted(folders) + files


def describe_scope(paths: list[str], indexed: set[str]) -> str:
    """ "Searched in: 2 folders, 1 file" (shown on each answer), or "all documents".

    `indexed` is every indexed file path; anything else in `paths` is a folder.
    """
    if not paths:
        return "Searched in: all documents"
    n_files = sum(1 for p in paths if p in indexed)
    n_folders = len(paths) - n_files
    parts = [
        f"{n} {word}{'s' if n != 1 else ''}"
        for n, word in ((n_folders, "folder"), (n_files, "file"))
        if n
    ]
    return "Searched in: " + ", ".join(parts)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    image_path: Path | None  # absolute render path
    score: float  # cosine (vector only) or RRF score (hybrid)
    rerank_score: float | None = None

    @property
    def path(self) -> str:
        return self.chunk.path

    @property
    def page(self) -> int | None:
        return self.chunk.page


def cosine(l2_distance: float) -> float:
    """Vectors are unit length, so |a - b|^2 = 2 - 2 cos."""
    return 1 - l2_distance**2 / 2


def rrf(ranked: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score = sum of 1 / (k + rank) over the lists an id appears in.

    Ranks start at 1. Ties keep the order of first appearance (the vector list comes first).
    """
    scores: dict[int, float] = {}
    for ids in ranked:
        for rank, rowid in enumerate(ids, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def collapse_duplicates(hits: list[Hit]) -> list[Hit]:
    """Keep the best-ranked hit among copies of the same content (same file hash and position)."""
    seen: set[tuple[str, int | None, int]] = set()
    out: list[Hit] = []
    for hit in hits:
        key = (hit.chunk.sha256, hit.chunk.page, hit.chunk.chunk_idx)
        if key not in seen:
            seen.add(key)
            out.append(hit)
    return out


@dataclass(frozen=True)
class Candidates:
    vector: list[Hit]  # nearest first, duplicates collapsed
    keyword: list[Hit]  # best bm25 first ([] when hybrid is off or no usable words)
    fused: list[Hit]  # what goes to the reranker: hybrid RRF, or the vector list


class Retriever:
    def __init__(self, settings: Settings, index: Index, registry: Registry) -> None:
        self.settings = settings
        self.index = index
        self.registry = registry

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return self.registry.embedder().embed([Item(text=query)], is_query=True)[0]

    def _hits(self, scored: list[tuple[int, float]]) -> list[Hit]:
        chunks = self.index.chunks([rowid for rowid, _ in scored])
        hits = [
            Hit(chunks[rowid], self._abs(chunks[rowid].image_path), score)
            for rowid, score in scored
            if rowid in chunks
        ]
        return collapse_duplicates(hits)

    def _abs(self, image_path: str | None) -> Path | None:
        return None if image_path is None else self.index.root / image_path

    def scope(self, paths: list[Path]) -> list[str]:
        """doc_ids of the files and folders in `paths`; EmptyScope if none is indexed."""
        absolute = [str(p.expanduser().resolve()) for p in paths]
        doc_ids = self.index.scope_doc_ids(absolute)
        if not doc_ids:
            raise EmptyScope(absolute)
        return doc_ids

    def vector(self, query_vec: NDArray[np.float32], doc_ids: list[str] | None = None) -> list[Hit]:
        found = self.index.vector_search(query_vec, self.settings.top_k, doc_ids)
        return self._hits([(rowid, cosine(d)) for rowid, d in found])

    def keyword(self, query: str, doc_ids: list[str] | None = None) -> list[Hit]:
        expr = fts_query(query)
        if expr is None:
            return []
        rowids = self.index.keyword_search(expr, self.settings.fts_k, doc_ids)
        return self._hits([(rowid, 0.0) for rowid in rowids])

    def candidates(
        self,
        query: str,
        *,
        hybrid: bool | None = None,
        query_vec: NDArray[np.float32] | None = None,
        doc_ids: list[str] | None = None,
    ) -> Candidates:
        """`doc_ids` (from `scope`) limits both searches to those documents."""
        hybrid = self.settings.hybrid if hybrid is None else hybrid
        vec = self.embed_query(query) if query_vec is None else query_vec
        vector = self.vector(vec, doc_ids)
        if not hybrid:
            return Candidates(vector, [], vector)
        keyword = self.keyword(query, doc_ids)
        fused_ids = rrf(
            [[h.chunk.rowid for h in vector], [h.chunk.rowid for h in keyword]],
            self.settings.rrf_k,
        )
        fused = self._hits(fused_ids)[: self.settings.top_k]
        return Candidates(vector, keyword, fused)

    def rerank_input(self, hit: Hit) -> Item:
        c = hit.chunk
        return embed_input(
            c.kind, c.text, hit.image_path, c.heading_path, Path(c.path).name, self.settings
        )

    def rerank(self, query: str, hits: list[Hit], n: int | None = None) -> list[Hit]:
        """The first `n` hits (default `rerank_candidates`) re-sorted by reranker score (stable
        for ties), followed by the rest in their original order.

        Reranking costs about a second per page image, so only the head of the list is scored.
        """
        n = self.settings.rerank_candidates if n is None else n
        head, tail = hits[:n], hits[n:]
        if not head:
            return tail
        scores = self.registry.reranker().score(
            Item(text=query), [self.rerank_input(h) for h in head]
        )
        scored = [replace(h, rerank_score=s) for h, s in zip(head, scores, strict=True)]
        return sorted(scored, key=lambda h: -(h.rerank_score or 0.0)) + tail

    def search(
        self,
        query: str,
        *,
        hybrid: bool | None = None,
        rerank: bool = True,
        k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        k = self.settings.rerank_k if k is None else k
        hits = self.candidates(query, hybrid=hybrid, doc_ids=doc_ids).fused
        if rerank:
            hits = self.rerank(query, hits)
        logger.info("search: query_len=%d candidates=%d", len(query), len(hits))
        return hits[:k]
