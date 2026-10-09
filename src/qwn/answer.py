"""Answering: retrieved hits → labelled sources → Generator → checked, rendered answer.

See PLAN.md → Prompts and parameters. `ask` screens the question and the answer with Guard
(qwn.guard). Nothing here logs question, source or answer text: only lengths, labels, counts and
timings.
"""

import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from qwn.config import Settings
from qwn.guard import Check, Screen
from qwn.ingest import HEADING_SEP
from qwn.interfaces import Answer, Source
from qwn.models import Registry
from qwn.prompts import ABSTAIN_TEXT
from qwn.retrieve import Hit, Retriever

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(S\d+)\]")
_MARKER = re.compile(r"(\s?)\[(S\d+)\]")


def parse_citations(text: str, labels: list[str]) -> list[str]:
    """Labels cited in `text`, in order of first appearance. Invented labels are dropped."""
    cited: list[str] = []
    invented = invented_citations(text, labels)
    for label in _CITATION.findall(text):
        if label in labels and label not in cited:
            cited.append(label)
    if invented:
        logger.warning("dropped invented citations: %s", sorted(set(invented)))
    return cited


def invented_citations(text: str, labels: list[str]) -> list[str]:
    """Every `[S#]` marker in `text` that names no source (repeats counted)."""
    return [label for label in _CITATION.findall(text) if label not in labels]


def strip_citations(text: str) -> str:
    """The answer without its `[S#]` markers (for matching expected strings)."""
    return _MARKER.sub("", text)


def plan_sources(sources: list[Source], max_images: int) -> tuple[list[Source], list[Source]]:
    """(sources that send their image, sources the model may cite), both in label order.

    The top `max_images` sources by score send their image; the rest send text only. A source
    with neither (an image beyond the limit) is left out, so the model can't cite it.
    """
    with_image = sorted(
        (s for s in sources if s.image_path is not None), key=lambda s: s.score, reverse=True
    )[:max_images]
    image_labels = {s.label for s in with_image}
    included = [s for s in sources if s.label in image_labels or s.text]
    return [s for s in sources if s.label in image_labels], included


def to_sources(hits: list[Hit], max_images: int) -> tuple[list[Source], list[Hit]]:
    """Label the hits the model can see S1..Sn, in rank order, and keep each one's hit.

    Hits left out by `plan_sources` (an image beyond `max_images`) get no label, so labels are
    contiguous and every label the model sees is citable.
    """
    raw = [
        Source(
            label=f"S{i}",
            chunk_id=h.chunk.chunk_id,
            path=h.path,
            page=h.page,
            text=h.chunk.text,
            image_path=h.image_path,
            score=h.rerank_score if h.rerank_score is not None else h.score,
        )
        for i, h in enumerate(hits, start=1)
    ]
    _, included = plan_sources(raw, max_images)
    keep = {s.label for s in included}
    kept = [(s, h) for s, h in zip(raw, hits, strict=True) if s.label in keep]
    sources = [
        Source(f"S{i}", s.chunk_id, s.path, s.page, s.text, s.image_path, s.score)
        for i, (s, _) in enumerate(kept, start=1)
    ]
    return sources, [h for _, h in kept]


def _plain(text: str) -> str:
    return text.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'").casefold()


def abstained(text: str, cited: list[str]) -> bool:
    """No citations and the fixed abstention sentence (case-insensitive)."""
    return not cited and _plain(ABSTAIN_TEXT) in _plain(text)


def cite_label(source: Source, hit: Hit | None = None) -> str:
    """`[S#]` as the user sees it: "q3.pdf p.4", "pricing.md > SSO", "chart.png"."""
    name = Path(source.path).name
    if source.page is not None:
        return f"{name} p.{source.page}"
    heading = hit.chunk.heading_path.split(HEADING_SEP)[-1] if hit is not None else ""
    return f"{name}{HEADING_SEP}{heading}" if heading else name


@dataclass(frozen=True)
class Response:
    text: str  # raw model text with [S#] markers
    sources: list[Source]  # what the model was given, labelled S1..Sn
    hits: list[Hit]  # the hit behind each source, same order
    cited: list[str]  # labels cited, first appearance order, invented ones dropped
    invented: list[str]  # [S#] markers naming no source
    abstained: bool
    prompt_tokens: int
    completion_tokens: int
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    checks: list[Check] = field(default_factory=list)  # Guard: prompt, then response

    @property
    def blocked(self) -> Check | None:
        """The Guard check that blocked this question or answer, if any."""
        return next((c for c in self.checks if c.action == "block"), None)

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.action == "warn"]

    def cited_sources(self) -> list[tuple[Source, Hit]]:
        by_label = {s.label: (s, h) for s, h in zip(self.sources, self.hits, strict=True)}
        return [by_label[label] for label in self.cited]

    def rendered(self) -> str:
        """The answer with `[S#]` shown as `[file p.N]`; invented markers removed."""
        labels = {s.label: cite_label(s, h) for s, h in zip(self.sources, self.hits, strict=True)}

        def sub(m: re.Match[str]) -> str:
            label = labels.get(m.group(2))
            return f"{m.group(1)}[{label}]" if label else ""

        return _MARKER.sub(sub, self.text).strip()


class Answerer:
    def __init__(self, settings: Settings, retriever: Retriever, registry: Registry) -> None:
        self.settings = settings
        self.retriever = retriever
        self.registry = registry

    def retrieve(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        rerank: bool = True,
        hybrid: bool | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[Hit]:
        """The `rerank_k` best hits for `question`, timing each stage into `timings`."""
        t = timings if timings is not None else {}
        start = time.perf_counter()
        qvec = self.retriever.embed_query(question)
        t["embed"] = time.perf_counter() - start
        start = time.perf_counter()
        hits = self.retriever.candidates(
            question, hybrid=hybrid, query_vec=qvec, doc_ids=doc_ids
        ).fused
        t["search"] = time.perf_counter() - start
        if rerank:
            start = time.perf_counter()
            hits = self.retriever.rerank(question, hits)
            t["rerank"] = time.perf_counter() - start
        return hits[: self.settings.rerank_k]

    def answer(
        self,
        question: str,
        hits: list[Hit],
        *,
        greedy: bool = False,
        timings: dict[str, float] | None = None,
    ) -> Response:
        t = dict(timings or {})
        sources, kept = to_sources(hits, self.settings.max_images)
        start = time.perf_counter()
        if sources:
            ans = self.registry.generator().answer(question, sources, greedy=greedy)
        else:  # nothing retrieved: no need to ask the model
            ans = Answer(text=ABSTAIN_TEXT, cited=[], prompt_tokens=0, completion_tokens=0)
        t["generate"] = time.perf_counter() - start
        if ans.prompt_tokens > self.settings.max_context:
            logger.warning(
                "prompt over max_context: %d > %d tokens",
                ans.prompt_tokens,
                self.settings.max_context,
            )
        labels = [s.label for s in sources]
        cited = [label for label in ans.cited if label in labels]
        response = Response(
            text=ans.text,
            sources=sources,
            hits=kept,
            cited=cited,
            invented=invented_citations(ans.text, labels),
            abstained=abstained(ans.text, cited),
            prompt_tokens=ans.prompt_tokens,
            completion_tokens=ans.completion_tokens,
            timings=t,
        )
        logger.info(
            "answer: question_len=%d sources=%d cited=%s invented=%d abstained=%s "
            "tokens=%d/%d timings=%s",
            len(question),
            len(sources),
            cited,
            len(response.invented),
            response.abstained,
            ans.prompt_tokens,
            ans.completion_tokens,
            {k: round(v, 2) for k, v in t.items()},
        )
        return response

    def ask(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        greedy: bool = False,
        guard: bool | None = None,
    ) -> Response:
        """Guard the question, retrieve, answer, then guard the answer.

        A blocked question is never retrieved or answered; a blocked answer is withheld (its text
        and sources are dropped). `guard=None` follows `Settings.guard_enabled`.
        """
        timings: dict[str, float] = {}
        screen = None
        if self.settings.guard_enabled if guard is None else guard:
            screen = Screen(self.registry.guard, self.settings.controversial)
        checks: list[Check] = []
        if screen is not None:
            start = time.perf_counter()
            checks.append(screen.prompt(question))
            timings["guard_prompt"] = time.perf_counter() - start
            if checks[-1].action == "block":
                return _withheld(checks, timings)
        hits = self.retrieve(question, doc_ids=doc_ids, timings=timings)
        response = self.answer(question, hits, greedy=greedy, timings=timings)
        if screen is None:
            return response
        start = time.perf_counter()
        checks.append(screen.response(question, strip_citations(response.text)))
        timings = {**response.timings, "guard_response": time.perf_counter() - start}
        if checks[-1].action == "block":
            return _withheld(checks, timings, response)
        return replace(response, checks=checks, timings=timings)


def _withheld(
    checks: list[Check], timings: dict[str, float], response: Response | None = None
) -> Response:
    """A Response that carries Guard's block instead of an answer."""
    return Response(
        text=checks[-1].message(),
        sources=[],
        hits=[],
        cited=[],
        invented=[],
        abstained=False,
        prompt_tokens=response.prompt_tokens if response else 0,
        completion_tokens=response.completion_tokens if response else 0,
        timings=timings,
        checks=checks,
    )
