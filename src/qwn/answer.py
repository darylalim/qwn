"""Answering: retrieved hits → labelled sources → Generator → checked, rendered answer.

See PLAN.md → Prompts and parameters. `ask` screens the question and the answer with Guard
(qwn.guard). With history, a follow-up is first rewritten into a standalone question (PLAN.md →
Evaluation → Multi-turn questions). `locate` finds the part of a cited page that supports the
answer (phase 6). Nothing here logs question, source or answer text: only lengths, labels, counts
and timings.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from qwn.config import Settings
from qwn.guard import Check, Screen
from qwn.ingest import HEADING_SEP
from qwn.interfaces import Answer, Box, Source, Turn
from qwn.models import Registry
from qwn.prompts import ABSTAIN_TEXT
from qwn.retrieve import Hit, Retriever

logger = logging.getLogger(__name__)

Stage = Literal["guard_prompt", "rewrite", "retrieve", "rerank", "generate", "guard_response"]
OnStage = Callable[[Stage], None]

_CITATION = re.compile(r"\[(S\d+)\]")
_MARKER = re.compile(r"(\s?)\[(S\d+)\]")
# a sentence ends at . ! or ? plus whitespace (unless a citation follows), or at a line break
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?!\[S\d+\])|\n+")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

_LABEL = re.compile(r"^(?:standalone question|question|rewritten)\s*:\s*", re.IGNORECASE)
_QUOTES = "\"'`\N{LEFT DOUBLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}"
_WORD = re.compile(r"\w+")
# words a follow-up uses to point back at the conversation, not at what it asks about
_FILLER = frozenset({
    "a", "about", "again", "also", "an", "and", "any", "are", "as", "at", "be", "been", "but",
    "by", "can", "could", "did", "do", "does", "else", "for", "from", "going", "he", "her",
    "hers", "him", "his", "how", "i", "if", "in", "is", "it", "its", "me", "more", "my", "no",
    "not", "now", "of", "on", "one", "or", "other", "our", "please", "same", "she", "so", "some",
    "tell", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "too", "us", "was", "we", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "will", "with", "would", "yes", "you", "your",
})  # fmt: skip
STEM_CHARS = 5  # "inspected" keeps "inspect", "enrolled" keeps "enrol"

MAX_BOX_AREA = 0.6  # a box covering more of the page than this highlights nothing useful
MAX_STANDALONE_CHARS = 400  # a longer rewrite is the model answering, not rewriting


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


def claim_for(text: str, label: str) -> str:
    """The answer sentences that cite `label`, joined, without markers or list bullets: what that
    source's page should show."""
    marker = f"[{label}]"
    sentences = [s for s in _SENTENCE_END.split(text) if marker in s]
    return " ".join(strip_citations(s).strip().lstrip("-*• ").strip() for s in sentences)


def parse_box(text: str) -> Box | None:
    """The one box in a locate reply (`{"bbox_2d": [x1, y1, x2, y2]}`, 0-1000 scale), or None.

    The reply is untrusted (the page may carry injected text), so anything but a lone object (or
    a one-item list) with four numbers, inside the page, with non-zero area and covering at most
    `MAX_BOX_AREA` of it, is rejected. The worst case is a wrong outline, never an action.
    """
    body = text.strip()
    if m := _FENCE.match(body):
        body = m.group(1)
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if isinstance(data, list):
        data = data[0] if len(data) == 1 else None
    coords = data.get("bbox_2d") if isinstance(data, dict) else None
    if not (
        isinstance(coords, list)
        and len(coords) == 4
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in coords)
    ):
        return None
    x0, y0, x1, y1 = (v / 1000 for v in coords)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):  # also rejects NaN
        return None
    if (x1 - x0) * (y1 - y0) > MAX_BOX_AREA:
        return None
    return Box(x0, y0, x1, y1)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _content_words(text: str) -> set[str]:
    words = (w for w in _WORD.findall(text.lower()) if w not in _FILLER)
    return {w[:STEM_CHARS] for w in words if len(w) > 1 or w.isdigit()}


def keeps_message(standalone: str, question: str) -> bool:
    """Whether a rewrite keeps at least one of the message's own words (by `STEM_CHARS` prefix).

    A rewrite that drops all of them usually repeats an earlier question ("Sprint 14 velocity?"
    after a revenue question came back as the revenue question). A message of filler words only
    ("and then?") has nothing to check, so any rewrite keeps it.
    """
    asked = _content_words(question)
    return not asked or bool(asked & _content_words(standalone))


def parse_standalone(text: str, question: str) -> str | None:
    """The standalone question in a rewrite reply, or None (the caller falls back to `question`).

    The first non-empty line, without surrounding quotes or backticks or a leading label, with
    whitespace collapsed. Empty, over `MAX_STANDALONE_CHARS`, an abstention or a question that
    drops every word of the message (`keeps_message`) is rejected: the reply is untrusted, and at
    worst it changes what is searched for.
    """
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    line = _LABEL.sub("", line.strip().strip(_QUOTES).strip())
    line = _collapse(line.strip(_QUOTES))
    if not line or len(line) > MAX_STANDALONE_CHARS or abstained(line):
        return None
    if not keeps_message(line, question):
        return None
    return line


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


def abstained(text: str) -> bool:
    """The fixed abstention sentence (case-insensitive), with or without citations: a refusal may
    first say what the sources do say (PLAN.md → Evaluation → Explained abstention)."""
    return _plain(ABSTAIN_TEXT) in _plain(text)


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
    standalone: str | None = None  # the question searched and answered, if it isn't the message

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
        on_stage: OnStage | None = None,
    ) -> list[Hit]:
        """The `rerank_k` best hits for `question`, timing each stage into `timings`."""
        t = timings if timings is not None else {}
        stage = on_stage or _ignore
        stage("retrieve")
        start = time.perf_counter()
        qvec = self.retriever.embed_query(question)
        t["embed"] = time.perf_counter() - start
        start = time.perf_counter()
        hits = self.retriever.candidates(
            question, hybrid=hybrid, query_vec=qvec, doc_ids=doc_ids
        ).fused
        t["search"] = time.perf_counter() - start
        if rerank:
            stage("rerank")
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
        on_stage: OnStage | None = None,
    ) -> Response:
        t = dict(timings or {})
        (on_stage or _ignore)("generate")
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
            abstained=abstained(ans.text),
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

    def locate(self, response: Response, label: str) -> Box | None:
        """Where cited source `label`'s page supports what the answer says about it: one extra
        greedy generation. None for text sources, uncited labels, or when the model can't say."""
        source = next((s for s in response.sources if s.label == label), None)
        if source is None or source.image_path is None or label not in response.cited:
            return None
        claim = claim_for(response.text, label)
        if not claim:
            return None
        start = time.perf_counter()
        box = self.registry.generator().locate(source.image_path, claim)
        logger.info(
            "locate: source=%s claim_len=%d found=%s seconds=%.2f",
            label,
            len(claim),
            box is not None,
            time.perf_counter() - start,
        )
        return box

    def standalone(self, question: str, history: list[Turn] | tuple[Turn, ...]) -> str:
        """`question` as one standalone question given the earlier `history` (oldest first).

        No history: `question` itself, with no model call. The generator returns `question` when
        its reply doesn't parse (and logs a warning).
        """
        if not history:
            return question
        start = time.perf_counter()
        standalone = self.registry.generator().rewrite(list(history), question)
        logger.info(
            "rewrite: turns=%d question_len=%d standalone_len=%d changed=%s seconds=%.2f",
            len(history),
            len(question),
            len(standalone),
            _collapse(standalone) != _collapse(question),
            time.perf_counter() - start,
        )
        return standalone

    def ask(
        self,
        question: str,
        *,
        history: list[Turn] | tuple[Turn, ...] = (),
        doc_ids: list[str] | None = None,
        greedy: bool = False,
        guard: bool | None = None,
        on_stage: OnStage | None = None,
    ) -> Response:
        """Guard the question, rewrite a follow-up, retrieve, answer, then guard the answer.

        A blocked question is never rewritten, retrieved or answered; a blocked answer is withheld
        (its text and sources are dropped). With `history` (earlier turns, oldest first), the
        message is rewritten into a standalone question first, and that question is retrieved,
        answered and checked; the generator never sees the history. `guard=None` follows
        `Settings.guard_enabled`. `on_stage` is called as each stage starts (the UI shows it in
        `st.status`).
        """
        stage = on_stage or _ignore
        timings: dict[str, float] = {}
        screen = None
        if self.settings.guard_enabled if guard is None else guard:
            screen = Screen(self.registry.guard, self.settings.controversial)
        checks: list[Check] = []
        if screen is not None:
            stage("guard_prompt")
            start = time.perf_counter()
            checks.append(screen.prompt(question))
            timings["guard_prompt"] = time.perf_counter() - start
            if checks[-1].action == "block":
                return _withheld(checks, timings)
        searched = question
        history = history[-self.settings.history_turns :] if self.settings.history_turns else ()
        if history:
            stage("rewrite")
            start = time.perf_counter()
            searched = self.standalone(question, history)
            timings["rewrite"] = time.perf_counter() - start
        standalone = searched if _collapse(searched) != _collapse(question) else None
        hits = self.retrieve(searched, doc_ids=doc_ids, timings=timings, on_stage=stage)
        response = self.answer(searched, hits, greedy=greedy, timings=timings, on_stage=stage)
        response = replace(response, standalone=standalone)
        if screen is None:
            return response
        stage("guard_response")
        start = time.perf_counter()
        checks.append(screen.response(searched, strip_citations(response.text)))
        timings = {**response.timings, "guard_response": time.perf_counter() - start}
        if checks[-1].action == "block":
            return replace(_withheld(checks, timings, response), standalone=standalone)
        return replace(response, checks=checks, timings=timings)


def _ignore(stage: Stage) -> None:
    pass


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
