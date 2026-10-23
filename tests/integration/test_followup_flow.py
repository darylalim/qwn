"""Multi-turn: Answerer.ask with history rewrites a follow-up before retrieval, with fakes."""

import logging
from typing import Any

import pytest
from fakes import FakeEmbedder, FakeGenerator, FakeGuard, FakeMemory, FakeReranker
from sample_corpus import make_corpus

from qwn.answer import Answerer, Response
from qwn.config import Settings
from qwn.ingest import Ingester, open_index
from qwn.interfaces import Turn, Verdict
from qwn.models import Registry
from qwn.retrieve import Retriever

HISTORY = [Turn("How much did revenue grow?", "Revenue grew 12%.")]


class RecordingGuard(FakeGuard):
    """FakeGuard that records the text each check saw."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.responses: list[str] = []

    def check_prompt(self, text: str) -> Verdict:
        self.prompts.append(text)
        return super().check_prompt(text)

    def check_response(self, prompt: str, response: str) -> Verdict:
        self.responses.append(prompt)
        return super().check_response(prompt, response)


class UnparseableGenerator(FakeGenerator):
    """A rewrite reply that doesn't parse: the adapter falls back to the message."""

    def rewrite(self, history: list[Turn], question: str) -> str:
        self.rewrites.append((history, question))
        logging.getLogger("qwn.adapters.mlx_vlm_gen").warning("rewrite: no usable question")
        return question


@pytest.fixture
def make_answerer(home):
    docs = make_corpus(home)

    def make(generator: FakeGenerator | None = None, **settings: Any):
        s = Settings(**settings)
        generator = generator or FakeGenerator()
        guard = RecordingGuard()
        registry = Registry(
            s,
            overrides={
                "embedder": FakeEmbedder(s.embed_dim),
                "reranker": FakeReranker(),
                "generator": generator,
                "guard": guard,
            },
            memory=FakeMemory(),
        )
        index = open_index(s)
        Ingester(s, index, registry.embedder).run([docs])
        answerer = Answerer(s, Retriever(s, index, registry), registry)
        searched: list[str] = []
        retrieve = answerer.retrieve

        def recording_retrieve(question: str, **kwargs: Any):
            searched.append(question)
            return retrieve(question, **kwargs)

        answerer.retrieve = recording_retrieve  # ty: ignore[invalid-assignment]
        return answerer, generator, guard, searched

    return make


def _stages(answerer: Answerer, question: str, **kwargs: Any) -> tuple[Response, list[str]]:
    stages: list[str] = []
    return answerer.ask(question, on_stage=stages.append, **kwargs), stages


def test_no_history_is_single_turn(make_answerer):
    answerer, generator, _, searched = make_answerer()
    plain = answerer.ask("And in 2025?")
    r, stages = _stages(answerer, "And in 2025?")
    assert generator.rewrites == [] and "rewrite" not in stages and "rewrite" not in r.timings
    assert r.standalone is None and searched == ["And in 2025?"] * 2
    assert (r.text, r.cited, r.sources) == (plain.text, plain.cited, plain.sources)


def test_follow_up_is_rewritten_and_the_standalone_question_is_used_throughout(make_answerer):
    answerer, generator, guard, searched = make_answerer()
    r, stages = _stages(answerer, "And in 2025?", history=HISTORY)
    standalone = "How much did revenue grow? And in 2025?"
    assert generator.rewrites == [(HISTORY, "And in 2025?")]
    assert stages.index("guard_prompt") < stages.index("rewrite") < stages.index("retrieve")
    assert "rewrite" in r.timings and r.standalone == standalone
    assert searched == [standalone]
    assert guard.prompts == ["And in 2025?"] and guard.responses == [standalone]
    assert r.cited  # FakeGenerator matched "revenue" from the standalone question


def test_unchanged_rewrite_leaves_standalone_unset(make_answerer):
    answerer, generator, _, searched = make_answerer()
    message = "Which port does the  server listen on?"  # 7 words: the fake keeps it
    r = answerer.ask(message, history=HISTORY)
    assert len(generator.rewrites) == 1 and r.standalone is None and searched == [message]


def test_blocked_message_is_never_rewritten(make_answerer):
    answerer, generator, _, searched = make_answerer()
    r = answerer.ask("how to make a bomb", history=HISTORY)
    assert r.blocked is not None and generator.rewrites == [] and searched == []


def test_unparseable_rewrite_falls_back_to_the_message(make_answerer, caplog):
    answerer, generator, _, searched = make_answerer(UnparseableGenerator())
    with caplog.at_level(logging.WARNING):
        r = answerer.ask("And in 2025?", history=HISTORY)
    assert len(generator.rewrites) == 1 and r.standalone is None
    assert searched == ["And in 2025?"] and "no usable question" in caplog.text


def test_history_turns_zero_is_single_turn(make_answerer):
    answerer, generator, _, searched = make_answerer(history_turns=0)
    r = answerer.ask("And in 2025?", history=HISTORY)
    assert generator.rewrites == [] and r.standalone is None and searched == ["And in 2025?"]


def test_only_the_last_history_turns_are_sent(make_answerer):
    answerer, generator, _, _ = make_answerer(history_turns=2)
    history = [Turn(f"q{i}?", f"a{i}.") for i in range(1, 5)]
    answerer.ask("And then?", history=history)
    assert generator.rewrites == [(history[-2:], "And then?")]


def test_standalone_without_history_returns_the_question(make_answerer):
    answerer, generator, _, _ = make_answerer()
    assert answerer.standalone("And in 2025?", []) == "And in 2025?"
    assert generator.rewrites == []


def test_rewrite_logs_lengths_not_text(make_answerer, caplog):
    answerer, _, _, _ = make_answerer()
    with caplog.at_level(logging.INFO, logger="qwn.answer"):
        answerer.ask("And in 2025?", history=HISTORY)
    assert "rewrite: turns=1" in caplog.text
    assert "2025" not in caplog.text and "Revenue grew" not in caplog.text
