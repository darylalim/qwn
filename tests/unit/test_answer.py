"""The answer flow with fakes: source labelling, abstention, citation rendering, logging."""

import logging
from pathlib import Path
from typing import cast

from fakes import FakeGenerator

from qwn.answer import Answerer, abstained, invented_citations, strip_citations, to_sources
from qwn.config import Settings
from qwn.index import Chunk
from qwn.ingest import HEADING_SEP as SEP
from qwn.models import Registry
from qwn.prompts import ABSTAIN_TEXT
from qwn.retrieve import Hit, Retriever


def hit(n: int, *, text="", image=True, page=None, heading="", path=None, rerank=None) -> Hit:
    path = path or f"/docs/doc{n}.pdf"
    kind = "pdf_page" if page else ("image" if image else "text")
    chunk = Chunk(n, f"c{n}", f"d{n}", path, f"sha{n}", kind, page, 0, heading, text, None)
    return Hit(chunk, Path(f"/idx/{n}.webp") if image else None, 1.0 / n, rerank)


def answerer(home, generator=None, **settings) -> tuple[Answerer, FakeGenerator]:
    gen = generator or FakeGenerator()
    s = Settings(**settings)
    registry = Registry(s, overrides={"generator": gen})
    return Answerer(s, cast(Retriever, None), registry), gen  # answer() never retrieves


# labelling


def test_sources_are_labelled_in_rank_order_with_rerank_scores():
    sources, kept = to_sources([hit(1, text="a", rerank=0.9), hit(2, text="b")], max_images=4)
    assert [s.label for s in sources] == ["S1", "S2"]
    assert sources[0].score == 0.9 and sources[1].score == 0.5  # rerank score, else fused
    assert [h.chunk.rowid for h in kept] == [1, 2]


def test_image_beyond_max_images_gets_no_label_and_labels_stay_contiguous():
    hits = [hit(1), hit(2), hit(3, text="page text"), hit(4)]  # 1, 2, 4 are image-only
    sources, kept = to_sources(hits, max_images=2)
    assert [s.label for s in sources] == ["S1", "S2", "S3"]
    assert [h.chunk.rowid for h in kept] == [1, 2, 3]  # hit 4: no image slot, no text


# abstention and citations


def test_abstained_needs_the_sentence_and_no_citations():
    assert abstained(ABSTAIN_TEXT, [])
    assert abstained(
        "Sorry. I COULDN\N{RIGHT SINGLE QUOTATION MARK}T FIND THIS IN YOUR DOCUMENTS.", []
    )
    assert not abstained(ABSTAIN_TEXT, ["S1"])
    assert not abstained("Revenue grew 12%.", [])


def test_invented_citations_counts_every_marker_without_a_source():
    assert invented_citations("a [S1] b [S7] c [S7] [S2]", ["S1", "S2"]) == ["S7", "S7"]


def test_strip_citations():
    assert strip_citations("Grew 12% [S2][S1]. Done [S3].") == "Grew 12%. Done."


# answering


def test_answer_cites_and_renders_file_and_page(home):
    a, _ = answerer(home)
    hits = [
        hit(1, text="Revenue grew 12 percent.", page=4, path="/docs/q3.pdf"),
        hit(2, text="Single sign-on", image=False, heading=f"Pricing{SEP}Enterprise",
            path="/docs/pricing.md"),
    ]  # fmt: skip
    r = a.answer("How much did revenue grow?", hits)
    assert r.cited == ["S1"] and not r.abstained and r.invented == []
    assert r.rendered() == "Revenue grew 12 percent. [q3.pdf p.4]"
    assert [s.label for s, _ in r.cited_sources()] == ["S1"]

    r = a.answer("Which plan has single sign-on?", hits)
    assert r.rendered().endswith(f"[pricing.md{SEP}Enterprise]")
    assert "generate" in r.timings


def test_invented_citations_are_reported_and_removed_from_the_rendering(home):
    a, _ = answerer(home, FakeGenerator(extra=" See also [S9]."))
    r = a.answer("revenue growth?", [hit(1, text="Revenue growth was 12%.", page=1)])
    assert r.invented == ["S9"] and r.cited == ["S1"]
    assert "[S9]" not in r.rendered() and r.rendered().endswith("See also.")


def test_no_hits_abstains_without_calling_the_model(home):
    a, gen = answerer(home)
    r = a.answer("anything at all?", [])
    assert r.abstained and r.text == ABSTAIN_TEXT and gen.calls == []


def test_unrelated_sources_abstain(home):
    a, _ = answerer(home)
    r = a.answer("What is the refund policy?", [hit(1, text="Bar chart of sales", page=1)])
    assert r.abstained and r.cited == [] and r.rendered() == ABSTAIN_TEXT


def test_generator_sees_only_labelled_sources(home):
    a, gen = answerer(home, max_images=1)
    a.answer("chart?", [hit(1), hit(2), hit(3, text="chart notes")])
    assert [s.label for s in gen.calls[0]] == ["S1", "S2"]
    assert gen.calls[0][1].text == "chart notes"


def test_answer_logs_never_contain_question_source_or_answer_text(home, caplog):
    sentinel = "ZEBRAQUARTZSENTINEL"
    a, _ = answerer(home)
    with caplog.at_level(logging.DEBUG):
        r = a.answer(f"What about {sentinel}?", [hit(1, text=f"{sentinel} facts.", page=1)])
    assert sentinel in r.text  # the fake quoted it, so the test would catch a leak
    assert "answer: question_len=" in caplog.text
    assert sentinel not in caplog.text
