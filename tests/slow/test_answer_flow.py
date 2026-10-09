"""Real models end to end: ingest a tiny corpus, then answer greedily, twice, identically."""

from pathlib import Path

import pytest
from pdf_fixture import write_pdf

from qwn.answer import Answerer
from qwn.config import Settings
from qwn.ingest import Ingester, open_index
from qwn.retrieve import Retriever

QUESTIONS = [
    ("How much did Northwind's revenue grow in Q3?", "report.pdf", "12"),
    ("Which plan includes single sign-on?", "pricing.md", "Enterprise"),
    ("What is the Wi-Fi password for the scout hut?", None, None),  # unanswerable
]


@pytest.fixture(scope="module")
def answerer(registry, tmp_path_factory):
    root = tmp_path_factory.mktemp("answer-flow")
    docs = root / "docs"
    write_pdf(
        docs / "report.pdf",
        [["Northwind quarterly report", "Q3 revenue grew 12% year over year."], ["Outlook"]],
    )
    (docs / "pricing.md").write_text(
        "# Pricing\n\n## Starter\n\nFive users.\n\n## Enterprise\n\nSingle sign-on and audit log.\n"
    )
    settings = Settings(index_dir=root / "index")
    index = open_index(settings)
    Ingester(settings, index, registry.embedder).run([docs])
    return Answerer(settings, Retriever(settings, index, registry), registry)


@pytest.mark.parametrize(("question", "file", "expected"), QUESTIONS)
def test_greedy_answers_are_reproducible_and_cited(answerer, question, file, expected):
    a = answerer.ask(question, greedy=True)
    b = answerer.ask(question, greedy=True)
    assert a.text == b.text and a.cited == b.cited
    assert a.invented == []
    if file is None:
        assert a.abstained
        return
    assert not a.abstained
    assert {Path(s.path).name for s, _ in a.cited_sources()} == {file}
    assert expected.lower() in a.text.lower()
