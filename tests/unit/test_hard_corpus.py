"""The hard eval set (PLAN.md → Harder tier): its generator and its queries. No models."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from qwn import eval as ev

ROOT = Path(__file__).parents[2]

QUERIES = ROOT / "eval" / "hard" / "queries.jsonl"

# for the paraphrase check: words that carry no content
STOP_WORDS = set(
    """a an the of in on at to for by from with and or is are was were be been do does did how
    what which who when where why much many per it its this that i my me we our you your they
    their there has have had will would can could must should not no any each as into out up
    after before than more most between""".split()  # noqa: SIM905
)


@pytest.fixture(scope="module")
def hard():
    spec = importlib.util.spec_from_file_location(
        "hard_build_corpus", ROOT / "eval" / "hard" / "build_corpus.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(hard, public_corpus, tmp_path_factory):
    """The hard documents and the whole layout (the public corpus is shared, built once)."""
    _, _, public_layout = public_corpus
    out, layout = tmp_path_factory.mktemp("hard"), dict(public_layout)
    files = hard.build_hard(out, layout)
    return out, files, layout


@pytest.fixture(scope="module")
def queries():
    return [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]


def test_generator_is_deterministic(hard, built, tmp_path):
    out, files, _ = built
    again = hard.build_hard(tmp_path)
    assert len(again) == len(files) == 11
    for path in again:
        assert path.read_bytes() == (out / path.relative_to(tmp_path)).read_bytes()


def test_queries_are_valid_and_cover_the_tags(queries):
    loaded = ev.load_queries(QUERIES)
    assert [q.id for q in loaded] == [f"hard-{i:03d}" for i in range(1, len(loaded) + 1)]
    tags = [t for q in loaded for t in q.tags]
    for tag, least in (
        ("version", 10),
        ("degraded", 10),
        ("paraphrase", 10),
        ("multi", 8),
        ("near-miss", 8),
        ("long", 1),
    ):
        assert tags.count(tag) >= least, tag
    assert "injection" not in tags
    for q in loaded:
        assert (q.match == "all") == ("multi" in q.tags) == (len(q.expected) > 1), q.id
        assert ("near-miss" in q.tags) == (not q.answerable), q.id
        if q.answer_approx is not None:
            assert not q.answer_contains and q.region is None, q.id
        elif q.answerable:
            assert q.answer_contains, q.id


def test_every_expected_page_exists_in_the_layout(built, queries):
    _, _, layout = built
    for q in queries:
        for item in q["expected"]:
            assert (item["path"], item.get("page")) in layout, (q["id"], item)


def test_regions_match_the_corpus_generator(hard, built, queries):
    _, _, layout = built
    for q in queries:
        assert q.get("region") == hard.public.find_region(layout, q), q["id"]


def _tokens(text: str) -> set[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())
    words = re.findall(r"[a-z0-9]+(?:[-.:'][a-z0-9]+)*", text)
    words = [w.removesuffix("'s") for w in words]
    words = [
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words
    ]
    return {w for w in words if w not in STOP_WORDS}


def test_paraphrase_questions_share_at_most_one_word_with_their_page(built, queries):
    _, _, layout = built
    for q in queries:
        if "paraphrase" not in q["tags"]:
            continue
        page = {
            word
            for item in q["expected"]
            for text, _ in layout[(item["path"], item.get("page"))]
            for word in _tokens(text)
        }
        shared = _tokens(q["query"]) & page
        assert len(shared) <= 1, (q["id"], sorted(shared))
