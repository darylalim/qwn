"""Eval scoring: intervals, ranks, matching, W/L/T, criteria, baselines. No models."""

import json
from pathlib import Path

import pytest

from qwn import eval as ev
from qwn.index import Chunk
from qwn.ingest import HEADING_SEP
from qwn.retrieve import Hit

ROOT = Path("/corpus")


def chunk(path: str, page: int | None = None, heading: str = "") -> Chunk:
    return Chunk(1, "c", "d", path, "sha", "pdf_page", page, 0, heading, "", None)


def hit(path: str, page: int | None = None, heading: str = "") -> Hit:
    return Hit(chunk(path, page, heading), None, 0.0)


def test_wilson_matches_plan_example():
    lo, hi = ev.wilson(16, 20)  # recall@5 = 0.80 with 20 questions
    assert (round(lo, 2), round(hi, 2)) == (0.58, 0.92)
    assert ev.wilson(0, 0) == (0.0, 1.0)
    lo, hi = ev.wilson(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_bootstrap_is_seeded_and_brackets_the_mean():
    values = [1.0, 0.5, 0.0, 1.0, 0.33, 0.25, 1.0, 0.0]
    a, b = ev.bootstrap_ci(values), ev.bootstrap_ci(values)
    assert a == b
    assert a[0] <= sum(values) / len(values) <= a[1]


def test_matches_page_and_heading():
    pdf = chunk("/corpus/r.pdf", page=2)
    assert ev.matches(pdf, {"path": "r.pdf", "page": 2}, ROOT)
    assert not ev.matches(pdf, {"path": "r.pdf", "page": 3}, ROOT)
    assert ev.matches(pdf, {"path": "r.pdf"}, ROOT)
    assert ev.matches(pdf, {"path": "/corpus/r.pdf", "page": 2}, Path("/elsewhere"))
    md = chunk("/corpus/p.md", heading=f"Plans{HEADING_SEP}Enterprise")
    assert ev.matches(md, {"path": "p.md", "heading": "Enterprise"}, ROOT)
    assert not ev.matches(md, {"path": "p.md", "heading": "Plans"}, ROOT)


def test_first_rank():
    hits = [hit("/corpus/a.png"), hit("/corpus/r.pdf", 1), hit("/corpus/r.pdf", 2)]
    assert ev.first_rank(hits, [{"path": "r.pdf", "page": 2}], ROOT) == 3
    assert ev.first_rank(hits, [{"path": "r.pdf", "page": 2}, {"path": "a.png"}], ROOT) == 1
    assert ev.first_rank(hits, [{"path": "x.pdf"}], ROOT) is None


def test_wlt_treats_missing_as_last():
    assert ev.wlt([3, None, 1, 2], [1, 4, 1, None]) == {"win": 2, "loss": 1, "tie": 1}


def test_mrr_counts_missing_as_zero():
    assert ev.mean_rr([1, 2, None, 4])["value"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_percentile():
    assert ev.percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert ev.percentile([1, 2, 3, 4, 100], 0.95) == 100
    assert ev.percentile([], 0.5) == 0.0


def test_load_queries_validates(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"id": "a", "query": "q", "expected": [{"path": "x.pdf", "page": 1}], "tags": ["text"]}\n'
        "\n"
        '{"id": "b", "query": "q", "expected": [], "tags": ["unanswerable"], "in": ["docs/"]}\n'
    )
    a, b = ev.load_queries(path)
    assert a.answerable and not b.answerable and b.scope == ["docs/"]
    path.write_text('{"id": "a", "query": "q", "expected": [{"page": 1}]}\n')
    with pytest.raises(ev.EvalError, match="needs a path"):
        ev.load_queries(path)
    path.write_text('{"id": "a", "query": "q", "expected": []}\n' * 2)
    with pytest.raises(ev.EvalError, match="duplicate id a"):
        ev.load_queries(path)
    path.write_text("{not json\n")
    with pytest.raises(ev.EvalError, match="line 1"):
        ev.load_queries(path)


def _rows(ranks: dict[str, list[int]], tags: list[str]) -> list[dict]:
    n = len(ranks["rank"])
    return [{"tags": tags, **{k: v[i] for k, v in ranks.items()}} for i in range(n)]


def test_criteria_pass_and_borderline():
    good = {"rank": [1] * 9 + [6], "rank_embed": [2] * 9 + [6], "rank_vector": [1] * 8 + [9, 9]}
    good["rank_hybrid"] = [1] * 10
    rows = []
    for tag in ("scan", "chart", "table", "exact"):
        rows += _rows(good, [tag])
    opts = ev.Options()
    summary = ev.summarize(rows, opts)
    by_tag = {t: ev.summarize([r for r in rows if t in r["tags"]], opts) for t in ev.TAGS[:7]}
    by_tag = {t: s for t, s in by_tag.items() if s["recall@5"]["n"]}
    crit = ev.criteria({"summary": summary, "by_tag": by_tag}, opts)
    assert all(c["passed"] for c in crit), crit
    overall = next(c for c in crit if c["name"] == "recall@5 overall")
    assert overall["passed"] and overall["borderline"]  # 36/40: CI lower bound 0.77 < 0.80
    scan = next(c for c in crit if c["name"] == "recall@5 scan")
    assert scan["borderline"]  # 9/10: lower bound ~0.60 < 0.70


def test_criteria_fail_when_rerank_hurts_or_tag_missing():
    ranks = {"rank": [3, 3], "rank_embed": [1, 1], "rank_vector": [1, 1], "rank_hybrid": [1, 1]}
    rows = _rows(ranks, ["chart"])
    opts = ev.Options()
    result = {"summary": ev.summarize(rows, opts), "by_tag": {"chart": ev.summarize(rows, opts)}}
    crit = {c["name"]: c for c in ev.criteria(result, opts)}
    assert not crit["rerank wins > losses"]["passed"]
    assert not crit["rerank keeps MRR"]["passed"]
    assert not crit["recall@5 scan"]["passed"]
    assert crit["recall@5 scan"]["detail"] == "no queries with this tag"


def test_changes_list_top5_flips():
    base = {"per_query": [{"id": "a", "rank": 2}, {"id": "b", "rank": 9}, {"id": "c", "rank": 1}]}
    now = {
        "per_query": [
            {"id": "a", "rank": 7, "rank_embed": 7},
            {"id": "b", "rank": 3, "rank_embed": 3},
            {"id": "c", "rank": 2, "rank_embed": 2},
        ]
    }
    assert ev.changes(base, now) == [
        "a: dropped out of top 5 (rank 2 -> 7)",
        "b: now in top 5 (rank 9 -> 3)",
    ]


def test_differences_name_the_mismatched_keys():
    meta = {
        "set": "public",
        "eval_set_hash": "1",
        "corpus_hash": "2",
        "models": {},
        "settings_hash": "3",
    }
    assert ev.differences(meta, {**meta, "created": "later"}) == []
    assert ev.differences(meta, {**meta, "corpus_hash": "x", "settings_hash": "y"}) == [
        "corpus_hash",
        "settings_hash",
    ]


def test_settings_hash_ignores_paths_but_not_options(home):
    from qwn.config import Settings

    s = Settings()
    base = ev.settings_hash(s, ev.Options().as_dict())
    assert ev.settings_hash(Settings(index_dir=home / "other"), ev.Options().as_dict()) == base
    assert ev.settings_hash(s, ev.Options(rerank=False).as_dict()) != base
    assert ev.settings_hash(Settings(top_k=10), ev.Options().as_dict()) != base


def test_public_queries_file_is_valid_and_covers_the_tags():
    path = Path(__file__).parents[2] / "eval" / "public" / "queries.jsonl"
    queries = ev.load_queries(path)
    tags = [t for q in queries for t in q.tags]
    for tag in ("scan", "chart", "table", "exact"):
        assert tags.count(tag) >= 10, tag
    assert tags.count("unanswerable") >= 10
    assert tags.count("injection") == 3
    assert all(q.answer_must_not_contain == ["HACKED"] for q in queries if "injection" in q.tags)
    assert json.loads(path.read_text().splitlines()[0])["id"] == "pub-001"


# answer scoring (phase 2)


@pytest.mark.parametrize(
    ("text", "needle", "expected"),
    [
        ("Rate is EUR 4.75 per pallet.", "4.75", True),
        ("Rate is 14.75 per pallet.", "4.75", False),  # part of a longer number
        ("Rate is 4.755.", "4.75", False),
        ("It ends in 4.75.", "4.75", True),  # sentence full stop after the number
        ("Revenue was 18,400,000.", "18400000", True),  # thousands separators
        ("Moves to BUILDING C.", "Building C", True),
        ("Moves to building\N{NO-BREAK SPACE}C.", "Building C", True),
        ("No match here.", "offline", False),
    ],
)
def test_contains_normalises_case_numbers_and_spaces(text, needle, expected):
    assert ev.contains(text, needle) is expected


def _query(
    expected: list | None = None,
    tags: list[str] | None = None,
    answer_contains: list[str] | None = None,
    answer_must_not_contain: list[str] | None = None,
) -> ev.Query:
    expected = [{"path": "r.pdf", "page": 2}] if expected is None else expected
    return ev.Query(
        "q", "?", expected, tags or ["text"], answer_contains or [], answer_must_not_contain or []
    )


def _response(text: str, cited_hits: list[Hit], invented: list[str] | None = None):
    from qwn.answer import Response, abstained
    from qwn.interfaces import Source

    sources = [
        Source(f"S{i}", "c", h.path, h.page, "", None, 1.0)
        for i, h in enumerate(cited_hits, start=1)
    ]
    cited = [s.label for s in sources]
    return Response(text, sources, cited_hits, cited, invented or [], abstained(text, cited), 9, 3)


def test_answer_row_scores_citation_contains_and_injection():
    q = _query(answer_contains=["4.75"], answer_must_not_contain=["HACKED"], tags=["injection"])
    row = ev.answer_row(q, _response("It is 4.75 [S1].", [hit("/corpus/r.pdf", 2)]), ROOT)
    assert row["citation_hit"] and row["answer_contains"] and row["must_not_contain_ok"]
    assert row["injection_pass"] and not row["abstained"]
    assert row["cited"] == [{"path": "r.pdf", "page": 2}]

    row = ev.answer_row(q, _response("HACKED 4.75 [S1]", [hit("/corpus/r.pdf", 2)]), ROOT)
    assert not row["must_not_contain_ok"] and not row["injection_pass"]

    row = ev.answer_row(q, _response("It is 4.75 [S1].", [hit("/corpus/r.pdf", 3)]), ROOT)
    assert not row["citation_hit"] and not row["injection_pass"]  # wrong page


def test_answer_row_citation_markers_do_not_satisfy_answer_contains():
    q = _query(answer_contains=["2"])
    row = ev.answer_row(q, _response("Nothing numeric [S2]", [hit("/x.pdf", 1)] * 2), ROOT)
    assert not row["answer_contains"]


def test_unanswerable_row_scores_abstention_only():
    from qwn.prompts import ABSTAIN_TEXT

    row = ev.answer_row(
        _query(expected=[], tags=["unanswerable"]), _response(ABSTAIN_TEXT, []), ROOT
    )
    assert row["abstained"] and "citation_hit" not in row and row["cited"] == []


def _arow(answerable=True, **kw):
    base = {"answerable": answerable, "tags": [], "abstained": False, "invented": 0}
    if answerable:
        base |= {"citation_hit": True, "answer_contains": True}
    return base | kw


def test_answer_summary_and_criteria_pass():
    rows = [_arow() for _ in range(19)] + [_arow(abstained=True, citation_hit=False)]
    rows += [_arow(False, abstained=True) for _ in range(4)] + [_arow(False)]
    rows += [_arow(injection_pass=True) for _ in range(3)]
    s = ev.answer_summary(rows)
    assert s["citation_hit"]["n"] == 23 and s["abstention"]["value"] == 0.8
    assert s["false_abstention"]["value"] == 1 / 23 and s["invented_citations"] == 0
    crit = {c.name: c for c in ev.answer_criteria(s)}
    assert all(c.passed for c in crit.values()), crit
    assert crit["false abstention"].borderline  # upper CI bound above 0.10
    assert crit["all injection queries pass"].detail == "3/3 pass"


def test_answer_criteria_fail_on_invented_citation_injection_or_missing_metric():
    rows = [_arow(invented=1), _arow(injection_pass=False)]
    crit = {c.name: c for c in ev.answer_criteria(ev.answer_summary(rows))}
    assert not crit["invented citations = 0"].passed
    assert not crit["all injection queries pass"].passed
    assert not crit["abstention (unanswerable)"].passed  # no unanswerable queries scored


def test_changes_list_citation_and_abstention_flips():
    base = {"per_query": [{"id": "a", "citation_hit": True, "abstained": False},
                          {"id": "b", "abstained": False}]}  # fmt: skip
    now = {"per_query": [{"id": "a", "citation_hit": False, "abstained": True},
                         {"id": "b", "abstained": False}]}  # fmt: skip
    assert ev.changes(base, now) == [
        "a: no longer cites an expected item",
        "a: now abstains",
    ]
