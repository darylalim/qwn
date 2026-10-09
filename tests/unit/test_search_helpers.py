"""fts_query escaping, RRF fusion, duplicate collapse, IDs."""

import sqlite3

import pytest

from qwn.index import Chunk, chunk_id_for, doc_id_for, fts_query, is_identifier
from qwn.retrieve import Hit, collapse_duplicates, cosine, rrf


def test_fts_query_keeps_only_identifiers():
    assert fts_query("What did PO-48213 cost?") == '"PO-48213"'
    assert (
        fts_query("Is SSO in Q3 or AES-256 (E-4031)?")
        == '"SSO" OR "Q3" OR "AES-256" OR "(E-4031)?"'
    )
    assert fts_query("How many days of annual leave do I get?") is None


@pytest.mark.parametrize(
    "word, expected",
    [("PO-48213", True), ("18.4M", True), ("2025", True), ("SSO", True), ("I", False),
     ("A", False), ("What", False), ("well-known", True), ("-", False), ("x", False)],
)  # fmt: skip
def test_is_identifier(word, expected):
    assert is_identifier(word) is expected


@pytest.mark.parametrize(
    "question",
    ['say "HI-2" NEAR(A1 B2)', "FOO-1* -BAR2", "COL:9", "AND OR NOT", '"""', "a ( ) ^", 'X"1'],
)
def test_fts_query_never_breaks_fts5(question):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(text)")
    conn.execute("INSERT INTO t VALUES ('say hi near foo bar col value and or not')")
    expr = fts_query(question)
    if expr is not None:
        conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expr,)).fetchall()


def test_fts_query_doubles_inner_quotes():
    assert fts_query('AB"12') == '"AB""12"'


def test_fts_query_drops_duplicates_and_empty():
    assert fts_query("PO-1 po-1 PO-1") == '"PO-1"'
    assert fts_query("") is None
    assert fts_query("? ! a") is None


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 4]], k=60)
    ids = [rowid for rowid, _ in fused]
    assert ids[0] == 3  # in both lists
    assert ids[1:] == [1, 2, 4]  # ties broken by first appearance
    assert fused[0][1] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_empty():
    assert rrf([[], []], k=60) == []


def _hit(rowid: int, sha: str, page: int | None = 1, idx: int = 0) -> Hit:
    chunk = Chunk(
        rowid, f"c{rowid}", f"d{rowid}", f"/p{rowid}", sha, "pdf_page", page, idx, "", "", None
    )
    return Hit(chunk, None, 1.0 / rowid)


def test_collapse_keeps_first_copy():
    hits = [_hit(1, "a"), _hit(2, "a"), _hit(3, "a", page=2), _hit(4, "b")]
    assert [h.chunk.rowid for h in collapse_duplicates(hits)] == [1, 3, 4]


def test_cosine_from_l2():
    assert cosine(0.0) == 1.0
    assert cosine(2**0.5) == pytest.approx(0.0)


def test_ids_are_stable_and_distinct():
    d = doc_id_for("/a/b.pdf")
    assert d == doc_id_for("/a/b.pdf") and len(d) == 64
    assert chunk_id_for(d, 1, 0) != chunk_id_for(d, 2, 0) != chunk_id_for(d, None, 0)
