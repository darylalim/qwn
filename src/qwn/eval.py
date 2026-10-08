"""`qwn eval`: retrieval metrics, intervals, exit criteria and baselines (PLAN.md → Evaluation).

Phase 1 scores retrieval only (`--no-generate`); answer metrics arrive with phase 2. Results never
include document text, only ids, paths, ranks and timings.
"""

import hashlib
import json
import logging
import math
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from qwn.config import Settings
from qwn.index import Chunk, Index, IndexMismatch
from qwn.ingest import HEADING_SEP, Ingester, ingest_hash, open_index
from qwn.models import Registry
from qwn.models_lock import pinned
from qwn.retrieve import Hit, Retriever

logger = logging.getLogger(__name__)

SetName = Literal["public", "private"]
TAGS = ("text", "scan", "chart", "table", "screenshot", "markdown", "exact", "injection")
KS = (1, 5, 10)
BOOTSTRAP_RESAMPLES = 1000

# Settings that change results (paths excluded): part of settings_hash.
RESULT_KEYS = (
    "top_k",
    "hybrid",
    "fts_k",
    "rrf_k",
    "rerank_k",
    "max_images",
    "max_pixels",
    "max_context",
    "max_tokens",
    "pdf_embed",
    "chunk_context",
    "embed_dim",
    "pdf_dpi",
    "chunk_tokens",
)


class EvalError(RuntimeError):
    pass


# queries


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    expected: list[dict[str, Any]]  # [] = unanswerable
    tags: list[str]
    answer_contains: list[str] = field(default_factory=list)
    answer_must_not_contain: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)  # "in" (phase 2)
    region: list[float] | None = None  # phase 6

    @property
    def answerable(self) -> bool:
        return bool(self.expected)


def load_queries(path: Path) -> list[Query]:
    if not path.is_file():
        raise EvalError(f"No eval questions at {path}")
    out: list[Query] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            q = Query(
                id=row["id"],
                query=row["query"],
                expected=row["expected"],
                tags=row.get("tags", []),
                answer_contains=row.get("answer_contains", []),
                answer_must_not_contain=row.get("answer_must_not_contain", []),
                scope=row.get("in", []),
                region=row.get("region"),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise EvalError(f"{path.name} line {n}: {type(e).__name__}: {e}") from None
        if q.id in seen:
            raise EvalError(f"{path.name} line {n}: duplicate id {q.id}")
        if any("path" not in item for item in q.expected):
            raise EvalError(f"{path.name} line {n}: every expected item needs a path")
        seen.add(q.id)
        out.append(q)
    return out


# matching


def resolve(item_path: str, root: Path) -> str:
    p = Path(item_path).expanduser()
    return str(p if p.is_absolute() else (root / p).resolve())


def matches(chunk: Chunk, item: dict[str, Any], root: Path) -> bool:
    if chunk.path != resolve(item["path"], root):
        return False
    if "page" in item and chunk.page != item["page"]:
        return False
    if "heading" in item:
        return chunk.heading_path.split(HEADING_SEP)[-1] == item["heading"]
    return True


def first_rank(hits: list[Hit], expected: list[dict[str, Any]], root: Path) -> int | None:
    """1-based rank of the first hit matching any expected item; None if absent."""
    for rank, hit in enumerate(hits, start=1):
        if any(matches(hit.chunk, item, root) for item in expected):
            return rank
    return None


def missing_paths(queries: list[Query], index: Index, root: Path) -> list[str]:
    indexed = {d.path for d in index.documents()}
    wanted = {resolve(item["path"], root) for q in queries for item in q.expected}
    return sorted(wanted - indexed)


# statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: list[float], *, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return (means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1])


def proportion(hits: list[bool]) -> dict[str, Any]:
    k, n = sum(hits), len(hits)
    lo, hi = wilson(k, n)
    return {"value": k / n if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)], "n": n}


def mean_rr(ranks: list[int | None]) -> dict[str, Any]:
    rr = [0.0 if r is None else 1 / r for r in ranks]
    lo, hi = bootstrap_ci(rr)
    value = sum(rr) / len(rr) if rr else 0.0
    return {"value": value, "ci95": [round(lo, 4), round(hi, 4)], "n": len(rr)}


def wlt(before: list[int | None], after: list[int | None]) -> dict[str, int]:
    """Did `after` move the first correct item up (win), down (loss) or not at all (tie)?"""
    inf = math.inf
    out = {"win": 0, "loss": 0, "tie": 0}
    for b, a in zip(before, after, strict=True):
        b_, a_ = inf if b is None else b, inf if a is None else a
        out["win" if a_ < b_ else "loss" if a_ > b_ else "tie"] += 1
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(p * len(s)) - 1)]


# hashes and metadata


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def corpus_hash(index: Index) -> str:
    meta = index.meta()
    docs = sorted(d.sha256 for d in index.documents())
    return sha256_json([docs, meta.get("embed_revision"), meta.get("ingest_hash")])


def settings_hash(settings: Settings, options: dict[str, bool]) -> str:
    return sha256_json({**{k: getattr(settings, k) for k in RESULT_KEYS}, **options})


def git_sha(home: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=home, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def model_ids(settings: Settings) -> dict[str, str]:
    repos = {
        "gen": settings.gen_model,
        "embed": settings.embed_model,
        "rerank": settings.rerank_model,
        "guard": settings.guard_model,
    }
    return {role: f"{repo}@{pinned(repo).revision}" for role, repo in repos.items()}


COMPARABLE = ("set", "eval_set_hash", "corpus_hash", "models", "settings_hash")


def differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return [key for key in COMPARABLE if a.get(key) != b.get(key)]


# the eval sets


@dataclass(frozen=True)
class EvalSet:
    name: SetName
    root: Path  # corpus root: expected paths are relative to it
    index_root: Path
    queries: Path
    baseline: Path
    disposable: bool  # public: rebuilt automatically


def eval_set(settings: Settings, name: SetName) -> EvalSet:
    home = settings.home
    if name == "public":
        return EvalSet(
            name,
            home / "data" / "eval-public",
            public_index_parent(settings) / ingest_hash(settings)[:8],
            home / "eval" / "public" / "queries.jsonl",
            home / "eval" / "public" / "baseline.json",
            disposable=True,
        )
    return EvalSet(
        name,
        settings.data_dir,
        settings.index_dir,
        home / "eval" / "private" / "queries.jsonl",
        home / "eval" / "private" / "baseline.json",
        disposable=False,
    )


def public_index_parent(settings: Settings) -> Path:
    return settings.index_dir / "eval-public"


def clean_public(settings: Settings) -> list[Path]:
    """Delete public eval indexes built with other ingest settings."""
    parent, keep = public_index_parent(settings), ingest_hash(settings)[:8]
    removed = [p for p in parent.glob("*") if p.is_dir() and p.name != keep]
    for p in removed:
        shutil.rmtree(p)
    return removed


def build_public_corpus(home: Path, root: Path) -> None:
    script = home / "eval" / "public" / "build_corpus.py"
    subprocess.run([sys.executable, str(script), str(root)], check=True, capture_output=True)


def prepare(
    settings: Settings, es: EvalSet, registry: Registry, *, say: Callable[[str], None]
) -> Index:
    """Open the set's index. The public one is (re)built and brought up to date automatically."""
    if not es.disposable:
        return open_index(settings, es.index_root)
    say(f"Building the public corpus in {es.root} …")
    build_public_corpus(settings.home, es.root)
    try:
        index = open_index(settings, es.index_root)
    except IndexMismatch:
        index = open_index(settings, es.index_root, reset=True)
    ingester = Ingester(settings, index, registry.embedder)
    report = ingester.run([es.root])
    if report.indexed:
        say(f"Indexed {len(report.indexed)} public documents.")
    if report.failed:
        raise EvalError(f"Public corpus failed to ingest: {report.failed}")
    ingester.prune(ingester.prune_plan([es.root]).remove)
    return index


# running


@dataclass(frozen=True)
class Options:
    rerank: bool = True
    hybrid: bool = True
    generate: bool = False  # phase 2

    def as_dict(self) -> dict[str, bool]:
        return {"rerank": self.rerank, "hybrid": self.hybrid, "generate": self.generate}


def run(
    settings: Settings,
    es: EvalSet,
    index: Index,
    registry: Registry,
    options: Options,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    queries = load_queries(es.queries)
    if missing := missing_paths(queries, index, es.root):
        listed = "\n  ".join(missing)
        raise EvalError(f"Expected files are not in the index:\n  {listed}")

    retriever = Retriever(settings, index, registry)
    t0 = time.perf_counter()
    registry.embedder()
    if options.rerank:
        registry.reranker()
    load_s = time.perf_counter() - t0

    timings: dict[str, list[float]] = {"embed": [], "search": [], "rerank": []}
    per_query: list[dict[str, Any]] = []
    answerable = [q for q in queries if q.answerable]
    for i, q in enumerate(answerable):
        if progress is not None:
            progress(i, len(answerable))
        t = time.perf_counter()
        qvec = retriever.embed_query(q.query)
        timings["embed"].append(time.perf_counter() - t)
        t = time.perf_counter()
        cand = retriever.candidates(q.query, hybrid=True, query_vec=qvec)
        timings["search"].append(time.perf_counter() - t)
        primary = cand.fused if options.hybrid else cand.vector
        row: dict[str, Any] = {
            "id": q.id,
            "tags": q.tags,
            "rank_vector": first_rank(cand.vector, q.expected, es.root),
            "rank_hybrid": first_rank(cand.fused, q.expected, es.root),
            "rank_embed": first_rank(primary, q.expected, es.root),
        }
        if options.rerank:
            t = time.perf_counter()
            reranked = retriever.rerank(q.query, primary)
            timings["rerank"].append(time.perf_counter() - t)
            row["rank_rerank"] = first_rank(reranked, q.expected, es.root)
        row["rank"] = row["rank_rerank"] if options.rerank else row["rank_embed"]
        per_query.append(row)
    if progress is not None:
        progress(len(answerable), len(answerable))
    for q in queries:
        if not q.answerable:
            per_query.append({"id": q.id, "tags": q.tags, "rank": None})

    meta = {
        "set": es.name,
        "eval_set_hash": hashlib.sha256(es.queries.read_bytes()).hexdigest(),
        "corpus_hash": corpus_hash(index),
        "models": model_ids(settings),
        "settings_hash": settings_hash(settings, options.as_dict()),
        "options": options.as_dict(),
        "git_sha": git_sha(settings.home),
        "qwn_version": version("qwn"),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    scored = [r for r in per_query if "rank_embed" in r]
    result = {
        "meta": meta,
        "summary": summarize(scored, options),
        "by_tag": {
            tag: summarize([r for r in scored if tag in r["tags"]], options)
            for tag in TAGS
            if any(tag in r["tags"] for r in scored)
        },
        "latency": latency(timings, load_s, registry),
        "per_query": per_query,
    }
    result["criteria"] = criteria(result, options)
    return result


def summarize(rows: list[dict[str, Any]], options: Options) -> dict[str, Any]:
    final = [r["rank"] for r in rows]
    out: dict[str, Any] = {}
    for k in KS:
        out[f"recall@{k}"] = proportion([r is not None and r <= k for r in final])
    out["mrr"] = mean_rr(final)
    if options.rerank:
        embed = [r["rank_embed"] for r in rows]
        out["recall@5_before_rerank"] = proportion([r is not None and r <= 5 for r in embed])
        out["mrr_before_rerank"] = mean_rr(embed)
        out["rerank_wlt"] = wlt(embed, final)
    vector, hybrid = [r["rank_vector"] for r in rows], [r["rank_hybrid"] for r in rows]
    out["recall@5_vector"] = proportion([r is not None and r <= 5 for r in vector])
    out["recall@5_hybrid"] = proportion([r is not None and r <= 5 for r in hybrid])
    out["hybrid_wlt"] = wlt(vector, hybrid)
    return out


def latency(timings: dict[str, list[float]], load_s: float, registry: Registry) -> dict[str, Any]:
    out: dict[str, Any] = {
        stage: {"p50_ms": percentile(v, 0.5) * 1000, "p95_ms": percentile(v, 0.95) * 1000}
        for stage, v in timings.items()
        if v
    }
    out["model_load_s"] = load_s
    try:
        out["peak_memory_gb"] = registry.memory_gb()["peak"]
    except Exception:  # no MLX (fakes on other platforms)
        out["peak_memory_gb"] = None
    return out


@dataclass(frozen=True)
class Criterion:
    name: str
    passed: bool
    detail: str
    borderline: bool = False


def _at_least(name: str, metric: dict[str, Any], threshold: float) -> Criterion:
    value, (lo, hi) = metric["value"], metric["ci95"]
    passed = value >= threshold
    detail = f"{value:.3f} (95% CI {lo:.2f}-{hi:.2f}, n={metric['n']}) vs >= {threshold:.2f}"
    return Criterion(name, passed, detail, borderline=passed and lo < threshold)


def criteria(result: dict[str, Any], options: Options) -> list[dict[str, Any]]:
    """Phase 1 exit criteria (PLAN.md → Evaluation → Exit criteria)."""
    s, by_tag = result["summary"], result["by_tag"]
    out = [_at_least("recall@5 overall", s["recall@5"], 0.80)]
    for tag in ("scan", "chart", "table", "exact"):
        if tag in by_tag:
            out.append(_at_least(f"recall@5 {tag}", by_tag[tag]["recall@5"], 0.70))
        else:
            out.append(Criterion(f"recall@5 {tag}", False, "no queries with this tag"))
    if options.rerank:
        w = s["rerank_wlt"]
        out.append(
            Criterion(
                "rerank wins > losses",
                w["win"] > w["loss"],
                f"W/L/T {w['win']}/{w['loss']}/{w['tie']}",
            )
        )
        before, after = s["mrr_before_rerank"]["value"], s["mrr"]["value"]
        out.append(Criterion("rerank keeps MRR", after >= before, f"{before:.3f} -> {after:.3f}"))
    exact = by_tag.get("exact")
    if exact is not None:
        v, h = exact["recall@5_vector"]["value"], exact["recall@5_hybrid"]["value"]
        out.append(Criterion("hybrid > vector on exact", h > v, f"recall@5 {v:.3f} -> {h:.3f}"))
    v, h = s["recall@5_vector"]["value"], s["recall@5_hybrid"]["value"]
    out.append(Criterion("hybrid >= vector overall", h >= v, f"recall@5 {v:.3f} -> {h:.3f}"))
    return [c.__dict__ for c in out]


# baselines and results


def changes(baseline: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Queries whose top-5 hit changed versus the baseline."""
    before = {r["id"]: r.get("rank") for r in baseline.get("per_query", [])}
    lines = []
    for r in result["per_query"]:
        if r["id"] not in before or "rank_embed" not in r:
            continue
        b, a = before[r["id"]], r["rank"]
        hit_b, hit_a = b is not None and b <= 5, a is not None and a <= 5
        if hit_b != hit_a:
            verb = "now in top 5" if hit_a else "dropped out of top 5"
            lines.append(f"{r['id']}: {verb} (rank {b} -> {a})")
    return lines


def write_result(home: Path, result: dict[str, Any]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = home / "eval" / "results" / f"{stamp}-{result['meta']['set']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return out


def load_baseline(es: EvalSet) -> dict[str, Any] | None:
    return json.loads(es.baseline.read_text()) if es.baseline.is_file() else None


def write_baseline(es: EvalSet, result: dict[str, Any]) -> None:
    es.baseline.parent.mkdir(parents=True, exist_ok=True)
    es.baseline.write_text(json.dumps(result, indent=1) + "\n")


# report


def _pct(m: dict[str, Any]) -> str:
    lo, hi = m["ci95"]
    return f"{m['value']:.3f} [{lo:.2f}, {hi:.2f}]"


def format_report(result: dict[str, Any]) -> str:
    s, meta = result["summary"], result["meta"]
    opts = meta["options"]
    lines = [
        f"Eval set: {meta['set']} · rerank {'on' if opts['rerank'] else 'off'} · "
        f"hybrid {'on' if opts['hybrid'] else 'off'} · {s['recall@5']['n']} answerable queries",
        "",
        "Retrieval (value [95% CI])",
    ]
    for key in ("recall@1", "recall@5", "recall@10", "mrr"):
        lines.append(f"  {key:<22} {_pct(s[key])}")
    if "rerank_wlt" in s:
        w = s["rerank_wlt"]
        lines.append(f"  {'recall@5 before rerank':<22} {_pct(s['recall@5_before_rerank'])}")
        lines.append(f"  {'MRR before rerank':<22} {_pct(s['mrr_before_rerank'])}")
        lines.append(f"  {'rerank W/L/T':<22} {w['win']}/{w['loss']}/{w['tie']}")
    h = s["hybrid_wlt"]
    lines.append(f"  {'recall@5 vector only':<22} {_pct(s['recall@5_vector'])}")
    lines.append(f"  {'recall@5 hybrid':<22} {_pct(s['recall@5_hybrid'])}")
    lines.append(f"  {'hybrid W/L/T':<22} {h['win']}/{h['loss']}/{h['tie']}")
    lines += ["", "By tag          n   recall@5   MRR    vector@5  hybrid@5"]
    for tag, t in result["by_tag"].items():
        lines.append(
            f"  {tag:<12} {t['recall@5']['n']:>3}   {t['recall@5']['value']:.3f}     "
            f"{t['mrr']['value']:.3f}  {t['recall@5_vector']['value']:.3f}     "
            f"{t['recall@5_hybrid']['value']:.3f}"
        )
    lat = result["latency"]
    lines += ["", "Latency (models loaded)"]
    for stage in ("embed", "search", "rerank"):
        if stage in lat:
            p50, p95 = lat[stage]["p50_ms"], lat[stage]["p95_ms"]
            lines.append(f"  {stage:<8} p50 {p50:7.1f} ms   p95 {p95:7.1f} ms")
    lines.append(f"  model load {lat['model_load_s']:.1f} s")
    if lat.get("peak_memory_gb") is not None:
        lines.append(f"  MLX peak memory {lat['peak_memory_gb']:.1f} GB")
    lines += ["", "Phase 1 exit criteria"]
    for c in result["criteria"]:
        mark = "✓ pass" if c["passed"] else "✗ FAIL"
        note = "  (borderline: CI lower bound under threshold)" if c["borderline"] else ""
        lines.append(f"  {mark}  {c['name']:<26} {c['detail']}{note}")
    return "\n".join(lines)


def passed(result: dict[str, Any]) -> bool:
    return all(c["passed"] for c in result["criteria"])
