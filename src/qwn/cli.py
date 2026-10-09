"""`qwn` command line. A thin layer over qwn.*: no model or retrieval logic of its own."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from qwn.config import HomeNotFound, Settings
from qwn.models import InsufficientMemory, ModelsInUse, Registry
from qwn.models_lock import MODELS, PinnedModel

if TYPE_CHECKING:
    from qwn.ingest import Failure, PrunePlan
    from qwn.interfaces import Source
    from qwn.retrieve import Hit, Retriever

# Offline unless a command opts in (only `qwn models pull`). huggingface_hub reads this once, at
# import, so it's set before anything imports it.
os.environ["HF_HUB_OFFLINE"] = "1"

app = typer.Typer(help="Private multimodal RAG over your documents, on-device with MLX.")
models_app = typer.Typer(help="Download and inspect the pinned models.")
app.add_typer(models_app, name="models")

LOCK_FILE = Path(__file__).with_name("models_lock.py")


@app.callback()
def main(ctx: typer.Context) -> None:
    from qwn import logging_setup

    try:
        ctx.obj = Settings()
    except HomeNotFound as e:
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(1) from None
    logging_setup.configure(ctx.obj.log_dir)


def make_registry(settings: Settings) -> Registry:
    """The one place the CLI gets models from; tests replace it to inject fakes."""
    return Registry(settings)


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg="red", err=True)
    return typer.Exit(1)


# errors that stop a command with a message instead of a traceback
EXPECTED_ERRORS: tuple[type[Exception], ...] = (ModelsInUse, InsufficientMemory)


def _selected(voice: bool) -> list[PinnedModel]:
    return [m for m in MODELS.values() if m.group == "core" or voice]


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


@models_app.command("pull")
def models_pull(
    voice: Annotated[bool, typer.Option(help="Also pull the phase 4 voice models.")] = False,
    update: Annotated[
        bool, typer.Option(help="Re-pin to each repo's latest revision before pulling.")
    ] = False,
) -> None:
    """Download the pinned models (the only command that uses the network)."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    from qwn.adapters import hub

    selected = _selected(voice)
    if update:
        selected = _repin(selected, hub.latest_revision)
    total = 0
    for m in selected:
        typer.echo(f"pulling {m.repo}@{m.revision[:7]} …")
        result = hub.pull(m.repo, m.revision)
        total += result.bytes
        typer.echo(f"  ok: {result.files} files, {_gb(result.bytes)}")
    typer.secho(f"Done: {len(selected)} models, {_gb(total)}. qwn now runs offline.", fg="green")


def _repin(selected: list[PinnedModel], latest: Callable[[str], str]) -> list[PinnedModel]:
    source = LOCK_FILE.read_text()
    out: list[PinnedModel] = []
    for m in selected:
        sha = latest(m.repo)
        if sha != m.revision:
            typer.echo(f"re-pinning {m.repo}: {m.revision[:7]} → {sha[:7]}")
            source = source.replace(m.revision, sha)
            m = PinnedModel(m.repo, sha, m.group, m.size_gb)
        out.append(m)
    LOCK_FILE.write_text(source)
    return out


@models_app.command("status")
def models_status(ctx: typer.Context) -> None:
    """Show each pinned model, its revision and whether it's downloaded."""
    from qwn.adapters import hub

    settings: Settings = ctx.obj
    configured = {
        settings.gen_model,
        settings.embed_model,
        settings.rerank_model,
        settings.guard_model,
    }
    for repo in sorted(configured - MODELS.keys()):
        typer.secho(f"not pinned: {repo} (set in config)", fg="red")
    for m in MODELS.values():
        try:
            size = _gb(hub.size_on_disk(hub.local_snapshot(m.repo, m.revision)))
            state = typer.style(f"✓ downloaded ({size})", fg="green")
        except hub.ModelNotDownloaded:
            cmd = "qwn models pull" + (" --voice" if m.group == "voice" else "")
            state = typer.style(f"✗ missing: run `{cmd}`", fg="red")
        typer.echo(f"{m.group:5}  {m.repo}@{m.revision[:7]}  {state}")


# ingest / search / status


@app.command()
def ingest(
    ctx: typer.Context,
    paths: Annotated[list[Path] | None, typer.Argument(help="Files or folders to index.")] = None,
    reindex: Annotated[
        bool, typer.Option(help="Drop the whole index and rebuild it from PATHs.")
    ] = False,
    prune: Annotated[
        bool, typer.Option(help="Remove indexed files under PATHs (or anywhere) that are gone.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before pruning.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Only show what would change.")] = False,
) -> None:
    """Index PDFs, images and markdown/text files (unchanged files are skipped)."""
    from qwn.index import IndexBusy, IndexMismatch
    from qwn.ingest import Ingester, ModelUnavailable, collect, open_index

    settings: Settings = ctx.obj
    paths = paths or []
    if not paths and not prune:
        raise _fail("Give at least one PATH (or --prune alone to check the whole index).")
    try:
        index = open_index(settings, reset=reindex and not dry_run)
    except IndexMismatch as e:
        raise _fail(str(e)) from None
    registry = make_registry(settings)
    ingester = Ingester(settings, index, registry.embedder)

    if dry_run:
        files, failed, _ignored = collect(paths)
        planned = ingester.plan(files)
        by_action = {a: [p for p in planned if p.action == a] for a in ("index", "move")}
        if reindex:
            by_action["index"] = planned
        typer.echo(
            f"Would index {len(by_action['index'])}, move {len(by_action['move'])}, "
            f"skip {len(planned) - len(by_action['index']) - len(by_action['move'])} unchanged."
        )
        for p in by_action["index"][:10]:
            typer.echo(f"  index  {p.path}")
        for p in by_action["move"][:10]:
            typer.echo(f"  move   {p.moved_from.path if p.moved_from else '?'} -> {p.path}")
        _print_failures(failed)
        if prune:
            _print_prune(ingester.prune_plan(paths), verb="Would remove")
        return

    def progress(i: int, n: int, path: Path) -> None:
        if i < n:
            typer.echo(f"[{i + 1}/{n}] {path.name}", err=True)

    try:
        report = ingester.run(paths, progress=progress) if paths else None
    except (ModelUnavailable, *EXPECTED_ERRORS) as e:
        raise _fail(str(e)) from None
    if report is not None:
        typer.secho(
            f"Indexed {len(report.indexed)}, unchanged {report.unchanged}, "
            f"moved {len(report.moved)}, failed {len(report.failed)}"
            + (f", ignored {report.ignored} unsupported" if report.ignored else "")
            + ".",
            fg="green" if not report.failed else "yellow",
        )
        _print_failures(report.failed)
    if prune:
        plan = ingester.prune_plan(paths)
        _print_prune(plan, verb="Will remove")
        if plan.remove and (yes or typer.confirm("Remove them from the index?")):
            try:
                ingester.prune(plan.remove)
            except IndexBusy as e:
                raise _fail(str(e)) from None
            typer.secho(f"Removed {len(plan.remove)} documents.", fg="green")


def _print_failures(failed: "list[Failure]") -> None:
    for f in failed:
        typer.secho(f"  ✗ {f.path}: {f.reason}", fg="yellow")


def _print_prune(plan: "PrunePlan", *, verb: str) -> None:
    for volume, n in plan.unmounted.items():
        typer.echo(f"{n} files on {volume} skipped (not mounted).")
    for root, n in plan.missing_roots.items():
        typer.echo(f"{n} files under {root} skipped (folder missing).")
    if not plan.remove:
        typer.echo("Nothing to prune.")
        return
    typer.echo(f"{verb} {len(plan.remove)} documents whose files are gone:")
    for doc in plan.remove[:10]:
        typer.echo(f"  {doc.path}")
    if len(plan.remove) > 10:
        typer.echo(f"  … and {len(plan.remove) - 10} more")


ScopeOption = Annotated[
    list[Path] | None,
    typer.Option("--in", help="Only search these files or folders (repeatable)."),
]


def _overrides(settings: Settings, **values: int | None) -> Settings:
    update = {k: v for k, v in values.items() if v is not None}
    return settings.model_copy(update=update) if update else settings


def _retriever(
    settings: Settings, scope: list[Path] | None
) -> "tuple[Retriever, list[str] | None]":
    """Open the index and resolve `--in`; stop with a message if either is unusable."""
    from qwn.index import IndexMismatch
    from qwn.ingest import open_index
    from qwn.retrieve import EmptyScope, Retriever

    try:
        index = open_index(settings)
    except IndexMismatch as e:
        raise _fail(str(e)) from None
    if index.counts()["chunks"] == 0:
        raise _fail("The index is empty. Add documents with `qwn ingest PATH...`.")
    retriever = Retriever(settings, index, make_registry(settings))
    try:
        doc_ids = retriever.scope(scope) if scope else None
    except EmptyScope as e:
        raise _fail(str(e)) from None
    return retriever, doc_ids


@app.command()
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="What to look for.")],
    rerank_k: Annotated[int | None, typer.Option(help="Results to show.")] = None,
    top_k: Annotated[int | None, typer.Option(help="Candidates before reranking.")] = None,
    rerank: Annotated[bool, typer.Option(help="Rerank the candidates.")] = True,
    hybrid: Annotated[
        bool, typer.Option(help="Vector + keyword search (off: vector only).")
    ] = True,
    scope: ScopeOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Show the best-matching pages, images and passages for QUERY."""
    settings = _overrides(ctx.obj, rerank_k=rerank_k, top_k=top_k)
    retriever, doc_ids = _retriever(settings, scope)
    try:
        hits = retriever.search(query, hybrid=hybrid, rerank=rerank, doc_ids=doc_ids)
    except EXPECTED_ERRORS as e:
        raise _fail(str(e)) from None

    if as_json:
        rows = [
            {
                "rank": i,
                "path": h.path,
                "page": h.page,
                "heading": h.chunk.heading_path or None,
                "kind": h.chunk.kind,
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score, 6),
                "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 6),
                "image_path": None if h.image_path is None else str(h.image_path),
            }
            for i, h in enumerate(hits, start=1)
        ]
        typer.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    for i, h in enumerate(hits, start=1):
        where = f"p.{h.page}" if h.page else (h.chunk.heading_path or h.chunk.kind)
        score = h.rerank_score if h.rerank_score is not None else h.score
        typer.secho(f"{i}. {h.path}  [{where}]  {score:.3f}", bold=True)
        snippet = " ".join(h.chunk.text.split())[:160]
        if snippet:
            typer.echo(f"   {snippet}")


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Your question.")],
    show_sources: Annotated[
        bool, typer.Option("--sources", help="Also list every source the model was given.")
    ] = False,
    scope: ScopeOption = None,
    top_k: Annotated[int | None, typer.Option(help="Candidates before reranking.")] = None,
    rerank_k: Annotated[int | None, typer.Option(help="Sources given to the model.")] = None,
    max_images: Annotated[int | None, typer.Option(help="Page images sent to the model.")] = None,
    guard: Annotated[
        bool | None,
        typer.Option(
            "--guard/--no-guard",
            help="Check the question and answer with Guard (default: the guard_enabled setting).",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Answer QUESTION from your documents, citing the pages it used."""
    from qwn.answer import Answerer, cite_label

    settings = _overrides(ctx.obj, top_k=top_k, rerank_k=rerank_k, max_images=max_images)
    retriever, doc_ids = _retriever(settings, scope)
    answerer = Answerer(settings, retriever, retriever.registry)
    try:
        response = answerer.ask(question, doc_ids=doc_ids, guard=guard)
    except EXPECTED_ERRORS as e:
        raise _fail(str(e)) from None
    except Exception as e:  # a model error: report it, keep the traceback in the log
        logging.getLogger(__name__).exception("ask failed")
        raise _fail(f"The model failed: {type(e).__name__}: {e}") from None

    pairs = list(zip(response.sources, response.hits, strict=True))
    if as_json:

        def row(source: "Source", hit: "Hit") -> dict[str, object]:
            return {
                "label": source.label,
                "cite": cite_label(source, hit),
                "path": source.path,
                "page": source.page,
                "heading": hit.chunk.heading_path or None,
                "score": round(source.score, 6),
            }

        out = {
            "answer": response.rendered(),
            "raw": response.text,
            "abstained": response.abstained,
            "blocked": response.blocked is not None,
            "guard": [c.as_dict() for c in response.checks],
            "cited": [row(s, h) for s, h in response.cited_sources()],
            "sources": [row(s, h) for s, h in pairs],
            "invented_citations": len(response.invented),
            "tokens": {"prompt": response.prompt_tokens, "completion": response.completion_tokens},
            "timings_s": {k: round(v, 3) for k, v in response.timings.items()},
        }
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return
    for check in response.warnings:
        typer.secho(f"⚠ {check.message()}", fg="yellow", err=True)
    if (block := response.blocked) is not None:
        typer.secho(f"⛔ {block.message()}", fg="red")
        return
    typer.echo(response.rendered())
    listed = pairs if show_sources else response.cited_sources()
    if listed:
        typer.echo("")
        typer.secho("Sources:" if show_sources else "Cited:", bold=True)
    cited = set(response.cited)
    for source, hit in listed:
        mark = "*" if show_sources and source.label in cited else " "
        where = f"  p.{source.page}" if source.page is not None else ""
        typer.echo(f" {mark}[{cite_label(source, hit)}]  {source.path}{where}")


@app.command()
def status(ctx: typer.Context) -> None:
    """Show the qwn home, the index, the model cache and memory."""
    from qwn.index import DB_NAME, Index
    from qwn.ingest import index_meta

    settings: Settings = ctx.obj
    typer.echo(f"home:   {settings.home}")
    typer.echo(f"data:   {settings.data_dir}")
    typer.echo(f"index:  {settings.index_dir}")
    typer.echo(f"logs:   {settings.log_dir}")

    db = settings.index_dir / DB_NAME
    if not db.exists():
        typer.echo("index:  empty (run `qwn ingest PATH...`)")
    else:
        index = Index(settings.index_dir, settings.embed_dim)
        counts = index.counts()
        kinds = ", ".join(f"{counts.get(k, 0)} {k}" for k in ("pdf", "image", "text"))
        size = sum(f.stat().st_size for f in settings.index_dir.glob(f"{DB_NAME}*"))
        size += _dir_size(settings.index_dir / "pages")  # eval-public/ indexes not counted
        typer.echo(
            f"index:  {counts['documents']} documents ({kinds}), {counts['chunks']} chunks, "
            f"{size / 1e6:.1f} MB on disk"
        )
        stored, current = index.meta(), index_meta(settings)
        typer.echo(f"embed:  {stored.get('embed_model')} (dim {stored.get('embed_dim')})")
        for key, value in current.items():
            if stored.get(key) != value:
                typer.secho(
                    f"        ✗ {key} differs from the settings: run `qwn ingest --reindex`",
                    fg="red",
                )

    from qwn.adapters import hub

    for m in MODELS.values():
        if m.group != "core":
            continue
        ok = hub.is_downloaded(m.repo, m.revision)
        mark = typer.style("✓ downloaded", fg="green") if ok else typer.style("✗ missing", fg="red")
        typer.echo(f"model:  {m.repo}@{m.revision[:7]}  {mark}")
    registry = make_registry(settings)
    try:
        mem = registry.memory_gb()
        typer.echo(
            f"memory: {mem['active']:.1f} GB active, {mem['recommended']:.1f} GB recommended "
            "(models load per process; none are loaded by `qwn status`)"
        )
    except Exception as e:  # MLX unavailable (e.g. not Apple Silicon)
        typer.echo(f"memory: unavailable ({type(e).__name__})")


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


@app.command("eval")
def eval_(
    ctx: typer.Context,
    set_: Annotated[str, typer.Option("--set", help="public or private.")] = "public",
    rerank: Annotated[bool, typer.Option(help="Score after reranking.")] = True,
    hybrid: Annotated[bool, typer.Option(help="Hybrid (on) or vector-only (off).")] = True,
    generate: Annotated[
        bool, typer.Option(help="Answer every query (greedy) and score the answers.")
    ] = True,
    guard: Annotated[
        bool, typer.Option("--guard", help="Also score Guard on eval/public/guard.jsonl.")
    ] = False,
    locate: Annotated[bool, typer.Option("--locate", help="Highlight metrics (phase 6).")] = False,
    update_baseline: Annotated[
        bool, typer.Option("--update-baseline", help="Save this run as the set's baseline.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Report even when not comparable with the baseline.")
    ] = False,
    clean: Annotated[
        bool, typer.Option("--clean", help="Delete public eval indexes for other settings.")
    ] = False,
) -> None:
    """Score retrieval and answers on the public or private eval set; check the exit criteria."""
    from qwn import eval as ev
    from qwn.index import IndexMismatch

    settings: Settings = ctx.obj
    if set_ not in ("public", "private"):
        raise _fail("--set must be public or private.")
    if locate:
        raise _fail("--locate arrives in phase 6.")
    if clean:
        for path in ev.clean_public(settings):
            typer.echo(f"removed {path}")
    es = ev.eval_set(settings, "public" if set_ == "public" else "private")
    registry = make_registry(settings)

    def progress(i: int, n: int) -> None:
        if i % 10 == 0 or i == n:
            typer.echo(f"  {i}/{n} queries", err=True)

    try:
        index = ev.prepare(settings, es, registry, say=typer.echo)
        result = ev.run(
            settings,
            es,
            index,
            registry,
            ev.Options(rerank=rerank, hybrid=hybrid, generate=generate, guard=guard),
            progress=progress,
        )
    except (ev.EvalError, IndexMismatch, *EXPECTED_ERRORS) as e:
        raise _fail(str(e)) from None

    path = ev.write_result(settings.home, result)
    typer.echo(ev.format_report(result))
    typer.echo(f"\nResults: {path}")

    baseline = ev.load_baseline(es)
    comparable = True
    if baseline is None:
        typer.echo("No baseline yet" + ("." if update_baseline else " (--update-baseline)."))
    elif diff := ev.differences(baseline["meta"], result["meta"]):
        comparable = False
        typer.secho(f"Not comparable with the baseline: {', '.join(diff)} differ.", fg="yellow")
    else:
        changed = ev.changes(baseline, result)
        typer.echo("Changes vs baseline:" if changed else "No per-query changes vs baseline.")
        for line in changed:
            typer.echo(f"  {line}")
    if update_baseline:
        ev.write_baseline(es, result)
        typer.secho(f"Baseline updated: {es.baseline}", fg="green")
    elif not comparable and not force:
        raise typer.Exit(1)
