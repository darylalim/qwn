"""`qwn` command line. A thin layer over qwn.*; phase 0 ships `qwn models pull/status`."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from qwn.config import HomeNotFound, Settings
from qwn.models_lock import MODELS, PinnedModel

# Offline unless a command opts in (only `qwn models pull`). huggingface_hub reads this once, at
# import, so it's set before anything imports it.
os.environ["HF_HUB_OFFLINE"] = "1"

app = typer.Typer(help="Private multimodal RAG over your documents, on-device with MLX.")
models_app = typer.Typer(help="Download and inspect the pinned models.")
app.add_typer(models_app, name="models")

LOCK_FILE = Path(__file__).with_name("models_lock.py")


@app.callback()
def main(ctx: typer.Context) -> None:
    try:
        ctx.obj = Settings()
    except HomeNotFound as e:
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(1) from None


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
            state = typer.style("✗ missing: run `qwn models pull`", fg="red")
        typer.echo(f"{m.group:5}  {m.repo}@{m.revision[:7]}  {state}")
