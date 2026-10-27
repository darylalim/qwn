"""`qwn` command line. A thin layer over qwn.*: no model or retrieval logic of its own."""

import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from qwn.config import HomeNotFound, Settings
from qwn.models import InsufficientMemory, ModelsInUse, Registry
from qwn.models_lock import MODELS, PinnedModel

if TYPE_CHECKING:
    from qwn.answer import Answerer, Response
    from qwn.conversations import Conversation, Exchange, SourceState, Store
    from qwn.ingest import Failure, PrunePlan
    from qwn.interfaces import Source
    from qwn.models import WarmLoad
    from qwn.retrieve import Hit, Retriever

# Offline unless a command opts in (only `qwn models pull`). huggingface_hub reads this once, at
# import, so it's set before anything imports it.
os.environ["HF_HUB_OFFLINE"] = "1"

app = typer.Typer(help="Private multimodal RAG over your documents, on-device with MLX.")
models_app = typer.Typer(help="Download and inspect the pinned models.")
app.add_typer(models_app, name="models")
eval_app = typer.Typer(help="Score retrieval and answers on the public, private or hard eval set.")
app.add_typer(eval_app, name="eval")

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
    question: Annotated[
        str | None, typer.Argument(help="Your question (or ask it out loud with --audio).")
    ] = None,
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
    audio: Annotated[
        Path | None,
        typer.Option(help="Ask a recorded question instead (WAV); it's transcribed on-device."),
    ] = None,
    speak: Annotated[
        Path | None, typer.Option(help="Also write the answer as speech to this WAV file.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Answer QUESTION from your documents, citing the pages it used."""
    from qwn.answer import Answerer, cite_label
    from qwn.voice import (
        SAMPLE_RATE,
        AudioError,
        NoSpeech,
        Transcript,
        Voice,
        encode_wav,
        load_audio,
        spoken_text,
    )

    if (question is None) == (audio is None):
        raise _fail("Give a QUESTION or --audio IN.wav (exactly one).")
    recording = None
    if audio is not None:
        try:
            recording = load_audio(audio)
        except (OSError, AudioError) as e:
            raise _fail(f"Can't read {audio}: {e}") from None
    settings = _overrides(ctx.obj, top_k=top_k, rerank_k=rerank_k, max_images=max_images)
    retriever, doc_ids = _retriever(settings, scope)
    answerer = Answerer(settings, retriever, retriever.registry)
    voice = Voice(settings, retriever.registry)
    transcript: Transcript | None = None
    try:
        if recording is not None:
            transcript = voice.transcribe(recording)
            if not transcript.text:
                raise NoSpeech(transcript.speech_s)
            question = transcript.text
            if not as_json:
                typer.secho(f"Heard: {question}", dim=True)
        response = answerer.ask(question or "", doc_ids=doc_ids, guard=guard)
        spoken, tts_s = None, 0.0
        if speak is not None and (say := spoken_text(response)) is not None:
            start = time.perf_counter()
            spoken = voice.speak(say)
            speak.write_bytes(encode_wav(spoken))
            tts_s = time.perf_counter() - start
    except NoSpeech as e:
        raise _fail(str(e)) from None
    except (OSError, *EXPECTED_ERRORS) as e:
        raise _fail(str(e)) from None
    except Exception as e:  # a model error: report it, keep the traceback in the log
        logging.getLogger(__name__).exception("ask failed")
        raise _fail(f"The model failed: {type(e).__name__}: {e}") from None
    timings = {**(transcript.timings if transcript else {}), **response.timings}
    if spoken is not None:
        timings["tts"] = tts_s

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
            "timings_s": {k: round(v, 3) for k, v in timings.items()},
        }
        if transcript is not None:
            out["transcript"] = transcript.text
        if speak is not None:
            out["speech"] = None if spoken is None else str(speak)
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return
    _print_response(response, all_sources=show_sources)
    if response.blocked is not None and speak is not None:
        typer.echo(f"Nothing spoken: {speak} not written.", err=True)
    elif spoken is not None:
        typer.echo(f"Spoken answer: {speak} ({len(spoken) / SAMPLE_RATE:.1f} s)", err=True)


def _plain(text: str) -> str:
    """Model or document text made safe to print: an echoed answer can't move the cursor or
    rewrite earlier lines (PLAN.md → CLI → `qwn chat`, Privacy and security)."""
    from qwn.conversations import printable

    return printable(text)


def _print_response(
    response: "Response",
    *,
    all_sources: bool,
    states: "dict[str, SourceState] | None" = None,
) -> None:
    """`qwn ask` and `qwn chat` output: the rewrite (if any), Guard warnings, then a block or
    the answer and its sources."""
    if response.standalone:
        typer.secho(f"Searched for: {_plain(response.standalone)}", dim=True)
    for check in response.warnings:
        typer.secho(f"⚠ {check.message()}", fg="yellow", err=True)
    if (block := response.blocked) is not None:
        typer.secho(f"⛔ {block.message()}", fg="red")
        return
    typer.echo(_plain(response.rendered()))
    _print_sources(response, all_sources=all_sources, states=states)


def _print_sources(
    response: "Response",
    *,
    all_sources: bool,
    states: "dict[str, SourceState] | None" = None,
) -> None:
    """The cited sources, or with `all_sources` every source the model was given (cited `*`).
    `states` (by chunk id) marks a saved answer's sources that changed or left the library."""
    from qwn.answer import cite_label
    from qwn.conversations import STATE_NOTES

    pairs = list(zip(response.sources, response.hits, strict=True))
    listed = pairs if all_sources else response.cited_sources()
    if listed:
        typer.echo("")
        typer.secho("Sources:" if all_sources else "Cited:", bold=True)
    cited = set(response.cited)
    for source, hit in listed:
        mark = "*" if all_sources and source.label in cited else " "
        where = f"  p.{source.page}" if source.page is not None else ""
        note = STATE_NOTES.get((states or {}).get(source.chunk_id, "ok"), "")
        note = f" {note}" if note else ""  # two spaces before it, as between the columns
        typer.echo(_plain(f" {mark}[{cite_label(source, hit)}]  {source.path}{where}{note}"))


CHAT_HELP = """Commands:
  /help               this list
  /quit               leave (or Ctrl-D)
  /new                start a new conversation (keeps the scope)
  /sources            every source the last answer was given, cited ones marked *
  /in PATH...         search only these files or folders (no PATH: everything); keeps the history
  /history            what the next follow-up is rewritten with
  /save               save this conversation; later answers are saved as you go
                      (in the index folder; nothing is saved otherwise)
  /export [PATH]      write this conversation as Markdown (here, or in or to PATH)
  /list [TEXT]        saved conversations, most recent first; TEXT: only those with these words
  /open N|ID          open a saved conversation (its number in /list, or its id)
  /rename N|ID TITLE  give a saved conversation a title of your own
  /delete N|ID        delete a saved conversation
Keys: Ctrl-C clears the line, or cancels an answer; ↑ recalls earlier questions."""
HISTORY_CHARS = 120  # /history cuts each answer here
PROMPT = "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} "


@app.command()
def chat(
    ctx: typer.Context,
    scope: ScopeOption = None,
    top_k: Annotated[int | None, typer.Option(help="Candidates before reranking.")] = None,
    rerank_k: Annotated[int | None, typer.Option(help="Sources given to the model.")] = None,
    max_images: Annotated[int | None, typer.Option(help="Page images sent to the model.")] = None,
    guard: Annotated[
        bool | None,
        typer.Option(
            "--guard/--no-guard",
            help="Check questions and answers with Guard (default: the guard_enabled setting).",
        ),
    ] = None,
    open_ref: Annotated[
        str | None,
        typer.Option(
            "--open",
            metavar="N|ID",
            help="Open a saved conversation (its number in /list, or its id).",
        ),
    ] = None,
) -> None:
    """Ask questions in a conversation: follow-ups like "and in 2025?" use the earlier ones."""
    import sys

    from qwn.answer import Answerer, history_caption
    from qwn.conversations import Store, UnknownConversation
    from qwn.models import WarmLoad

    settings = _overrides(ctx.obj, top_k=top_k, rerank_k=rerank_k, max_images=max_images)
    retriever, doc_ids = _retriever(settings, scope)
    store = Store(retriever.index.root)
    opened = None
    if open_ref is not None:  # before the models load: an unknown one stops here
        try:
            opened = store.load(store.resolve(open_ref).id)
        except UnknownConversation:
            raise _fail(
                f"No saved conversation {_plain(open_ref)}. Run `qwn chat` and type /list."
            ) from None
    registry = retriever.registry
    try:
        registry.process_lock.acquire()  # as `qwn ui`: one process owns the models
    except ModelsInUse as e:
        raise _fail(str(e)) from None
    try:
        answerer = Answerer(settings, retriever, registry)
        session = _Chat(answerer, settings.history_turns, scope or [], doc_ids, guard, store)
        typer.echo("qwn chat: ask about your documents. /help for commands, Ctrl-D to quit.")
        typer.echo(f"{session.scope_line()} · {history_caption(settings.history_turns)}")
        if opened is not None:
            session.open(opened, keep_scope=bool(scope))
        warm = WarmLoad(registry)
        if settings.warm_load:
            warm.start()
            typer.secho("Loading models in the background…", dim=True)
        session.run(warm, interactive=sys.stdin.isatty(), status=sys.stderr.isatty())
    finally:
        registry.process_lock.release()


class _Chat:
    """One `qwn chat` session: the scope and the completed exchanges, in memory until /save (or
    /open); from then on each new answer is autosaved (PLAN.md → CLI → Autosave saved
    conversations)."""

    def __init__(
        self,
        answerer: "Answerer",
        history_turns: int,
        scope: list[Path],
        doc_ids: list[str] | None,
        guard: bool | None,
        store: "Store",
    ) -> None:
        self.answerer = answerer
        self.history_turns = history_turns
        self.scope = scope
        self.doc_ids = doc_ids
        self.guard = guard
        self.store = store
        self.exchanges: list[Exchange] = []
        self.last: Response | None = None  # for /sources: None before an answer, after a block
        self.saved: Conversation | None = None  # the conversation as last saved or opened
        self.lost = False  # autosave found the file deleted elsewhere: these exchanges aren't saved
        self.unreadable: str | None = None  # a file autosave couldn't read: /save never writes it
        self.warned: str | None = None  # the command just refused for unsaved exchanges

    def scope_line(self) -> str:
        if self.doc_ids is None:
            n = self.answerer.retriever.index.counts()["documents"]
            return f"Searching in: all documents ({n})"
        paths = ", ".join(str(p) for p in self.scope)
        n = len(self.doc_ids)
        return f"Searching in: {paths} ({n} document{'s' if n != 1 else ''})"

    def run(self, warm: "WarmLoad", *, interactive: bool, status: bool) -> None:
        if interactive:
            import readline  # noqa: F401  (arrow keys and ↑ history for input(); no history file)
        self.status = status
        warned = False
        while True:
            if warm.state == "failed" and not warned:  # e.g. not enough memory: load lazily
                typer.secho(f"Couldn't preload the models: {warm.error}", fg="yellow", err=True)
                warned = True
            try:
                line = input(PROMPT if interactive else "")
            except EOFError:
                if interactive:
                    typer.echo("")
                    if not self.may_discard("/quit"):
                        continue
                return  # a pipe has nothing more to say: exit at once
            except KeyboardInterrupt:  # discards the line
                typer.echo("")
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self.command(line) == "quit":
                    return
                continue
            self.warned = None
            self.ask(line, warm)

    def unsaved(self) -> int:
        """Exchanges a failed autosave left off disk: newer than the last save, or all of them
        when the file was deleted elsewhere. A conversation never saved counts none: it is
        discarded silently, as saving starts only when you ask."""
        if self.saved is None:
            return len(self.exchanges) if self.lost else 0
        return len(self.exchanges) - len(self.saved.exchanges)

    def may_discard(self, name: str) -> bool:
        """Whether `name` (/quit, /new, /open) may drop unsaved exchanges: it asks once, and
        the same command again right away goes ahead."""
        n = self.unsaved()
        if n <= 0 or self.warned == name:
            self.warned = None
            return True
        self.warned = name
        newer = f"{n} newer questions aren't" if n != 1 else "1 newer question isn't"
        them = "them" if n != 1 else "it"
        typer.echo(f"{newer} saved. /save to keep {them}, or repeat the command to discard {them}.")
        return False

    def command(self, line: str) -> str | None:
        import shlex

        from qwn.retrieve import EmptyScope

        first, rest = [*line.split(maxsplit=1), ""][:2]
        if first == "/rename":  # the title is free text: "Bob's notes"
            self.warned = None
            self.rename(line)
            return None
        if first == "/list":  # so is a search: "Bob's"
            self.warned = None
            self.list_saved(rest)
            return None
        try:
            name, *args = shlex.split(line)
        except ValueError as e:  # an unclosed quote
            typer.secho(f"Can't read that command: {e}.", fg="red", err=True)
            return None
        if name in ("/quit", "/new", "/open"):
            if name == "/open" and not args:
                typer.echo("Which one? /open N or /open ID (see /list).")
                return None
            if not self.may_discard("/quit" if name == "/quit" else name):
                return None
        else:
            self.warned = None
        if name == "/quit":
            return "quit"
        if name == "/help":
            typer.echo(CHAT_HELP)
        elif name == "/new":
            self.exchanges, self.last, self.saved, self.lost = [], None, None, False
            typer.echo("New conversation.")
        elif name == "/sources":
            if self.last is None:
                typer.echo("No answer yet.")
            else:
                _print_sources(self.last, all_sources=True, states=self.states(self.last))
        elif name == "/save":
            self.save()
        elif name == "/export":
            self.export(args[0] if args else None)
        elif name == "/open":
            self.open_ref(args[0])
        elif name == "/delete":
            if not args:
                typer.echo("Which one? /delete N or /delete ID (see /list).")
            else:
                self.delete(args[0])
        elif name == "/in":
            paths = [Path(a) for a in args]
            try:
                doc_ids = self.answerer.retriever.scope(paths) if paths else None
            except EmptyScope as e:
                typer.secho(str(e), fg="red", err=True)
                return None
            self.scope, self.doc_ids = paths, doc_ids
            typer.echo(self.scope_line())
        elif name == "/history":
            self.show_history()
        else:
            typer.echo(f"Unknown command {_plain(name)}. Type /help.")
        return None

    def show_history(self) -> None:
        from qwn.answer import conversation

        if self.history_turns == 0:
            typer.echo("Follow-ups aren't rewritten: each question is answered on its own.")
            return
        turns = conversation(self.pairs(), self.history_turns)
        if not turns:
            typer.echo("No history yet.")
        for turn in turns:
            answer = " ".join(turn.answer.split())
            if len(answer) > HISTORY_CHARS:
                answer = answer[: HISTORY_CHARS - 1] + "…"
            typer.echo(_plain(f"{PROMPT}{turn.question}"))
            typer.secho(f"  {_plain(answer)}", dim=True)

    def ask(self, question: str, warm: "WarmLoad") -> None:
        from qwn.answer import STAGE_LABELS, Stage, conversation
        from qwn.conversations import Exchange
        from qwn.retrieve import EmptyScope

        registry = self.answerer.registry

        def on_stage(stage: "Stage") -> None:
            if self.status:
                busy = registry.lock.locked() or warm.state == "loading"
                label = "Waiting for the model…" if busy else STAGE_LABELS[stage]
                typer.echo("\r\x1b[2K" + typer.style(label, dim=True), nl=False, err=True)

        history = conversation(self.pairs(), self.history_turns)
        try:
            response = self.answerer.ask(
                question, history=history, doc_ids=self.doc_ids, guard=self.guard, on_stage=on_stage
            )
        except KeyboardInterrupt:
            self._clear_status()
            typer.echo("Cancelled.")
            return
        except (EmptyScope, *EXPECTED_ERRORS) as e:
            self._clear_status()
            typer.secho(str(e), fg="red", err=True)
            self.last = None
            return
        except Exception as e:  # a model error: report it, keep the traceback in the log
            self._clear_status()
            logging.getLogger(__name__).exception("chat: ask failed")
            typer.secho(f"The model failed: {type(e).__name__}: {e}", fg="red", err=True)
            self.last = None
            return
        self._clear_status()
        _print_response(response, all_sources=False)
        self.exchanges.append(Exchange(uuid.uuid4().hex[:12], question, response, self.searched()))
        self.last = response if response.blocked is None else None
        self.autosave()

    def autosave(self) -> None:
        """Write the new exchange to the saved (or opened) conversation's file. Quiet when it
        works; one line when the file moved on, went or can't be read elsewhere."""
        from qwn.conversations import make

        if self.saved is None:
            return
        try:
            done = self.store.autosave(make(self.exchanges, self.saved))
        except OSError as e:
            typer.secho(f"Couldn't save the conversation: {e}", fg="red", err=True)
            return
        if done.conversation is not None:  # saved or copied
            self.saved = done.conversation
        if done.state == "copied":
            typer.echo(
                f"This conversation changed elsewhere, so it was saved as a copy: {self.saved.id}."
            )
        elif done.state == "gone":
            self.saved, self.lost = None, True
            typer.echo("This conversation was deleted elsewhere. /save to keep it again.")
        elif done.state == "unreadable":
            self.unreadable = self.saved.id
            typer.echo(
                "Couldn't save to this conversation's file (see the log). /save writes a new copy."
            )

    def pairs(self) -> "list[tuple[str, Response]]":
        return [(e.question, e.response) for e in self.exchanges]

    def searched(self) -> list[str]:
        """The scope as saved with an exchange: absolute paths, empty for all documents."""
        if self.doc_ids is None:
            return []
        return [str(p.expanduser().resolve()) for p in self.scope]

    def states(self, response: "Response") -> "dict[str, SourceState]":
        from qwn.conversations import source_state

        documents = {d.path: d.sha256 for d in self.answerer.retriever.index.documents()}
        return {h.chunk.chunk_id: source_state(h, documents) for h in response.hits}

    def save(self) -> None:
        from qwn.conversations import make

        if not self.exchanges:
            typer.echo("Nothing to save yet.")
            return
        if self.saved is not None and not self.unsaved():
            typer.echo("Already saved: new answers are saved as you go.")
            return
        previous = self.saved if self.saved and self.saved.id != self.unreadable else None
        try:
            self.saved = self.store.save(make(self.exchanges, previous))
        except OSError as e:
            typer.secho(f"Couldn't save the conversation: {e}", fg="red", err=True)
            return
        self.lost = False
        n = len(self.saved.exchanges)
        title = _plain(self.saved.title)
        typer.echo(
            f"Saved as {self.saved.id}: {title} ({n} question{'s' if n != 1 else ''}). "
            "New answers are saved as you go."
        )

    def export(self, where: str | None) -> None:
        """Write the conversation as Markdown: to a file PATH (never over an existing one), or a
        dated name in a folder PATH or the current folder (`-2` etc. when it's taken)."""
        from qwn.conversations import (
            export_name,
            free_path,
            make,
            markdown,
            now,
            source_states,
            write_export,
        )

        if not self.exchanges:
            typer.echo("Nothing to export yet.")
            return
        conversation = make(self.exchanges, self.saved)
        exported = now()
        target = Path(where).expanduser() if where else Path.cwd()
        if target.is_dir():
            target = free_path(target, export_name(conversation, exported))
        documents = {d.path: d.sha256 for d in self.answerer.retriever.index.documents()}
        text = markdown(conversation, source_states(conversation, documents), exported)
        try:
            write_export(target, text)
        except FileExistsError:
            typer.secho(
                _plain(f"{target} exists. /export PATH to choose another."), fg="red", err=True
            )
            return
        except FileNotFoundError:
            typer.secho(_plain(f"No folder {target.parent}."), fg="red", err=True)
            return
        except OSError as e:
            typer.secho(_plain(f"Couldn't export to {target}: {e.strerror}."), fg="red", err=True)
            return
        n = len(conversation.exchanges)
        typer.echo(_plain(f"Exported {n} question{'s' if n != 1 else ''} to {target}."))

    def rename(self, line: str) -> None:
        """/rename N|ID TITLE…: the rest of the line is the title, quotes optional."""
        from dataclasses import replace

        from qwn.conversations import UnknownConversation

        _, ref, title = [*line.split(maxsplit=2), "", ""][:3]
        if len(title) >= 2 and title[0] == title[-1] and title[0] in "'\"":
            title = title[1:-1]
        if not ref:
            typer.echo("Which one? /rename N TITLE or /rename ID TITLE (see /list).")
            return
        if not title.strip():
            typer.echo("Which title? /rename N TITLE.")
            return
        try:
            renamed = self.store.rename(self.store.resolve(ref).id, title)
        except UnknownConversation:
            typer.secho(f"No saved conversation {_plain(ref)}. Type /list.", fg="red", err=True)
            return
        except ValueError as e:  # nothing left after dropping control characters
            typer.secho(str(e), fg="red", err=True)
            return
        except OSError as e:
            typer.secho(f"Couldn't rename the conversation: {e}", fg="red", err=True)
            return
        if self.saved is not None and self.saved.id == renamed.id:
            self.saved = replace(self.saved, title=renamed.title)  # the next /save keeps it
        end = "" if renamed.title.endswith((".", "?", "!", "…")) else "."
        typer.echo(_plain(f"Renamed to {renamed.title}{end}"))

    def list_saved(self, text: str = "") -> None:
        """/list [TEXT…]: every saved conversation, or only those with these words, each with
        its /list number (a search never renumbers what /open N then opens)."""
        from qwn.conversations import saved_when

        found, total = self.store.search(text)
        if not total:
            typer.echo("No saved conversations.")
            return
        if not found:
            typer.echo(f"No saved conversation matches {' '.join(_plain(text).split())}.")
            return
        for i, s in found:
            n = f"{s.exchanges} question{'s' if s.exchanges != 1 else ''}"
            typer.echo(_plain(f"{i:>3}  {s.id}  {saved_when(s.saved)}  {n}  {s.title}"))
        if text.strip():
            typer.echo(f"{len(found)} of {total} saved conversations match.")

    def open_ref(self, ref: str) -> None:
        from qwn.conversations import UnknownConversation

        try:
            conversation = self.store.load(self.store.resolve(ref).id)
        except UnknownConversation:
            typer.secho(f"No saved conversation {_plain(ref)}. Type /list.", fg="red", err=True)
            return
        self.open(conversation, keep_scope=False)

    def open(self, conversation: "Conversation", *, keep_scope: bool) -> None:
        """Replace the history (and, unless `--in` was given, the scope) with a saved
        conversation, then show the scope and the last exchange."""
        from qwn.retrieve import EmptyScope

        self.exchanges = list(conversation.exchanges)
        self.saved, self.lost = conversation, False
        last = conversation.exchanges[-1].response
        self.last = last if last.blocked is None else None
        typer.echo(f"Opened {conversation.id}: {_plain(conversation.title)}")
        if keep_scope:
            typer.echo(f"{self.scope_line()} (from --in)")
        else:
            paths = [Path(p) for p in conversation.scope]
            try:
                self.scope = paths
                self.doc_ids = self.answerer.retriever.scope(paths) if paths else None
            except EmptyScope:
                self.scope, self.doc_ids = [], None
                typer.echo(
                    "The saved folders and files are no longer indexed: searching all documents."
                )
            typer.echo(self.scope_line())
        n = len(self.exchanges)
        if n > 1:
            typer.secho(f"{n - 1} earlier question{'s' if n > 2 else ''}: /history", dim=True)
        typer.echo(_plain(f"{PROMPT}{conversation.exchanges[-1].question}"))
        _print_response(last, all_sources=False, states=self.states(last))

    def delete(self, ref: str) -> None:
        from qwn.conversations import UnknownConversation

        try:
            summary = self.store.resolve(ref)
        except UnknownConversation:
            typer.secho(f"No saved conversation {_plain(ref)}. Type /list.", fg="red", err=True)
            return
        self.store.delete(summary.id)
        if self.saved is not None and self.saved.id == summary.id:
            self.saved = None  # still open, but no longer saved: /save makes a new file
        end = "" if summary.title.endswith((".", "?", "!", "…")) else "."
        typer.echo(_plain(f"Deleted {summary.title}{end}"))

    def _clear_status(self) -> None:
        if self.status:
            typer.echo("\r\x1b[2K", nl=False, err=True)


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
        size += _dir_size(settings.index_dir / "pages")  # eval-public/ and eval-hard/ not counted
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


@eval_app.callback(invoke_without_command=True)
def eval_(
    ctx: typer.Context,
    set_: Annotated[str, typer.Option("--set", help="public, private, hard or turns.")] = "public",
    rerank: Annotated[bool, typer.Option(help="Score after reranking.")] = True,
    hybrid: Annotated[bool, typer.Option(help="Hybrid (on) or vector-only (off).")] = True,
    generate: Annotated[
        bool, typer.Option(help="Answer every query (greedy) and score the answers.")
    ] = True,
    guard: Annotated[
        bool, typer.Option("--guard", help="Also score Guard on eval/public/guard.jsonl.")
    ] = False,
    locate: Annotated[
        bool, typer.Option("--locate", help="Also score highlights on cited pages (phase 6).")
    ] = False,
    update_baseline: Annotated[
        bool, typer.Option("--update-baseline", help="Save this run as the set's baseline.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Report even when not comparable with the baseline.")
    ] = False,
    clean: Annotated[
        bool,
        typer.Option("--clean", help="Delete public and hard eval indexes for other settings."),
    ] = False,
) -> None:
    """Score retrieval and answers on an eval set; check exit criteria (none for hard or turns)."""
    if ctx.invoked_subcommand is not None:
        return
    from qwn import eval as ev
    from qwn.index import IndexMismatch

    settings: Settings = ctx.obj
    if set_ not in ("public", "private", "hard", "turns"):
        raise _fail("--set must be public, private, hard or turns.")
    if locate and not generate:
        raise _fail("--locate scores highlights on answers: drop --no-generate.")
    if clean:
        for path in ev.clean_disposable(settings):
            typer.echo(f"removed {path}")
    es = ev.eval_set(settings, set_)
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
            ev.Options(rerank=rerank, hybrid=hybrid, generate=generate, guard=guard, locate=locate),
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


REVIEW_CHOICES = {
    "a": "accept",
    "e": "edit",
    "u": "unanswerable",
    "s": "skip",
    "d": "discard",
    "q": "quit",
}


@eval_app.command("review")
def eval_review(ctx: typer.Context) -> None:
    """Turn 👍/👎 drafts from the Chat page into private eval queries, one at a time."""
    from qwn import eval as ev

    settings: Settings = ctx.obj
    drafts_path = ev.candidates_path(settings.home)
    queries_path = ev.eval_set(settings, "private").queries
    drafts = ev.read_jsonl(drafts_path)
    if not drafts:
        typer.echo("No drafts to review. Rate answers with 👍/👎 on the Chat page first.")
        return
    added = 0
    for n, draft in enumerate(drafts, start=1):
        up = draft["rating"] == "up"
        typer.secho(
            f"\nDraft {n}/{len(drafts)}  {'👍' if up else '👎'}  {draft['created']}", bold=True
        )
        typer.echo(f"  question: {draft['query']}")
        if "asked" in draft:  # a Chat follow-up: `query` is its rewrite
            typer.echo(f"  asked:    {draft['asked']}")
        if draft["abstained"]:
            typer.echo('  answer:   abstained ("I couldn\'t find this")')
        typer.echo(f"  cited:    {ev.format_expected(draft['cited']) or '(nothing)'}")
        if draft.get("scope"):
            typer.echo(f"  scope:    {', '.join(draft['scope'])}")
        row = None
        while True:
            options = "[a]ccept, " if up else ""
            key = (
                typer.prompt(
                    f"  {options}[e]dit, [u]nanswerable, [s]kip, [d]iscard, [q]uit", default="s"
                )
                .strip()
                .lower()[:1]
            )
            choice = REVIEW_CHOICES.get(key)
            if choice == "accept" and not up:
                typer.echo("  A 👎 draft's citations were wrong: use [e]dit to set the pages.")
                continue
            if choice is not None:
                break
            typer.echo("  Choose one of the letters shown.")
        if choice == "quit":
            break
        if choice == "skip":
            continue
        if choice in ("accept", "unanswerable", "edit"):
            qid = ev.next_query_id(queries_path)
            if choice == "accept":
                row = ev.draft_query(draft, qid)
            elif choice == "unanswerable":
                row = ev.draft_query(draft, qid, expected=[])
            else:
                default = "" if draft["abstained"] else ev.format_expected(draft["cited"])
                expected = ev.parse_expected(
                    typer.prompt(
                        "  expected (path:page, path#heading, path; comma-separated)",
                        default=default,
                    )
                )
                contains = _csv(typer.prompt("  answer_contains (comma-separated)", default=""))
                tags = _csv(typer.prompt("  tags (comma-separated)", default=""))
                row = ev.draft_query(
                    draft, qid, expected=expected, answer_contains=contains, tags=tags or None
                )
            ev.append_query(queries_path, row)
            added += 1
            typer.secho(f"  added {qid}", fg="green")
        ev.remove_draft(drafts_path, draft["draft_id"])
    if added:
        typer.echo(
            f"\nAdded {added} queries to {queries_path}. The eval set changed: re-run "
            "`qwn eval --set private --update-baseline` once you're done."
        )


def _csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


UI_APP = Path(__file__).parent / "ui" / "app.py"


@app.command()
def ui(ctx: typer.Context) -> None:
    """Open the Streamlit app (localhost only). It warm-loads the core models in the background."""
    import sys

    settings: Settings = ctx.obj
    # Streamlit reads .streamlit/config.toml from the working directory: run from the home.
    os.chdir(settings.home)
    # No file watcher: it walks every loaded module on each rerun, and transformers' lazy modules
    # then try to import torch (not installed), logging a traceback per module. Hot reload is for
    # development: `streamlit run src/qwn/ui/app.py` directly keeps it.
    argv = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(UI_APP),
        "--server.fileWatcherType",
        "none",
    ]
    os.execv(sys.executable, argv)
