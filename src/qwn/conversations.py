"""Saved conversations: explicit save and reopen, one JSON file per conversation.

See PLAN.md → CLI → Saved conversations. Files live in `index_root / "conversations"`, beside the
page renders their sources point at. The Chat page and `qwn chat` both use this module; neither
has storage code of its own. A saved file holds questions, answers and document text, so it is
loaded strictly (nothing half-loaded, no image path outside the index) and everything shown from
it goes through the same filters as a fresh answer. Nothing here logs titles, questions or
answers: only ids, counts, file names and timings.

Export (`markdown`) and rename are in PLAN.md → CLI → Export and rename conversations: an export
is the only file qwn writes your questions to outside the index, and only when you ask.
"""

import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from qwn.answer import Response, cite_label
from qwn.guard import Check
from qwn.index import Chunk
from qwn.interfaces import Source, Verdict
from qwn.retrieve import Hit

logger = logging.getLogger(__name__)

FORMAT = 1
FOLDER = "conversations"
TITLE_CHARS = 60
_ID = re.compile(r"^[0-9a-f]{12}$")
_reported: set[Path] = set()  # unreadable files already logged, once per process

SourceState = Literal["ok", "changed", "missing"]
STATE_NOTES: dict[SourceState, str] = {
    "changed": " (changed since this answer)",
    "missing": " (no longer in your library)",
}

# ANSI escape sequences (CSI, OSC, two-byte), then any other C0/C1 control but newline and tab
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_HEADING = re.compile(r"^( {0,3})#", re.MULTILINE)
_EXPORT_NAME = "qwn-conversation-{:%Y-%m-%d-%H%M}{}.md"


class UnknownConversation(LookupError):
    """No saved conversation with this id, or its file can't be read."""

    def __init__(self, ref: str) -> None:
        super().__init__(f"No saved conversation {ref}.")
        self.ref = ref


@dataclass(frozen=True)
class Exchange:  # one answered question
    id: str  # the Chat message id (12 hex characters)
    question: str  # the message as typed or transcribed
    response: Response
    scope: list[str] = field(default_factory=list)  # absolute paths searched; empty = all
    spoken: bool = False


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created: str  # local ISO-8601 with offset
    saved: str  # "" until saved
    exchanges: tuple[Exchange, ...]

    @property
    def scope(self) -> list[str]:
        """The scope the last question was asked in."""
        return list(self.exchanges[-1].scope) if self.exchanges else []

    def pairs(self) -> list[tuple[str, Response]]:
        """(question, Response) pairs, for `qwn.answer.conversation`."""
        return [(e.question, e.response) for e in self.exchanges]


@dataclass(frozen=True)
class Summary:  # one row of `Store.list`
    id: str
    title: str
    saved: str
    exchanges: int


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def printable(text: str) -> str:
    """Model or document text with escape sequences and control characters (but newline and
    tab) dropped, so it can't move a terminal's cursor or hide in an exported file."""
    return _CONTROL.sub("", _ANSI.sub("", text))


def title_for(question: str) -> str:
    """The first question with whitespace collapsed, cut to `TITLE_CHARS` with an ellipsis."""
    title = " ".join(question.split())
    return title if len(title) <= TITLE_CHARS else title[: TITLE_CHARS - 1].rstrip() + "…"


def clean_title(text: str) -> str:
    """A title you typed: control characters dropped, whitespace collapsed, cut as `title_for`."""
    title = title_for(printable(text))
    if not title:
        raise ValueError("A title can't be empty.")
    return title


def make(exchanges: Iterable[Exchange], previous: Conversation | None = None) -> Conversation:
    """The conversation to save: `previous`'s id, title and creation time if it was saved or
    opened before (saving again overwrites its file, and keeps a renamed title), else new ones."""
    items = tuple(exchanges)
    if not items:
        raise ValueError("a conversation needs at least one exchange")
    return Conversation(
        id=previous.id if previous else new_id(),
        title=previous.title if previous else title_for(items[0].question),
        created=previous.created if previous else now(),
        saved=previous.saved if previous else "",
        exchanges=items,
    )


def saved_time(conversation: Conversation) -> str:
    """ "14:02": when it was last saved, for the Chat page's caption."""
    return datetime.fromisoformat(conversation.saved).strftime("%H:%M")


def saved_when(saved: str) -> str:
    """ "2026-10-26 14:02", for lists."""
    return datetime.fromisoformat(saved).strftime("%Y-%m-%d %H:%M")


def source_state(hit: Hit, documents: dict[str, str]) -> SourceState:
    """Whether a saved source still shows what the answer saw. `documents` is `{path: sha256}`
    from `Index.documents()`. A reindexed edit keeps the same render path (`doc_id` hashes the
    path), so the sha256 decides `changed`."""
    sha = documents.get(hit.path)
    if sha is None or (hit.image_path is not None and not hit.image_path.is_file()):
        return "missing"
    return "ok" if sha == hit.chunk.sha256 else "changed"


def source_states(conversation: Conversation, documents: dict[str, str]) -> dict[str, SourceState]:
    """`source_state` for every source in the conversation, by chunk id."""
    return {
        h.chunk.chunk_id: source_state(h, documents)
        for e in conversation.exchanges
        for h in e.response.hits
    }


def export_name(conversation: Conversation, exported: str) -> str:
    """The default export file name, from the export time: never the title, which could hold a
    `/` or say what the documents are in a file listing (`conversation` is unused for that)."""
    return _EXPORT_NAME.format(datetime.fromisoformat(exported), "")


def free_path(folder: Path, name: str) -> Path:
    """`folder / name`, or with `-2`, `-3`, … before `.md` when that file exists."""
    path, n = folder / name, 1
    while path.exists():
        n += 1
        path = folder / f"{name.removesuffix('.md')}-{n}.md"
    return path


def write_export(path: Path, text: str) -> None:
    """Write an export `0o600`, never over an existing file (`FileExistsError`, even for one
    that appears after the caller checked), never creating folders (`FileNotFoundError`)."""
    start = time.perf_counter()
    data = text.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    path.chmod(0o600)  # os.open's mode is masked by the umask
    logger.info(
        "conversation exported: bytes=%d seconds=%.3f", len(data), time.perf_counter() - start
    )


def markdown(conversation: Conversation, states: dict[str, SourceState], exported: str) -> str:
    """The conversation as Markdown, for your notes: questions, answers with `[file p.N]`
    citations, and each answer's cited sources (path, page, whether it changed since). No
    source text or page images. Pure: `states` from `source_states`, `exported` a local
    ISO-8601 time. Everything that can carry document text goes through `_md`."""
    n = len(conversation.exchanges)
    saved = f"saved {saved_when(conversation.saved)}" if conversation.saved else "not saved"
    lines = [
        f"# {_md_line(conversation.title)}",
        "",
        f"Exported from qwn, {saved_when(exported)} · {n} question{'s' if n != 1 else ''} · "
        f"{saved}",
    ]
    for i, e in enumerate(conversation.exchanges, start=1):
        r = e.response
        lines += ["", f"## {i}. {_md_line(e.question)}"]
        if e.spoken:
            lines += ["", "(spoken question)"]
        if r.standalone:
            lines += ["", f"Searched for: {_md_line(r.standalone)}"]
        if e.scope:
            lines += ["", f"Searched in: {_md_line(', '.join(e.scope))}"]
        lines += [x for check in r.warnings for x in ("", f"> ⚠ {_md_line(check.message())}")]
        if (block := r.blocked) is not None:
            lines += ["", f"> ⛔ {_md_line(block.message())}"]
            continue
        lines += ["", _md(r.rendered())]
        cited = r.cited_sources()
        if cited:
            lines += ["", "Sources:", ""]
        for source, hit in cited:
            page = f", page {source.page}" if source.page is not None else ""
            note = STATE_NOTES.get(states.get(source.chunk_id, "ok"), "")
            lines.append(
                f"- [{_md_line(cite_label(source, hit))}] {_md_line(source.path)}{page}{note}"
            )
    return "\n".join(lines) + "\n"


def _md(text: str) -> str:
    """Model or document text made inert in a Markdown viewer: no control characters, raw HTML
    shown as text, no image a viewer would fetch, no line that becomes a heading. Links stay
    text with their URL: following one takes a click."""
    text = printable(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _HEADING.sub(r"\1\\#", text.replace("![", "!\\["))


def _md_line(text: str) -> str:
    """`_md` on one line (a heading, a path): newlines and tabs collapsed to spaces."""
    return _md(" ".join(printable(text).split()))


class Store:
    """Saved conversations in `index_root / "conversations"`, one `<id>.json` each."""

    def __init__(self, index_root: Path) -> None:
        self.index_root = index_root
        self.folder = index_root / FOLDER

    def path(self, conversation_id: str) -> Path:
        if not _ID.match(conversation_id):
            raise UnknownConversation(conversation_id)
        return self.folder / f"{conversation_id}.json"

    def save(self, conversation: Conversation) -> Conversation:
        """Write the whole conversation atomically, with `saved` set to now."""
        start = time.perf_counter()
        saved = Conversation(
            conversation.id,
            conversation.title,
            conversation.created,
            now(),
            conversation.exchanges,
        )
        self._write(saved)
        logger.info(
            "conversation saved: id=%s exchanges=%d seconds=%.3f",
            saved.id,
            len(saved.exchanges),
            time.perf_counter() - start,
        )
        return saved

    def rename(self, conversation_id: str, title: str) -> Conversation:
        """Give a saved conversation a title of your own. Only the title changes: `saved` stays,
        so a rename doesn't move it up the list. `ValueError` for an empty title."""
        renamed = replace(self._read(self.path(conversation_id)), title=clean_title(title))
        self._write(renamed)
        logger.info("conversation renamed: id=%s", renamed.id)
        return renamed

    def _write(self, conversation: Conversation) -> None:
        """Atomically: a temp file in the folder, then `os.replace`."""
        target = self.path(conversation.id)
        data = json.dumps(self._dump(conversation), indent=2, sort_keys=True, ensure_ascii=False)
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.folder.chmod(0o700)  # mkdir's mode is masked by the umask
        fd, tmp = tempfile.mkstemp(dir=self.folder, prefix=".", suffix=".tmp")  # 0o600
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def list(self) -> list[Summary]:
        """Every readable saved conversation, most recently saved first."""
        if not self.folder.is_dir():
            return []
        found: list[Summary] = []
        for file in self.folder.glob("*.json"):
            if not _ID.match(file.stem):
                continue
            try:
                c = self._read(file)
            except UnknownConversation:
                if file not in _reported:  # the Chat page lists on every rerun
                    _reported.add(file)
                    logger.warning("unreadable saved conversation left out: %s", file.name)
                continue
            found.append(Summary(c.id, c.title, c.saved, len(c.exchanges)))
        found.sort(key=lambda s: datetime.fromisoformat(s.saved), reverse=True)
        return found

    def load(self, conversation_id: str) -> Conversation:
        conversation = self._read(self.path(conversation_id))
        logger.info(
            "conversation loaded: id=%s exchanges=%d", conversation.id, len(conversation.exchanges)
        )
        return conversation

    def delete(self, conversation_id: str) -> None:
        """Unlink the file; a missing one is fine (two tabs can delete the same conversation)."""
        self.path(conversation_id).unlink(missing_ok=True)
        logger.info("conversation deleted: id=%s", conversation_id)

    def resolve(self, ref: str) -> Summary:
        """`ref` as an id, or as a 1-based number in `list` order (`qwn chat`'s /open N)."""
        summaries = self.list()
        if _ID.match(ref):
            match = next((s for s in summaries if s.id == ref), None)
        elif ref.isdigit() and 1 <= int(ref) <= len(summaries):
            match = summaries[int(ref) - 1]
        else:
            match = None
        if match is None:
            raise UnknownConversation(ref)
        return match

    # file format

    def _read(self, file: Path) -> Conversation:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            conversation = self._load(data)
        except (OSError, ValueError, TypeError, KeyError, _Bad):
            raise UnknownConversation(file.stem) from None
        if conversation.id != file.stem:
            raise UnknownConversation(file.stem)
        return conversation

    def _dump(self, c: Conversation) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "id": c.id,
            "title": c.title,
            "created": c.created,
            "saved": c.saved,
            "exchanges": [
                {
                    "id": e.id,
                    "question": e.question,
                    "spoken": e.spoken,
                    "scope": list(e.scope),
                    "response": self._dump_response(e.response),
                }
                for e in c.exchanges
            ],
        }

    def _dump_response(self, r: Response) -> dict[str, Any]:
        return {
            "text": r.text,
            "sources": [
                {
                    "label": s.label,
                    "chunk_id": s.chunk_id,
                    "path": s.path,
                    "page": s.page,
                    "text": s.text,
                    "image_path": self._relative(s.image_path),
                    "score": s.score,
                }
                for s in r.sources
            ],
            "hits": [
                {
                    **{f.name: getattr(h.chunk, f.name) for f in fields(Chunk)},
                    "score": h.score,
                    "rerank_score": h.rerank_score,
                }
                for h in r.hits
            ],
            "cited": list(r.cited),
            "invented": list(r.invented),
            "abstained": r.abstained,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "checks": [c.as_dict() for c in r.checks],
            "standalone": r.standalone,
        }

    def _relative(self, path: Path | None) -> str | None:
        if path is None or not path.is_relative_to(self.index_root):
            return None  # never happens for retrieved sources; a lost thumbnail at worst
        return path.relative_to(self.index_root).as_posix()

    def _absolute(self, relative: str | None) -> Path | None:
        """A stored image path back under this index root. An absolute path or one with `..`
        is rejected: a hand-edited file must not point an image outside the index."""
        if relative is None:
            return None
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or not relative:
            raise _Bad("image path outside the index")
        return self.index_root / p

    def _load(self, data: Any) -> Conversation:
        d = _obj(data)
        if _int(d, "format") != FORMAT:
            raise _Bad("unknown format")
        conversation_id = _str(d, "id")
        if not _ID.match(conversation_id):
            raise _Bad("bad id")
        created, saved = _str(d, "created"), _str(d, "saved")
        datetime.fromisoformat(created)
        datetime.fromisoformat(saved)  # ValueError if not a time
        exchanges = tuple(self._load_exchange(_obj(e)) for e in _list(d, "exchanges"))
        if not exchanges:
            raise _Bad("no exchanges")
        return Conversation(conversation_id, _str(d, "title"), created, saved, exchanges)

    def _load_exchange(self, d: dict[str, Any]) -> Exchange:
        return Exchange(
            id=_str(d, "id"),
            question=_str(d, "question"),
            response=self._load_response(_obj(d["response"])),
            scope=[_as(p, str) for p in _list(d, "scope")],
            spoken=_bool(d, "spoken"),
        )

    def _load_response(self, d: dict[str, Any]) -> Response:
        sources = [self._load_source(_obj(s)) for s in _list(d, "sources")]
        hits = [self._load_hit(_obj(h)) for h in _list(d, "hits")]
        labels = [s.label for s in sources]
        cited = [_as(c, str) for c in _list(d, "cited")]
        if len(sources) != len(hits) or not set(cited) <= set(labels):
            raise _Bad("sources, hits and citations don't match")
        return Response(
            text=_str(d, "text"),
            sources=sources,
            hits=hits,
            cited=cited,
            invented=[_as(c, str) for c in _list(d, "invented")],
            abstained=_bool(d, "abstained"),
            prompt_tokens=_int(d, "prompt_tokens"),
            completion_tokens=_int(d, "completion_tokens"),
            checks=[_load_check(_obj(c)) for c in _list(d, "checks")],
            standalone=_opt_str(d, "standalone"),
        )

    def _load_source(self, d: dict[str, Any]) -> Source:
        return Source(
            label=_str(d, "label"),
            chunk_id=_str(d, "chunk_id"),
            path=_str(d, "path"),
            page=_opt_int(d, "page"),
            text=_str(d, "text"),
            image_path=self._absolute(_opt_str(d, "image_path")),
            score=_float(d, "score"),
        )

    def _load_hit(self, d: dict[str, Any]) -> Hit:
        image = _opt_str(d, "image_path")
        kind = _str(d, "kind")
        if kind not in ("pdf_page", "image", "text"):
            raise _Bad("unknown chunk kind")
        chunk = Chunk(
            rowid=_int(d, "rowid"),
            chunk_id=_str(d, "chunk_id"),
            doc_id=_str(d, "doc_id"),
            path=_str(d, "path"),
            sha256=_str(d, "sha256"),
            kind=kind,
            page=_opt_int(d, "page"),
            chunk_idx=_int(d, "chunk_idx"),
            heading_path=_str(d, "heading_path"),
            text=_str(d, "text"),
            image_path=image,
        )
        rerank = d["rerank_score"]
        return Hit(
            chunk,
            self._absolute(image),
            _float(d, "score"),
            None if rerank is None else _number(rerank),
        )


def _load_check(d: dict[str, Any]) -> Check:
    stage, label, refusal, action = d["stage"], d["label"], d["refusal"], d["action"]
    if (
        stage not in ("prompt", "response")
        or label not in ("Safe", "Controversial", "Unsafe", None)
        or refusal not in ("Yes", "No", None)
        or action not in ("allow", "warn", "block")
    ):
        raise _Bad("unknown check")
    categories = [_as(c, str) for c in _list(d, "categories")]
    return Check(stage, Verdict(label, categories, refusal), action)


# strict readers: a missing key is KeyError, a wrong type _Bad


class _Bad(ValueError):
    pass


def _as[T](value: Any, kind: type[T]) -> T:
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise _Bad(f"expected {kind.__name__}")
    return value


def _obj(value: Any) -> dict[str, Any]:
    return _as(value, dict)


def _list(d: dict[str, Any], key: str) -> list[Any]:
    return _as(d[key], list)


def _str(d: dict[str, Any], key: str) -> str:
    return _as(d[key], str)


def _opt_str(d: dict[str, Any], key: str) -> str | None:
    return None if d[key] is None else _str(d, key)


def _int(d: dict[str, Any], key: str) -> int:
    return _as(d[key], int)


def _opt_int(d: dict[str, Any], key: str) -> int | None:
    return None if d[key] is None else _int(d, key)


def _bool(d: dict[str, Any], key: str) -> bool:
    return _as(d[key], bool)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Bad("expected a number")
    return float(value)


def _float(d: dict[str, Any], key: str) -> float:
    return _number(d[key])
