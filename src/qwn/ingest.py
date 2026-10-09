"""Ingest: files → chunks → vectors → index (PLAN.md → Data flow, Data model, Bad input).

Each file is hashed, rendered or chunked, and embedded outside any transaction, then committed in
one. A file that fails is recorded with its reason and the run moves on to the next one.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qwn.config import Settings
from qwn.index import (
    DocKind,
    Document,
    Index,
    NewChunk,
    doc_id_for,
    page_dir,
)
from qwn.interfaces import Embedder, Item
from qwn.models_lock import pinned

logger = logging.getLogger(__name__)

EXTENSIONS: dict[str, DocKind] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".md": "text",
    ".markdown": "text",
    ".txt": "text",
}
SLIDES = {".pptx", ".ppt", ".key"}
CHARS_PER_TOKEN = 4  # chunk_tokens is approximate: no tokenizer needed to chunk
EMBED_TEXT_CHARS = 2000  # page text sent with a page image (PLAN.md → Data flow)
HEADING_SEP = " \u203a "  # single right-pointing angle quote, as in PLAN.md
BATCH = {"pdf": 8, "image": 8, "text": 32}  # items per embed call
TMP_PREFIX = ".tmp-"

# Settings that change what's stored. A change means a rebuild (PLAN.md → Rebuild trigger).
INGEST_KEYS = ("pdf_dpi", "max_pixels", "chunk_tokens", "chunk_context", "pdf_embed", "embed_dim")


def ingest_hash(settings: Settings) -> str:
    values = {key: getattr(settings, key) for key in INGEST_KEYS}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


def index_meta(settings: Settings) -> dict[str, str]:
    return {
        "embed_model": settings.embed_model,
        "embed_revision": pinned(settings.embed_model).revision,
        "embed_dim": str(settings.embed_dim),
        "ingest_hash": ingest_hash(settings),
    }


def open_index(settings: Settings, root: Path | None = None, *, reset: bool = False) -> Index:
    """Open (creating if needed) the index; raise IndexMismatch if it was built differently."""
    index = Index(root or settings.index_dir, settings.embed_dim)
    if reset:
        index.reset(index_meta(settings))
        shutil.rmtree(index.root / "pages", ignore_errors=True)
    else:
        index.open(index_meta(settings))
    return index


# chunking (md/txt)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def sections(text: str, *, markdown: bool) -> list[tuple[str, str]]:
    """(heading path, body) per markdown section; a heading with no body is skipped."""
    out: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    path, lines, in_fence = "", [], False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body and not (len(lines) and _HEADING.match(lines[0]) and len(body.splitlines()) == 1):
            out.append((path, body))

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = _HEADING.match(line) if markdown and not in_fence else None
        if heading:
            flush()
            level, title = len(heading[1]), heading[2].strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path, lines = HEADING_SEP.join(t for _, t in stack), [line]
        else:
            lines.append(line)
    flush()
    return out


def split(body: str, max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most `max_chars`; split long paragraphs at sentences."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        while len(para) > max_chars:
            # prefer a sentence end, then a space, past halfway; else cut mid-word
            cut = para.rfind(". ", 0, max_chars) + 1
            if cut <= max_chars // 2:
                cut = para.rfind(" ", 0, max_chars)
            if cut <= max_chars // 2:
                cut = max_chars
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 2 + len(piece) <= max_chars:
            chunks[-1] += "\n\n" + piece
        else:
            chunks.append(piece)
    return chunks


def chunk_text(text: str, *, chunk_tokens: int, markdown: bool) -> list[tuple[str, str]]:
    """(heading path, chunk text) for a markdown or plain-text file."""
    max_chars = chunk_tokens * CHARS_PER_TOKEN
    return [
        (path, chunk)
        for path, body in sections(text, markdown=markdown)
        for chunk in split(body, max_chars)
    ]


def context_header(file_name: str, heading_path: str) -> str:
    """E.g. 'pricing.md > Enterprise > SSO': what a md/txt chunk is about, for the models."""
    return HEADING_SEP.join(p for p in (file_name, heading_path) if p)


def embed_input(
    kind: str,
    text: str,
    image_path: Path | None,
    heading_path: str,
    file_name: str,
    settings: Settings,
) -> Item:
    """What the embedder (and the reranker, which mirrors it) sees for one chunk."""
    if kind == "pdf_page":
        page_text = text[:EMBED_TEXT_CHARS] if settings.pdf_embed == "image+text" else ""
        return Item(text=page_text or None, image_path=image_path)
    if kind == "image":
        return Item(image_path=image_path)
    if settings.chunk_context:
        return Item(text=f"{context_header(file_name, heading_path)}\n\n{text}")
    return Item(text=text)


# files


class Rejected(ValueError):
    """A file that can't be indexed; the message is the reason shown to the user."""


class ModelUnavailable(RuntimeError):
    """The embedder couldn't be loaded (in use, not downloaded, no memory): stop the whole run."""


@dataclass(frozen=True)
class Failure:
    path: str
    reason: str


@dataclass
class Report:
    indexed: list[str] = field(default_factory=list)
    unchanged: int = 0
    moved: list[tuple[str, str]] = field(default_factory=list)
    failed: list[Failure] = field(default_factory=list)
    ignored: int = 0  # unsupported files skipped quietly in folder walks
    cancelled: bool = False


Action = Literal["index", "unchanged", "move"]


@dataclass(frozen=True)
class Planned:
    path: Path
    action: Action
    sha256: str
    moved_from: Document | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def collect(paths: list[Path]) -> tuple[list[Path], list[Failure], int]:
    """Files to ingest under `paths`, sorted. Named bad paths fail; folder walks skip quietly."""
    files: set[Path] = set()
    failed: list[Failure] = []
    ignored = 0
    for raw in paths:
        path = raw.expanduser().resolve()
        if path.is_dir():
            for f in path.rglob("*"):
                if any(part.startswith(".") for part in f.relative_to(path).parts):
                    continue
                if f.is_file():
                    if f.suffix.lower() in EXTENSIONS:
                        files.add(f)
                    else:
                        ignored += 1
        elif path.is_file():
            ext = path.suffix.lower()
            if ext in EXTENSIONS:
                files.add(path)
            elif ext in SLIDES:
                failed.append(Failure(str(path), "slides: export to PDF first"))
            else:
                failed.append(Failure(str(path), f"unsupported file type ({ext or 'none'})"))
        else:
            failed.append(Failure(str(path), "not found"))
    return sorted(files), failed, ignored


class Ingester:
    def __init__(self, settings: Settings, index: Index, embedder: Callable[[], Embedder]) -> None:
        self.settings = settings
        self.index = index
        self._embedder = embedder  # called only when something needs embedding
        self.pages = index.root / "pages"

    # planning (no writes, no models)

    def plan(self, files: list[Path]) -> list[Planned]:
        out: list[Planned] = []
        for path in files:
            sha = sha256_file(path)
            existing = self.index.document(str(path))
            if existing is not None and existing.sha256 == sha:
                out.append(Planned(path, "unchanged", sha))
                continue
            moved = None
            if existing is None:
                gone = [d for d in self.index.documents_with_sha(sha) if not Path(d.path).exists()]
                moved = gone[0] if gone else None
            out.append(Planned(path, "move" if moved else "index", sha, moved))
        return out

    # running

    def run(
        self,
        paths: list[Path],
        *,
        progress: Callable[[int, int, Path], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Report:
        files, failed, ignored = collect(paths)
        report = Report(failed=failed, ignored=ignored)
        self._clean_tmp()
        for i, path in enumerate(files):
            if progress is not None:
                progress(i, len(files), path)
            if cancel is not None and cancel.is_set():  # checked right before each file
                report.cancelled = True
                break
            try:
                [planned] = self.plan([path])
                if planned.action == "unchanged":
                    report.unchanged += 1
                elif planned.action == "move" and planned.moved_from is not None:
                    self._move(planned.moved_from, path)
                    report.moved.append((planned.moved_from.path, str(path)))
                else:
                    self._index(path, planned.sha256)
                    report.indexed.append(str(path))
            except ModelUnavailable:
                raise
            except Exception as e:
                reason = str(e) if isinstance(e, Rejected | ValueError) else _error(e)
                logger.warning("ingest failed: %s (%s)", path, type(e).__name__)
                report.failed.append(Failure(str(path), reason))
        if progress is not None and not report.cancelled:
            progress(len(files), len(files), Path())
        logger.info(
            "ingest: %d indexed, %d unchanged, %d moved, %d failed",
            len(report.indexed),
            report.unchanged,
            len(report.moved),
            len(report.failed),
        )
        return report

    def _move(self, old: Document, path: Path) -> None:
        new = self.index.rekey(old, str(path), path.stat().st_mtime)
        src, dst = self.index.root / page_dir(old.doc_id), self.index.root / page_dir(new.doc_id)
        if src.exists():
            shutil.rmtree(dst, ignore_errors=True)
            src.rename(dst)
        logger.info("moved %s -> %s", old.path, path)

    def _index(self, path: Path, sha: str) -> None:
        kind = EXTENSIONS[path.suffix.lower()]
        doc_id = doc_id_for(str(path))
        tmp = self.pages / f"{TMP_PREFIX}{doc_id}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            chunks, items = self._build(path, kind, doc_id, tmp)
            vectors = self._embed(items, kind)
            if sha256_file(path) != sha:
                raise Rejected("changed during ingest")
            doc = Document(
                doc_id,
                str(path),
                sha,
                path.stat().st_mtime,
                kind,
                len(chunks),
                datetime.now(UTC).isoformat(timespec="seconds"),
            )
            self.index.replace(doc, chunks, vectors)
            final = self.index.root / page_dir(doc_id)
            shutil.rmtree(final, ignore_errors=True)
            if tmp.exists():
                tmp.rename(final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        logger.info("indexed %s (%s, %d chunks)", path, kind, len(chunks))

    def _build(
        self, path: Path, kind: DocKind, doc_id: str, tmp: Path
    ) -> tuple[list[NewChunk], list[Item]]:
        s = self.settings
        stored = page_dir(doc_id)
        chunks: list[NewChunk] = []
        items: list[Item] = []
        if kind == "pdf":
            from qwn.adapters import pdf

            try:
                pages = pdf.render(
                    path,
                    tmp,
                    dpi=s.pdf_dpi,
                    max_pages=s.max_pdf_pages,
                    max_image_pixels=s.max_image_pixels,
                )
            except pdf.PdfRejected as e:
                raise Rejected(str(e)) from None
            for p in pages:
                name = p.image_path.name
                chunks.append(NewChunk("pdf_page", p.number, 0, "", p.text, f"{stored}/{name}"))
                items.append(embed_input("pdf_page", p.text, p.image_path, "", path.name, s))
        elif kind == "image":
            from qwn.adapters import images

            dst = tmp / "0.webp"
            try:
                images.normalize(
                    path, dst, max_pixels=s.max_pixels, max_image_pixels=s.max_image_pixels
                )
            except images.ImageRejected as e:
                raise Rejected(str(e)) from None
            chunks.append(NewChunk("image", None, 0, "", "", f"{stored}/0.webp"))
            items.append(embed_input("image", "", dst, "", path.name, s))
        else:
            text = self._read_text(path)
            markdown = path.suffix.lower() in (".md", ".markdown")
            for i, (heading, body) in enumerate(
                chunk_text(text, chunk_tokens=s.chunk_tokens, markdown=markdown)
            ):
                chunks.append(NewChunk("text", None, i, heading, body, None))
                items.append(embed_input("text", body, None, heading, path.name, s))
        return chunks, items

    def _read_text(self, path: Path) -> str:
        size = path.stat().st_size
        if size > self.settings.max_text_bytes:
            raise Rejected(f"text file too large ({size:,} > {self.settings.max_text_bytes:,})")
        data = path.read_bytes()
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            logger.warning("not valid UTF-8, decoding with replacement: %s", path)
            return data.decode("utf-8", errors="replace")

    def _embed(self, items: list[Item], kind: DocKind) -> NDArray[np.float32]:
        if not items:
            return np.zeros((0, self.settings.embed_dim), dtype=np.float32)
        try:
            embedder = self._embedder()
        except Exception as e:
            raise ModelUnavailable(str(e)) from e
        size = BATCH[kind]
        parts = [
            embedder.embed(items[i : i + size], is_query=False) for i in range(0, len(items), size)
        ]
        return np.concatenate(parts).astype(np.float32)

    def _clean_tmp(self) -> None:
        if self.pages.exists():
            for leftover in self.pages.glob(f"{TMP_PREFIX}*"):
                shutil.rmtree(leftover, ignore_errors=True)

    # pruning

    def prune_plan(self, paths: list[Path]) -> "PrunePlan":
        """Documents under `paths` (all, if empty) whose file is gone. Unmounted drives and
        missing folders you named are skipped, not treated as deleted."""
        roots = [p.expanduser().resolve() for p in paths]
        plan = PrunePlan()
        for doc in self.index.documents():
            path = Path(doc.path)
            root = next((r for r in roots if path == r or path.is_relative_to(r)), None)
            if roots and root is None:
                continue
            if path.exists():
                continue
            volume = _volume(path)
            if volume is not None and not volume.exists():
                plan.unmounted[volume.name] += 1
            elif root is not None and not root.exists():
                plan.missing_roots[str(root)] += 1
            else:
                plan.remove.append(doc)
        return plan

    def prune(self, docs: list[Document]) -> None:
        if not docs:
            return
        self.index.delete([d.doc_id for d in docs])
        for d in docs:
            shutil.rmtree(self.index.root / page_dir(d.doc_id), ignore_errors=True)
        if len(docs) >= 100:
            self.index.vacuum()
        logger.info("pruned %d documents", len(docs))


@dataclass
class PrunePlan:
    remove: list[Document] = field(default_factory=list)
    unmounted: Counter[str] = field(default_factory=Counter)  # volume name → documents
    missing_roots: Counter[str] = field(default_factory=Counter)  # folder → documents


def _volume(path: Path) -> Path | None:
    parts = path.parts
    if len(parts) > 2 and parts[1] == "Volumes":
        return Path("/Volumes", parts[2])
    return None


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
