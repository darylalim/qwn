"""SQLite + sqlite-vec index at index_dir/qwn.db (PLAN.md → Data model).

The only module that talks to SQLite. Reads use one connection per thread (WAL allows concurrent
readers); writes go through one in-process lock, and each document is replaced in one transaction.
"""

import hashlib
import re
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import sqlite_vec
from numpy.typing import ArrayLike, NDArray

SCHEMA_VERSION = "1"
DB_NAME = "qwn.db"

DocKind = Literal["pdf", "image", "text"]
ChunkKind = Literal["pdf_page", "image", "text"]

_SCHEMA = """
CREATE TABLE documents (
  doc_id      TEXT PRIMARY KEY,
  path        TEXT NOT NULL UNIQUE,
  sha256      TEXT NOT NULL,
  mtime       REAL NOT NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('pdf','image','text')),
  n_chunks    INTEGER NOT NULL,
  indexed_at  TEXT NOT NULL
);
CREATE INDEX documents_sha ON documents(sha256);

CREATE TABLE chunks (
  rowid        INTEGER PRIMARY KEY,
  chunk_id     TEXT NOT NULL UNIQUE,
  doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  kind         TEXT NOT NULL CHECK (kind IN ('pdf_page','image','text')),
  page         INTEGER,
  chunk_idx    INTEGER NOT NULL,
  heading_path TEXT NOT NULL DEFAULT '',
  text         TEXT NOT NULL DEFAULT '',
  image_path   TEXT
);
CREATE INDEX chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{dim}]);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='rowid',
                                           tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text); END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text); END;
CREATE TRIGGER chunks_au AFTER UPDATE OF text ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text); END;
"""
_TABLES = ("chunks_fts", "vec_chunks", "chunks", "documents", "meta")


class IndexMismatch(RuntimeError):
    def __init__(self, key: str, stored: str, current: str) -> None:
        self.key = key
        super().__init__(
            f"The index was built with a different {key} ({stored!r}, now {current!r}). "
            "Rebuild it with `qwn ingest --reindex PATH...`."
        )


class IndexBusy(RuntimeError):
    def __init__(self) -> None:
        super().__init__("The index is being written by another qwn process. Try again shortly.")


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: str  # absolute
    sha256: str
    mtime: float
    kind: DocKind
    n_chunks: int
    indexed_at: str


@dataclass(frozen=True)
class NewChunk:
    kind: ChunkKind
    page: int | None
    chunk_idx: int
    heading_path: str
    text: str
    image_path: str | None  # relative to the index folder: pages/<doc_id>/<page>.webp


@dataclass(frozen=True)
class Chunk:
    rowid: int
    chunk_id: str
    doc_id: str
    path: str
    sha256: str
    kind: ChunkKind
    page: int | None
    chunk_idx: int
    heading_path: str
    text: str
    image_path: str | None  # relative to the index folder


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def doc_id_for(path: str) -> str:
    return sha256_text(path)


def chunk_id_for(doc_id: str, page: int | None, chunk_idx: int) -> str:
    return sha256_text(f"{doc_id}:{page}:{chunk_idx}")


def page_dir(doc_id: str) -> str:
    return f"pages/{doc_id}"


def is_identifier(word: str) -> bool:
    """Code-like words keyword search is for: PO-48213, E-4031, Q3, 18.4M, AES-256, SSO."""
    core = word.strip("\"'()[]{}<>.,;:!?*")
    return len(re.sub(r"\W", "", core)) >= 2 and (
        any(c.isdigit() for c in core) or "-" in core or (core.isupper() and core.isalpha())
    )


def fts_query(text: str) -> str | None:
    """Question → FTS5 query over its identifier-like words, each quoted, OR-joined.

    Only identifiers: ordinary words would match almost every text chunk and, through RRF, push
    image-only pages (scans, charts) out of the results (PLAN.md → Search, phase 1 finding).
    Quoting makes FTS operators plain words. None if the question has no identifiers.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for word in text.split():
        if not is_identifier(word) or word.lower() in seen:
            continue
        seen.add(word.lower())
        terms.append('"' + word.replace('"', '""') + '"')
    return " OR ".join(terms) or None


def _vec(v: ArrayLike) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


_CHUNK_COLUMNS = (
    "c.rowid, c.chunk_id, c.doc_id, d.path, d.sha256, c.kind, c.page, c.chunk_idx, "
    "c.heading_path, c.text, c.image_path"
)


class Index:
    def __init__(self, root: Path, dim: int) -> None:
        self.root = root  # the index folder: qwn.db + pages/
        self.path = root / DB_NAME
        self.dim = dim
        self._local = threading.local()
        self._write_lock = threading.Lock()

    # connections

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.root.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, isolation_level=None)  # explicit transactions
            if not hasattr(conn, "enable_load_extension"):
                conn.close()
                raise RuntimeError(
                    "This Python's sqlite3 can't load extensions, which sqlite-vec needs. "
                    "Use uv's managed Python (pyproject sets python-preference = only-managed): "
                    "rm -rf .venv && uv sync"
                )
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close this thread's connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @contextmanager
    def _write(self) -> Generator[sqlite3.Connection]:
        with self._write_lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as e:
                if "locked" in str(e) or "busy" in str(e):
                    raise IndexBusy from None
                raise
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # schema and meta

    def open(self, meta: dict[str, str]) -> None:
        """Create the schema if it's missing, else check every `meta` key against the stored one."""
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        if not exists:
            self._create(meta)
            return
        stored = self.meta()
        for key, value in meta.items():
            if stored.get(key) != value:
                raise IndexMismatch(key, stored.get(key, "<missing>"), value)

    def reset(self, meta: dict[str, str]) -> None:
        """Drop every table and recreate an empty index (`--reindex`)."""
        with self._write() as c:
            for table in _TABLES:
                c.execute(f"DROP TABLE IF EXISTS {table}")
        self._create(meta)

    def _create(self, meta: dict[str, str]) -> None:
        with self._write() as c:
            for statement in _split(_SCHEMA.format(dim=self.dim)):
                c.execute(statement)
            rows = {**meta, "schema_version": SCHEMA_VERSION}
            c.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", rows.items())

    def meta(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT key, value FROM meta").fetchall())

    # documents

    def document(self, path: str) -> Document | None:
        row = self.conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
        return Document(*row) if row else None

    def documents(self) -> list[Document]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY path").fetchall()
        return [Document(*row) for row in rows]

    def documents_with_sha(self, sha256: str) -> list[Document]:
        rows = self.conn.execute(
            "SELECT * FROM documents WHERE sha256 = ? ORDER BY path", (sha256,)
        ).fetchall()
        return [Document(*row) for row in rows]

    def replace(self, doc: Document, chunks: list[NewChunk], vectors: NDArray[np.float32]) -> None:
        """Replace `doc` and all its chunks and vectors in one transaction."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._write() as c:
            _delete(c, [doc.doc_id])
            c.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", _doc_row(doc))
            for chunk, vector in zip(chunks, vectors, strict=True):
                cur = c.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, kind, page, chunk_idx, heading_path, "
                    "text, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_id_for(doc.doc_id, chunk.page, chunk.chunk_idx),
                        doc.doc_id,
                        chunk.kind,
                        chunk.page,
                        chunk.chunk_idx,
                        chunk.heading_path,
                        chunk.text,
                        chunk.image_path,
                    ),
                )
                c.execute(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, _vec(vector)),
                )

    def rekey(self, old: Document, new_path: str, mtime: float) -> Document:
        """A moved file: new path and doc_id, same chunks (rowids kept, so vectors stay put)."""
        new = Document(
            doc_id_for(new_path),
            new_path,
            old.sha256,
            mtime,
            old.kind,
            old.n_chunks,
            old.indexed_at,
        )
        with self._write() as c:
            c.execute("PRAGMA defer_foreign_keys = ON")  # chunks point at the old id until moved
            c.execute(
                "UPDATE documents SET doc_id = ?, path = ?, mtime = ? WHERE doc_id = ?",
                (new.doc_id, new.path, new.mtime, old.doc_id),
            )
            old_dir, new_dir = page_dir(old.doc_id), page_dir(new.doc_id)
            rows = c.execute(
                "SELECT rowid, page, chunk_idx, image_path FROM chunks WHERE doc_id = ?",
                (old.doc_id,),
            ).fetchall()
            for rowid, page, chunk_idx, image_path in rows:
                if image_path is not None and image_path.startswith(old_dir + "/"):
                    image_path = new_dir + image_path[len(old_dir) :]
                c.execute(
                    "UPDATE chunks SET doc_id = ?, chunk_id = ?, image_path = ? WHERE rowid = ?",
                    (new.doc_id, chunk_id_for(new.doc_id, page, chunk_idx), image_path, rowid),
                )
        return new

    def delete(self, doc_ids: list[str]) -> None:
        with self._write() as c:
            _delete(c, doc_ids)

    # search

    def vector_search(self, query: NDArray[np.float32], k: int) -> list[tuple[int, float]]:
        """(rowid, L2 distance) of the `k` nearest chunks, nearest first."""
        rows = self.conn.execute(
            "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (_vec(query), k),
        ).fetchall()
        return [(int(r), float(d)) for r, d in rows]

    def keyword_search(self, query: str, k: int) -> list[int]:
        """Rowids matching `query` (an FTS5 expression from `fts_query`), best bm25 first."""
        rows = self.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) "
            "LIMIT ?",
            (query, k),
        ).fetchall()
        return [int(r) for (r,) in rows]

    def chunks(self, rowids: list[int]) -> dict[int, Chunk]:
        if not rowids:
            return {}
        marks = ",".join("?" * len(rowids))
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN documents d USING (doc_id) "
            f"WHERE c.rowid IN ({marks})",
            rowids,
        ).fetchall()
        return {row[0]: Chunk(*row) for row in rows}

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN documents d USING (doc_id) "
            "WHERE c.doc_id = ? ORDER BY c.page, c.chunk_idx",
            (doc_id,),
        ).fetchall()
        return [Chunk(*row) for row in rows]

    # maintenance

    def counts(self) -> dict[str, int]:
        c = self.conn
        out = dict(c.execute("SELECT kind, COUNT(*) FROM documents GROUP BY kind").fetchall())
        out["documents"] = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        out["chunks"] = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        out["vectors"] = c.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        return out

    def vacuum(self) -> None:
        with self._write_lock:
            self.conn.execute("VACUUM")


def _doc_row(d: Document) -> tuple[str, str, str, float, str, int, str]:
    return (d.doc_id, d.path, d.sha256, d.mtime, d.kind, d.n_chunks, d.indexed_at)


def _delete(c: sqlite3.Connection, doc_ids: list[str]) -> None:
    # vec0 is a virtual table: no cascade, so its rows go first. chunks_fts follows the cascade
    # through the chunks_ad trigger.
    for doc_id in doc_ids:
        c.execute(
            "DELETE FROM vec_chunks WHERE rowid IN (SELECT rowid FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        c.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def _split(script: str) -> list[str]:
    """Split the schema into statements; trigger bodies end with 'END;'."""
    statements, buf = [], ""
    for line in script.strip().splitlines():
        buf += line + "\n"
        stripped = buf.strip()
        if stripped.endswith(";") and (not stripped.startswith("CREATE TRIGGER") or "END;" in line):
            statements.append(stripped)
            buf = ""
    return statements
