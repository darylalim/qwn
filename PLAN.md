# Plan: Local multimodal RAG + voice assistant (M2 Max, 32 GB, all-MLX)

> Status: **final draft for implementation** (planned 2026-10-04). Implement in a fresh session,
> phase by phase. Library facts below were verified against PyPI and GitHub on 2026-10-04. Phase 0
> re-checks them against the installed versions before any feature work.

## Goal

Ask questions, typed (v1) or spoken (phase 4), over your own PDFs, slides, screenshots, images and
markdown. Answers come from **Qwen3-VL-8B**, which reads the actual page images and cites the source
file and page. Prompts and answers pass through Qwen3Guard. Everything runs locally on Apple Silicon
with MLX, through a **Typer CLI** and a **Streamlit UI** that share the same `qwn.*` core.

## Decisions

| Topic | Decision |
|---|---|
| Corpus | PDFs/slides (page images + text), images/screenshots, markdown/text. **No video.** |
| Voice | Deferred to phase 4. v1 is text-only. |
| Interface | Typer CLI (ingest/eval/scripting) **and** Streamlit (daily use). Both are thin layers over `qwn.*`. |
| Model I/O | Our own `Protocol` interfaces with exact types; library calls hidden behind adapters. |
| Libraries | `mlx-vlm` for generation, embedding **and** reranking; `mlx-lm` for Guard; `mlx-embeddings` as fallback (**GPL-3.0**: needs a license decision before it's adopted). PDF via **pypdfium2** (not PyMuPDF, which is AGPL). |
| License | **Apache-2.0** (`LICENSE` + `license`/`license-files` in `pyproject.toml`, already applied). Runtime dependencies must be permissive (see License). |
| Index | SQLite + `sqlite-vec` in one file (`index/qwn.db`, WAL), tables `documents`/`chunks`/`vec_chunks`/`meta`; each file re-indexed in one transaction, keyed by content hash. |
| Page renders | WebP q85 at 150 dpi on disk under `index/pages/`. |
| UI | Streamlit app with 3 pages: Chat, Library, System. |
| CLI | `ingest`, `search`, `ask`, `eval`, `status`, `ui`. |
| Testing | 3 tiers + fakes: unit, integration, UI (AppTest), plus `slow` real-model tests. |
| Eval | recall@1/5/10 + MRR (with/without rerank), latency by stage, citation hit. JSON results + committed baseline. |
| Config | `pydantic-settings`: defaults → `qwn.toml` → `QWN_*` env → CLI flags / UI sliders. |
| Tooling | uv (Python 3.12), ruff, ty, pytest (already set up). |
| Claude Code hooks | 5 project hooks in `.claude/settings.json`: format+lint on edit, protected paths, uv-only and heavy-command confirmation, stop-time quality gate, session orientation. |
| CI | GitHub Actions, one job on `macos-15` (arm64): `uv sync --locked`, ruff format/check, ty, `pytest -m "not slow"` (including hook tests), `HF_HUB_OFFLINE=1`. Slow tests run locally before merging. |
| Releases | Push to `main` changing `pyproject.toml` with an untagged version → CI gate → tag `vX.Y.Z` + GitHub Release (notes, wheel, sdist). No PyPI. Bump with `uv version --bump`; one minor release per phase. |

## Stack and memory budget

| Role | Repo (`mlx-community/…`) | Library | Weights |
|---|---|---|---|
| Generator | `Qwen3-VL-8B-Instruct-4bit` | `mlx-vlm` | 5.8 GB |
| Embedder | `Qwen3-VL-Embedding-2B-8bit` | `mlx-vlm` (fallback `mlx-embeddings`) | 2.7 GB |
| Reranker | `Qwen3-VL-Reranker-2B-8bit` | `mlx-vlm` (fallback `mlx-embeddings`) | 2.7 GB |
| Guard | `Qwen3Guard-Gen-0.6B-MLX` | `mlx-lm` | 1.2 GB |
| ASR (phase 4) | `Qwen3-ASR-1.7B-8bit` | `mlx-audio` | 2.5 GB |
| TTS (phase 4) | `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | `mlx-audio` | 2.9 GB |
| **Total** | | | **≈ 18 GB weights + 3–5 GB activations/KV ≈ 22 GB** |

- Text-only v1 is about 13 GB. ASR/TTS are lazy-loaded in phase 4.
- Raise the GPU wired limit if needed: `sudo sysctl iogpu.wired_limit_mb=26000` (resets on reboot).

### Runtime dependencies (versions as of 2026-10-04)

| Package | Version floor | Notes |
|---|---|---|
| `mlx-vlm` | `>=0.7.4` | Pulls `mlx>=0.32.2`, `transformers>=5.14`, **and `mlx-audio>=0.5.2`** |
| `mlx-lm` | `>=0.32.0` | `transformers>=5.7`, compatible with the above |
| `sqlite-vec` | `>=0.1.9` | Vector search inside SQLite (0.2 MB wheel). SQLite itself ships with Python |
| `pypdfium2` | `>=5.14` | PDF → page image + text (BSD-3/Apache-2.0; PDFium, Chrome's PDF engine) |
| `pillow` | latest | Image loading/resizing, WebP encoding |
| `typer` | `>=0.27` | CLI |
| `streamlit` | `>=1.65` | UI (`numpy<3`, OK) |
| `pydantic-settings` | latest | Config, TOML + env |
| `mlx-embeddings` | `==0.1.0` | **Fallback only, GPL-3.0**: added only if mlx-vlm embed/rerank fails phase 0 **and** the user accepts GPL for the project (otherwise write a minimal in-house adapter) |

These pins overlap, so a single environment should resolve. Phase 0 confirms it with
`uv lock`.

## Data flow

```
INGEST   PDF ─► pypdfium2 ─► page WebP (150 dpi) + page text ─┐
         image / screenshot ─► normalised WebP copy ─────────┼─► Embedder ─► 1024-d, L2-norm ─► SQLite + sqlite-vec
         .md / .txt ─► ~800-token chunks (heading-aware) ───┘

QUERY    question ─► Guard.check_prompt ─► Embedder(is_query) ─► top_k=50
                    ─► Reranker ─► rerank_k=5 ─► Generator (≤ max_images page images + text)
                    ─► answer with [S#] citations ─► Guard.check_response ─► rendered "[file p.N]"
```

## Model interfaces (`src/qwn/interfaces.py`)

The rest of the code depends only on these. Adapters live in `src/qwn/adapters/`, fakes in
`tests/fakes.py`.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class Item:                        # one thing to embed or rerank
    text: str | None = None
    image_path: Path | None = None # at least one of text/image_path is set

@dataclass(frozen=True)
class Source:                      # a retrieved, reranked chunk handed to the generator
    label: str                     # "S1".."S5"
    chunk_id: str
    path: str
    page: int | None               # 1-based for PDFs, None otherwise
    text: str
    image_path: Path | None
    score: float

@dataclass(frozen=True)
class Answer:
    text: str                      # raw model text containing [S#] markers
    cited: list[str]               # labels actually cited, in order of first appearance
    prompt_tokens: int
    completion_tokens: int

@dataclass(frozen=True)
class Verdict:
    label: Literal["Safe", "Controversial", "Unsafe"] | None   # None = unparseable
    categories: list[str] = field(default_factory=list)
    refusal: Literal["Yes", "No"] | None = None                # response checks only

class Embedder(Protocol):
    model_id: str
    dim: int                       # output dim after truncation
    def embed(self, items: list[Item], *, is_query: bool) -> NDArray[np.float32]: ...
    # returns shape (len(items), dim), each row L2-normalised

class Reranker(Protocol):
    model_id: str
    def score(self, query: Item, docs: list[Item]) -> list[float]: ...
    # one score per doc; higher = more relevant; deterministic

class Generator(Protocol):
    model_id: str
    def answer(self, question: str, sources: list[Source]) -> Answer: ...

class Guard(Protocol):
    model_id: str
    def check_prompt(self, text: str) -> Verdict: ...
    def check_response(self, prompt: str, response: str) -> Verdict: ...
```

### Adapter notes (verified 2026-10-04; re-check in phase 0)

- **Generator (`mlx-vlm`)**: `model, processor = mlx_vlm.load(repo)`,
  `config = mlx_vlm.utils.load_config(repo)`,
  `prompt = mlx_vlm.prompt_utils.apply_chat_template(processor, config, prompt_or_messages, num_images=n)`,
  `mlx_vlm.generate(model, processor, prompt, image=[paths...], verbose=False)`.
  Check the return type (string or result object with token counts) and the sampling keyword names
  in the installed version.
- **Embedder / Reranker (`mlx-vlm`)**: mlx-vlm 0.7.4 has `embedding_loader.load_embedding_model`
  and `reranker_loader.load_reranker`, and its server lists **Qwen3-VL-Embedding** and
  **Qwen3-VL rerankers** (inputs: objects with `text` / `image`). The Python entry points aren't
  documented. In phase 0, read `mlx_vlm/embedding_loader.py`, `reranker.py` and the `/v1/embeddings`
  and `/v1/rerank` handlers in `mlx_vlm/server/`, then call the same functions in-process.
- **Fallback (`mlx-embeddings` 0.1.0)**: `model, processor = mlx_embeddings.load(repo)`.
  Embedding: `model.process([{"text"|"image"|"instruction": ...}], processor=processor)` →
  `(n, 2048)`. Reranking: `model.process({"instruction", "query", "documents": [...]},
  processor=processor)` → `(n,)`. It uses last-token pooling and scores reranking as yes minus no
  logit. **Do not** follow the README snippets on the mlx-community Embedding/Reranker repos:
  they're copy-paste SigLIP boilerplate.
- **Guard (`mlx-lm`)**: `model, tok = mlx_lm.load("mlx-community/Qwen3Guard-Gen-0.6B-MLX")`.
  Build the prompt with `tok.apply_chat_template(messages, tokenize=False)`, **no generation
  prompt**, which is what the official card does. Prompt check: `[user]`. Response check:
  `[user, assistant]`. Decode greedily with `max_tokens=128`.
- **Matryoshka truncation**: take the first `embed_dim` (default 1024) values, then re-normalise
  with L2. This happens in the adapter, so `Embedder.dim` is what callers see.
- **Determinism**: Reranker and Guard must give the same output for the same input (single scoring
  pass / greedy decoding). The `slow` contract tests check this.

### Prompts and parameters (`src/qwn/prompts.py`)

- **Embedding instructions**
  - query: `"Retrieve images or text relevant to the user's query."`
  - document: no instruction (the model's default is `"Represent the user's input."`)
- **Reranker instruction**: `"Given a question, judge whether this page, image or passage helps answer it."`
- **Generator system prompt**:
  ```
  You answer questions using only the provided sources. Each source is labelled [S1]..[Sn]
  and may be a page image, an image, or a text passage. Cite every claim with its label,
  e.g. "Revenue grew 12% [S2]." If the sources don't contain the answer, say so plainly.
  Do not invent sources or page numbers.
  ```
  User turn: the page images in source order, then a text block of `[S#] <path> p.<page>: <text excerpt ≤ 1500 chars>` lines, then `Question: ...`.
- **Generation parameters**: start with Qwen3-VL-Instruct's recommended sampling
  (temperature 0.7, top_p 0.8, top_k 20; check the card in phase 0). `max_tokens=1024`. Context
  capped at `max_context=16384` tokens.
- **Citations**: the model cites `[S#]`. `qwn.answer.parse_citations` extracts the labels, and the
  UI/CLI renders them as `[file p.N]`. Labels the model makes up (not in `sources`) are dropped and
  logged.
- **Guard parsing** (from the official card):
  - `Safety: (Safe|Unsafe|Controversial)`
  - categories `(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)` (Jailbreak applies to prompts only)
  - `Refusal: (Yes|No)` (responses only)
  - If the output can't be parsed: `label=None`. Treat this as **Controversial** (warn) and log it.

## Storage layout

| What | Where | Size guide | Git |
|---|---|---|---|
| Models | `~/.cache/huggingface/hub` (shared, outside the repo) | ~13 GB v1, ~18 GB with voice | — |
| Source files | Streamlit uploads are copied to `data_dir`. CLI ingest indexes files **in place** (absolute paths) | your corpus | ignored |
| Index database | `index_dir/qwn.db` (single SQLite file, WAL mode) | ~4 KB per chunk for vectors + text | ignored |
| Page renders | `index_dir/pages/<doc_id>/<page>.webp` (WebP q85, 150 dpi) | ~50–150 KB per page | ignored |
| Eval runs | `eval/results/*.json`; `eval/baseline.json` is committed | KB | results ignored |

Back up or move the index by copying `index_dir/`. Delete it to start fresh.

### `.gitignore` conventions (applied 2026-10-04)

- **Anchor root-only paths with a leading `/`** (`/data/`, `/index/`, `/eval/results/`,
  `/qwn.toml`, `/dist/`, `/.venv/`). Without the anchor, `data/` also matches `tests/data/` and
  `src/qwn/data/`, and those files are silently left out of commits. Use unanchored patterns only
  for things that can appear anywhere (`__pycache__/`, `*.log`, `.DS_Store`).
- **Ignore the real file, commit an example:** `qwn.toml` / `qwn.example.toml`, `.env*` /
  `!.env.example`.
- **Shared vs personal Claude Code files:** commit `.claude/settings.json` and `.claude/hooks/`;
  ignore `.claude/settings.local.json` and `CLAUDE.local.md`.
- **Editor folders** (`.vscode/`, `.idea/`) belong in each developer's global ignore
  (`~/.config/git/ignore`), not here.
- **Check new rules before committing:** `git check-ignore -v <path>`, especially for anything
  under `tests/` or `src/`. The current file was tested against 34 paths (20 to ignore, 14 to keep);
  all behaved as intended.

## Data model (SQLite + sqlite-vec at `index_dir/qwn.db`)

```sql
PRAGMA journal_mode = WAL;      -- CLI and Streamlit can read while one writes
PRAGMA foreign_keys = ON;

CREATE TABLE documents (
  doc_id      TEXT PRIMARY KEY,           -- sha256(abs path)
  path        TEXT NOT NULL UNIQUE,       -- absolute path
  sha256      TEXT NOT NULL,              -- content hash
  mtime       REAL NOT NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('pdf','image','text')),
  n_chunks    INTEGER NOT NULL,
  indexed_at  TEXT NOT NULL               -- ISO-8601
);

CREATE TABLE chunks (
  rowid       INTEGER PRIMARY KEY,        -- joins to vec_chunks.rowid
  chunk_id    TEXT NOT NULL UNIQUE,       -- sha256(f"{doc_id}:{page}:{chunk_idx}")
  doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  kind        TEXT NOT NULL CHECK (kind IN ('pdf_page','image','text')),
  page        INTEGER,                    -- 1-based for PDF pages, else NULL
  chunk_idx   INTEGER NOT NULL,           -- 0 for pages/images
  text        TEXT NOT NULL DEFAULT '',   -- page text / chunk text / '' for images
  image_path  TEXT                        -- index_dir/pages/<doc_id>/<page>.webp
);
CREATE INDEX chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[1024]);   -- rowid = chunks.rowid

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- keys: embed_model, embed_dim, schema_version

-- Later (hybrid search): CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='rowid');
```

- **PDF text**: `page.get_textpage().get_text_range()` (pypdfium2). Checked on 2026-10-04 with a
  generated 2-page PDF: text came back exact; a 150 dpi render is 1275×1651 px, ~8 KB as WebP q85.
- **Library**: `sqlite-vec>=0.1.9`. Load it with `sqlite_vec.load(conn)` after
  `conn.enable_load_extension(True)`. Checked: the uv Python 3.12 build has SQLite 3.53 with
  extension loading enabled. The `vec0` dimension comes from `meta.embed_dim` when the schema is
  created.
- **Virtual tables don't cascade.** `ON DELETE CASCADE` cleans up `chunks`, but `vec_chunks` rows
  must be deleted explicitly (`DELETE FROM vec_chunks WHERE rowid IN (SELECT rowid FROM chunks
  WHERE doc_id = ?)`) **before** the chunks are deleted.
- **Re-indexing one document is a single transaction**: render and embed *outside* the transaction
  (slow), then `BEGIN; delete vec rows; delete chunks; insert chunks + vec rows; upsert documents;
  COMMIT;`. A crash leaves either the old or the new version, never a mix. Write the page renders to
  a temporary folder and rename it into place after the commit.
- **Ingest algorithm**: walk the paths, skipping anything not in {pdf, png, jpg, jpeg, webp, md,
  txt}. For each file, compute sha256. If it matches `documents.sha256`, skip. Otherwise render or
  chunk it, embed in batches (8 images or 32 texts), and commit as above.
- **Images**: re-encode each image to WebP q85 in `index_dir/pages/<doc_id>/0.webp`, keeping the
  aspect ratio, capped at `max_pixels`. PDF pages: pypdfium2 `page.render(scale=pdf_dpi / 72).to_pil()` → Pillow →
  WebP q85.
- **Removed files**: `qwn ingest --prune` deletes rows and render folders for files that no longer
  exist.
- **Rebuild trigger**: if `meta.embed_model`, `meta.embed_dim` or `meta.schema_version` differs
  from the settings, stop with a message suggesting `--reindex` (drops and recreates all tables
  and `pages/`).
- **Search**:
  `SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?` (brute-force KNN),
  then join `chunks`. Vectors are L2-normalised, so the default L2 distance ranks exactly like
  cosine. Report `cosine = 1 - distance**2 / 2`. Brute force is fine to ~100k chunks.
- **Maintenance**: run `VACUUM` after `--reindex` or a large `--prune`.
- **Fallback** (if sqlite-vec is a problem): store the embedding as a float32 BLOB column on
  `chunks` and search with an in-memory numpy matrix. Reload it when `meta.index_version` (bumped
  on every commit) changes. Only `index.py` changes.

## CLI (`src/qwn/cli.py`, Typer, entry point `qwn`)

```
qwn ingest PATH... [--reindex] [--prune] [--dry-run]
qwn search QUERY [-k 5] [--no-rerank] [--json]
qwn ask QUESTION [--sources] [--no-guard] [--json]
qwn eval [--file eval/queries.jsonl] [--no-rerank] [--out eval/results/]
qwn status            # index stats, embed model/dim, loaded models, MLX memory
qwn ui                # runs: streamlit run src/qwn/ui/app.py
```

Common flags override settings (`--top-k`, `--rerank-k`, `--max-images`, `--config PATH`).

## Streamlit UI (`src/qwn/ui/`)

Run as an `st.navigation` app with 3 pages. Models are loaded through `@st.cache_resource` wrappers
around `qwn.models`. The UI has no model or retrieval logic of its own.

- **Chat**
  - History in `st.chat_message`.
  - `st.status` shows the steps: Guard → Retrieving → Reranking → Answering → Guard.
  - Each `[file p.N]` citation opens an `st.expander` with the page thumbnail, rerank score and
    text excerpt.
  - Guard results: Controversial → `st.warning`, Unsafe → a blocked-message bubble.
- **Library**
  - `st.file_uploader` (pdf/png/jpg/jpeg/webp/md/txt) saves files into `data_dir`, then ingests
    them with `st.progress`.
  - `st.dataframe` lists documents, with Delete and Re-index buttons.
- **System**
  - Shows loaded models and MLX active/peak memory.
  - Sliders for `top_k`, `rerank_k`, `max_images`, plus guard toggles. Values live in
    `st.session_state` and are applied per request (they don't persist).
- **State**: `st.session_state.messages`, `st.session_state.settings`. Single local user.

## Settings (`src/qwn/config.py`)

```python
class Settings(BaseSettings):
    data_dir: Path = Path("data")
    index_dir: Path = Path("index")
    gen_model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    embed_model: str = "mlx-community/Qwen3-VL-Embedding-2B-8bit"
    rerank_model: str = "mlx-community/Qwen3-VL-Reranker-2B-8bit"
    guard_model: str = "mlx-community/Qwen3Guard-Gen-0.6B-MLX"
    embed_dim: int = 1024
    top_k: int = 50
    rerank_k: int = 5
    max_images: int = 4
    max_pixels: int = 1_003_520          # 1280 * 28 * 28
    max_context: int = 16_384
    max_tokens: int = 1024
    pdf_dpi: int = 150
    chunk_tokens: int = 800
    guard_enabled: bool = True
    controversial: Literal["warn", "block"] = "warn"
    model_config = SettingsConfigDict(env_prefix="QWN_", toml_file="qwn.toml")
    # Override settings_customise_sources to include TomlConfigSettingsSource. Setting
    # toml_file alone doesn't load the TOML file.
```

Precedence: CLI flags / UI sliders (passed as init kwargs) > `QWN_*` env > `qwn.toml` > defaults.
Commit an example `qwn.example.toml`; `qwn.toml` is gitignored.

## Testing

```
tests/fakes.py       FakeEmbedder (hash-seeded unit vectors), FakeReranker (word overlap),
                     FakeGenerator (cites S1), FakeGuard (keyword rules)
tests/unit/          chunking, IDs/hashing, guard regex parsing (incl. unparseable),
                     citation parsing (incl. invented labels), prompt building,
                     MRL truncate + renorm, settings precedence
tests/integration/   tmp_path SQLite index (real sqlite-vec); 2-page text PDF from `tests/pdf_fixture.py` (dependency-free writer, see License); PIL image; markdown file
                     ingest → re-ingest skips → modify one file → only it re-indexes → --prune;
                     crash between embed and commit leaves the old version intact
tests/ui/            streamlit.testing.v1.AppTest smoke test per page, with fakes injected
tests/hooks/         run each .claude/hooks/*.sh via subprocess with JSON payloads; assert the
                     allow/deny/ask decisions and exit codes from the hooks section's test list
tests/slow/          @pytest.mark.slow real-model contract tests: shapes, norms, determinism,
                     guard on known safe/unsafe prompts, mlx-vlm vs reference embedding (cos ≥ 0.99)
```

- Default `uv run pytest -m "not slow"`: runs in under 10 s with no downloads. `uv run pytest`
  runs everything.
- Inject fakes through a `qwn.models.Registry` that accepts overrides. Avoid monkeypatching
  library internals.

## Evaluation (`src/qwn/eval.py`, `qwn eval`)

- `eval/queries.jsonl`: about 20 hand-written questions over your own corpus. Paths are relative
  to `data_dir`, pages are 1-based. Example:
  ```json
  {"id": "q01", "query": "...", "expected": [{"path": "docs/x.pdf", "page": 3}], "tags": ["chart", "pdf"]}
  ```
- Metrics: recall@1/5/10 and MRR, **embedding only vs + rerank**; latency p50/p95 for embed,
  search, rerank and generate; **citation_hit** (does the answer cite an expected page), only when
  generation is enabled.
- Output: a table to stdout, the full JSON to `eval/results/<timestamp>.json` (gitignored), and
  `eval/baseline.json` (committed, updated on purpose with `--update-baseline`). The report shows
  the difference from the baseline.

## Phases

| # | Phase | Deliverable | Exit criterion |
|---|---|---|---|
| 0 | Env + checks | **Install and test the Claude Code hooks first**; add `ci.yml`, `release.yml`, `[tool.uv] required-version` and `tests/hooks/` (`extend-exclude` for `*.md` is already done); add runtime deps; `interfaces.py`; adapters; `scripts/smoke_test.py`; `tests/slow/` contract tests | `uv lock` resolves; 4 core models loaded together under 24 GB; VL-8B ≥ 35 tok/s; slow tests pass (or the fallback library is adopted) |
| 1 | Retrieval | `config`, `ingest`, `index` (SQLite schema + sqlite-vec), `retrieve`; `qwn ingest/search/status`; unit + integration tests | recall@5 ≥ 0.8 on 20 queries; rerank beats embedding alone |
| 2 | Answering | `prompts`, `answer`; `qwn ask`; `qwn eval` with citation_hit | Correct page cited in most of the 20 queries |
| 3 | Safety | `guard` wired into ask/chat; controversial policy | Known unsafe prompts blocked; normal prompts pass; unparseable output → warn |
| 4 | Voice | ASR/TTS adapters (mlx-audio already installed via mlx-vlm); `st.audio_input` + playback | About 3 s or less from end of speech to first audio |
| 5 | Streamlit UI | 3-page app; AppTest smoke tests; `qwn ui` | Full flow usable from the browser |

Phase 5 may begin once phase 2 is done; the Chat page doesn't need Guard or voice.

## Repo layout (target)

```
qwn/
├─ .claude/                 # settings.json (hooks) + hooks/*.sh, committed
├─ .github/workflows/       # ci.yml (macOS arm64, reusable), release.yml
├─ pyproject.toml           # uv; runtime deps above; dev group: ruff, ty, pytest
├─ PLAN.md   qwn.example.toml
├─ src/qwn/
│  ├─ interfaces.py         # Protocols + dataclasses (above)
│  ├─ config.py             # Settings
│  ├─ models.py             # lazy Registry (with test overrides) + memory logging
│  ├─ adapters/             # mlx_vlm_gen.py, mlx_vlm_embed.py, mlx_vlm_rerank.py, mlx_lm_guard.py
│  ├─ prompts.py  ingest.py  index.py  retrieve.py  answer.py  guard.py  eval.py
│  ├─ cli.py
│  └─ ui/                   # app.py, pages/chat.py, pages/library.py, pages/system.py
├─ scripts/smoke_test.py
├─ eval/queries.jsonl  eval/baseline.json
├─ tests/                   # fakes.py, unit/, integration/, ui/, hooks/, slow/
└─ data/  index/            # gitignored
```

## License

**Apache-2.0** (decided 2026-10-04, already applied: `LICENSE` holds the official text, and
`pyproject.toml` has `license = "Apache-2.0"` and `license-files = ["LICENSE"]`; the built wheel
carries `License-Expression: Apache-2.0`).

- **Why:** permissive with an explicit patent grant. It matches the Qwen models, Streamlit and
  transformers; everything else in the stack is MIT/BSD.
- **Dependency license audit (2026-10-04):**

  | License | Packages |
  |---|---|
  | MIT | mlx, mlx-lm, mlx-vlm, mlx-audio, typer, pydantic-settings |
  | Apache-2.0 | streamlit, transformers, all Qwen models used (VL-8B, Embedding, Reranker, Guard, ASR, TTS) |
  | MIT or Apache-2.0 | sqlite-vec |
  | BSD-3 / Apache-2.0 | pypdfium2 |
  | Permissive (MIT-CMU, BSD, …) | pillow, numpy |
  | **Excluded: AGPL-3.0** | PyMuPDF. Replaced by pypdfium2 so the combined program stays permissive |
  | **Fallback only: GPL-3.0** | mlx-embeddings. Needs the user's decision before adoption |

- **Rule for new dependencies:** check the license before `uv add`. Anything copyleft (GPL, AGPL,
  LGPL with static linking, SSPL) needs the user's explicit OK and an update to this table.
- **Model weights** aren't redistributed; they're downloaded from Hugging Face at runtime under
  their own Apache-2.0 terms. Phase 0 adds a README "Models and licenses" section crediting them.
- **Test PDF fixtures** come from `tests/pdf_fixture.py`, a ~35-line dependency-free writer for
  multi-page text PDFs (Helvetica, one line per page). It was prototyped on 2026-10-04: pypdfium2
  parsed it, rendered it and extracted the text exactly.

## Claude Code hooks

Install these first, at the start of phase 0, before any code is written. They run in every
implementation session and make the plan's rules automatic instead of relying on memory.

| # | Event (matcher) | Script | What it enforces | On violation |
|---|---|---|---|---|
| H1 | `PostToolUse` (`Write\|Edit\|MultiEdit`) | `py-format.sh` | Every edited `.py` is `ruff format`ted and `ruff check --fix`ed immediately | Exit 2: lint errors ruff can't fix are reported back to Claude to fix |
| H2 | `PreToolUse` (`Write\|Edit\|MultiEdit`) | `protect-paths.sh` | No hand edits to generated or user-owned paths: `uv.lock`, `.venv/`, `data/`, `index/`, `eval/baseline.json`, `.streamlit/secrets.toml` | `deny` with the right command to use instead |
| H3 | `PreToolUse` (`Bash`) | `guard-bash.sh` | uv only (no `pip install`); confirm before anything that downloads or loads ~13 GB of models, starts Streamlit, or deletes `data/`/`index/` | `deny` (pip) / `ask` (heavy or destructive) |
| H4 | `Stop` | `quality-gate.sh` | If `.py`/`pyproject.toml` changed: `ruff format --check`, `ruff check`, `ty check`, `pytest -m "not slow"` must pass before Claude finishes | Exit 2: Claude keeps working with the failure output |
| H5 | `SessionStart` (`startup`) | `session-context.sh` | A fresh session starts knowing the branch, recent commits and "follow PLAN.md" | — (stdout becomes context) |

Exit-code meaning: 0 = OK. **2 = blocking**: stderr goes to Claude; for PreToolUse the tool call is
blocked, and for Stop Claude can't finish. Any other code = non-blocking error shown to the user.
PreToolUse hooks may instead print JSON with
`hookSpecificOutput.permissionDecision` = `allow` / `deny` / `ask`.

### `.claude/settings.json` (committed)

```json
{
  "hooks": {
    "SessionStart": [
      { "matcher": "startup",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/session-context.sh" }] }
    ],
    "PreToolUse": [
      { "matcher": "Write|Edit|MultiEdit",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect-paths.sh" }] },
      { "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/guard-bash.sh" }] }
    ],
    "PostToolUse": [
      { "matcher": "Write|Edit|MultiEdit",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/py-format.sh",
                    "timeout": 60, "statusMessage": "ruff format + check" }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/quality-gate.sh",
                    "timeout": 300, "statusMessage": "Quality gate: ruff, ty, fast tests" }] }
    ]
  }
}
```

### Scripts (`.claude/hooks/`, `chmod +x`, committed)

Constraints: macOS ships bash 3.2, so no bash-4 features. `jq` is at `/usr/bin/jq` (checked). Use
`uv run --frozen` so a hook never rewrites `uv.lock`. All five scripts were pipe-tested from this
plan on 2026-10-04 (see Installing and checking).

`py-format.sh` (H1)

```bash
#!/usr/bin/env bash
# PostToolUse(Write|Edit|MultiEdit): format + lint the edited Python file; report unfixable lint to Claude.
set -uo pipefail
f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
case "$f" in *.py|*.pyi) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
ruff="$CLAUDE_PROJECT_DIR/.venv/bin/ruff"            # direct call: fast, no project build
[ -x "$ruff" ] || ruff="uv run --frozen --quiet ruff"
$ruff format --quiet "$f" 2>/dev/null                 # syntax errors surface in the check below
if ! out=$($ruff check --fix --quiet "$f" 2>&1); then
  printf 'ruff found issues it could not fix in %s:\n%s\n' "$f" "$out" >&2
  exit 2
fi
```

`protect-paths.sh` (H2)

```bash
#!/usr/bin/env bash
# PreToolUse(Write|Edit|MultiEdit): block edits to generated or user-owned paths.
set -uo pipefail
f=$(jq -r '.tool_input.file_path // empty')
[ -z "$f" ] && exit 0
rel=${f#"$CLAUDE_PROJECT_DIR"/}
case "$rel" in
  uv.lock)                 why="uv.lock is generated; change dependencies with 'uv add' / 'uv remove'." ;;
  .venv/*)                 why=".venv is managed by uv; run 'uv sync'." ;;
  data/*|index/*)          why="data/ and index/ hold the user's corpus and generated index; change them via 'qwn ingest'." ;;
  eval/baseline.json)      why="Update the baseline only with 'qwn eval --update-baseline'." ;;
  .streamlit/secrets.toml) why="The secrets file is user-managed." ;;
  *) exit 0 ;;
esac
jq -n --arg r "$why" \
  '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
```

`guard-bash.sh` (H3)

```bash
#!/usr/bin/env bash
# PreToolUse(Bash): enforce uv; confirm before model downloads/loads, servers, or deleting user data.
set -uo pipefail
cmd=$(jq -r '.tool_input.command // empty')
decide() {
  jq -n --arg d "$1" --arg r "$2" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: $d, permissionDecisionReason: $r}}'
  exit 0
}
has() { printf '%s' "$cmd" | grep -Eq "$1"; }

has '(^|[;&|[:space:]])(pip3?|python3? -m pip|uv pip) install' &&
  decide deny "Use 'uv add <pkg>' (or 'uv add --dev <pkg>') so pyproject.toml and uv.lock stay in sync."

if has 'pytest' && ! has 'not slow' && ! has 'tests/(unit|integration|ui)'; then
  decide ask "Full pytest includes @slow tests that download ~12 GB of models and use ~20 GB of memory."
fi

has 'smoke_test\.py|hf download|huggingface-cli download|qwn (ingest|search|ask|eval|ui)|streamlit run' &&
  decide ask "This downloads or loads MLX models (~13 GB) or starts a long-running server."

has 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*([[:space:]]+[^[:space:]]+)*[[:space:]]+([^[:space:]]*/)?(data|index)/?([[:space:];&|]|$)' &&
  decide ask "This deletes the user's corpus or index."

exit 0
```

`quality-gate.sh` (H4)

```bash
#!/usr/bin/env bash
# Stop: if Python changed and isn't committed yet, require format/lint/types/fast tests to pass.
set -uo pipefail
input=$(cat)
[ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false')" = "true" ] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
git status --porcelain -- '*.py' pyproject.toml | grep -q . || exit 0
fail() { printf '%s failed:\n%s\n' "$1" "$(printf '%s' "$2" | tail -40)" >&2; exit 2; }
out=$(uv run --frozen ruff format --check . 2>&1) || fail "ruff format --check" "$out"
out=$(uv run --frozen ruff check . 2>&1)          || fail "ruff check" "$out"
out=$(uv run --frozen ty check 2>&1)              || fail "ty check" "$out"
out=$(uv run --frozen pytest -m "not slow" -q -x 2>&1); rc=$?
[ "$rc" -eq 0 ] || [ "$rc" -eq 5 ] || fail "pytest -m 'not slow'" "$out"   # 5 = no tests collected
exit 0
```

`session-context.sh` (H5)

```bash
#!/usr/bin/env bash
# SessionStart(startup): orient a fresh implementation session. Stdout is added to Claude's context.
cd "$CLAUDE_PROJECT_DIR" || exit 0
echo "Branch: $(git branch --show-current)"
echo "Recent commits:"; git log --oneline -5
echo "Implement PLAN.md phase by phase. Check which phase's exit criteria are already met before starting."
```

### Installing and checking (start of phase 0)

1. Create the scripts and `.claude/settings.json` as above, run `chmod +x .claude/hooks/*.sh`, and
   commit them.
2. Pipe-test each script with a sample payload (`export CLAUDE_PROJECT_DIR=$PWD` first). These
   expected results were confirmed against the scripts above on 2026-10-04:
   - H1: `echo '{"tool_input":{"file_path":"'$PWD'/src/qwn/__init__.py"}}' | .claude/hooks/py-format.sh; echo $?` → `0`.
     For a temporary file with `import os` and an undefined name: it gets formatted, the unused
     import is removed, and `F821` is reported with exit 2 (~35 ms). A syntax error is reported
     with exit 2. Non-`.py` files are skipped.
   - H2: `uv.lock`, `index/qwn.db` → `deny` JSON; `src/qwn/x.py` → no output (allowed)
   - H3: `pip install foo`, `uv pip install foo` → `deny`. `uv run pytest`, `uv run qwn ingest data/`,
     `streamlit run …`, `rm -rf index`, `rm -rf ./data`, `rm -r -f /abs/path/index` → `ask`.
     `uv add foo`, `pytest -m "not slow"`, `pytest tests/unit`, `rm -rf .ruff_cache`,
     `rm -rf indexer`, `rm -rf data_old`, `ls data` → no output (allowed).
   - H4: `echo '{}' | .claude/hooks/quality-gate.sh; echo $?` → `0` on a clean tree, and `0` with
     `{"stop_hook_active":true}`
   - H5: `.claude/hooks/session-context.sh` → branch + recent commits + the PLAN.md reminder
3. Validate the JSON: `jq -e '.hooks.Stop[0].hooks[0].command' .claude/settings.json`.
4. Have the user open `/hooks` once, or restart the session. The settings watcher only picks up a
   newly created `.claude/` after a reload.
5. Prove H1 fires: make an edit that leaves a `.py` file unformatted and confirm the hook fixes it.

### Known trade-offs

- **H4 only checks uncommitted changes.** If Claude commits before stopping, the gate is skipped,
  so implementation sessions must run `uv run pytest -m "not slow"` before each commit.
  `stop_hook_active` lets Claude stop on the *second* attempt, so a test that can't be fixed can't
  trap the session in a loop. The failure output is still visible.
- **H3's `ask` list is pattern-based.** It's a speed bump against accidental 12 GB downloads, not
  a security boundary.
- **H1 calls `.venv/bin/ruff` directly.** `uv run` rebuilds the project first, and a build failure
  would be misreported as lint. It falls back to `uv run` only when `.venv` doesn't exist yet.

## GitHub Actions CI

A single job on **macOS arm64** (`macos-15`), the same platform as the M2 Max. It installs the same
prebuilt packages (mlx, sqlite-vec, pypdfium2) and runs the hook scripts under macOS bash 3.2 (the
runner image has Bash 3.2.57 and jq 1.8.2, checked 2026-10-04). CI never loads real models: the
`slow` tests are deselected, and `HF_HUB_OFFLINE=1` makes any accidental model download fail
immediately instead of pulling ~12 GB.

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
  workflow_call:                 # reused by release.yml as its gate

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  check:
    name: Lint, types, fast tests (macOS arm64)
    runs-on: macos-15            # Apple Silicon
    timeout-minutes: 15
    env:
      HF_HUB_OFFLINE: "1"        # no model downloads in CI; anything that tries fails fast
    steps:
      - uses: actions/checkout@v7
        with:
          persist-credentials: false

      - uses: astral-sh/setup-uv@v10   # caches uv's package downloads automatically on GitHub runners
        with:
          version-file: pyproject.toml # reads [tool.uv] required-version

      - name: Install (fails if uv.lock is out of date)
        run: uv sync --locked

      - name: Ruff format
        run: uv run --frozen ruff format --check .

      - name: Ruff lint
        run: uv run --frozen ruff check --output-format=github .

      - name: Type check
        run: uv run --frozen ty check

      - name: Fast tests (unit, integration, UI, hooks)
        run: uv run --frozen pytest -m "not slow" -ra
```

Supporting changes in `pyproject.toml`:

```toml
[tool.ruff]
# already applied (commit ecbaf41):
extend-exclude = ["*.md"]  # PLAN.md/README.md code blocks are illustrative, not source

[tool.uv]
required-version = ">=0.12.22,<0.13"   # CI and your machine use the same uv
```

**Why exclude Markdown:** ruff 0.16 also formats Python code blocks inside `.md` files. Without the
exclude, `ruff format --check .` fails on PLAN.md's illustrative snippets, which breaks both CI and
the H4 Stop hook. This was found by running the CI steps locally on 2026-10-04, and the fix was
checked in a scratch copy: format and lint both pass.

### Design notes

- **Same checks as the H4 Stop hook.** The local gate and CI run the same four commands, so
  "passes locally" means "passes in CI". CI adds `uv sync --locked`, which catches a `uv.lock` that
  doesn't match `pyproject.toml`.
- **Hook scripts are tested in CI** through `tests/hooks/` (see Testing), so a broken hook is
  caught before it can affect an implementation session.
- **Least privilege:** `contents: read`; the checkout doesn't keep the token
  (`persist-credentials: false`); no secrets are used.
- **Cost:** about 3 minutes per run with a warm cache. That's free on a public repo. On a private
  repo macOS minutes are billed at 10×, so roughly 30 billed minutes per run.
- **Speed:** concurrency cancels older runs of a PR when a new commit is pushed. Pushes to `main`
  are never cancelled.
- **Slow tests stay local** (decided). GitHub runners can't run the ~20 GB real-model suite. It's
  part of the merge checklist below.

### Merge checklist (each phase's PR into `main`)

1. CI is green.
2. `uv run pytest` passes locally on the M2 Max, **including** `slow` tests.
3. From phase 1 on: `uv run qwn eval` meets the phase's exit criterion. If the change was
   intentional, update `eval/baseline.json` with `--update-baseline` in the same PR.
4. The version is bumped with `uv version --bump minor` when the PR completes a phase (merging it
   triggers the release).
5. After the first push to GitHub, protect `main`: require the `Lint, types, fast tests (macOS arm64)`
   check, and require branches to be up to date before merging.

## Automatic releases

When a push to `main` changes `pyproject.toml` and the version has **no `v<version>` tag yet**, the
`release` workflow runs CI and, if it passes, creates a tag and a **GitHub Release** with
auto-generated notes and the wheel and sdist attached. Nothing is published to PyPI (decided). The
name `qwn` was free on PyPI on 2026-10-04, and Trusted Publishing can be added later as one extra
job.

### Versioning and how to release

- **One command bumps the version:** `uv version --bump patch|minor|major` (pre-releases:
  `uv version --bump minor --bump rc`). It updates **both** `pyproject.toml` and `uv.lock`. Never
  hand-edit the version: `uv.lock` records it, so `uv sync --locked` in CI fails on a hand-edit
  (H2 also blocks `uv.lock` edits).
- **Policy: each completed phase is a minor release.** Phase 0 → `0.1.0` (the current version, not
  yet tagged), phase 1 → `0.2.0`, … phase 5 → `0.6.0`, then `1.0.0` once you use it daily. Fixes
  between phases → patch.
- **Flow:** bump on the phase branch → PR (CI) → merge to `main` → `release.yml` → tag `vX.Y.Z` +
  GitHub Release.
- **Note:** `0.1.0` has no tag, so the **first** merge to `main` that touches `pyproject.toml`
  (phase 0) publishes `v0.1.0`. That's intended. Bump first if you'd rather not release phase 0.

### `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    branches: [main]
    paths: [pyproject.toml]
  workflow_dispatch:             # re-run a release that failed after the fix is merged

permissions:
  contents: read

concurrency:
  group: release                 # one release at a time, never cancelled mid-way
  cancel-in-progress: false

jobs:
  detect:
    name: Detect version bump
    runs-on: ubuntu-latest
    outputs:
      release: ${{ steps.check.outputs.release }}
      version: ${{ steps.check.outputs.version }}
      tag: ${{ steps.check.outputs.tag }}
      prerelease: ${{ steps.check.outputs.prerelease }}
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0         # need all tags
          persist-credentials: false
      - uses: astral-sh/setup-uv@v10
        with:
          version-file: pyproject.toml
      - id: check
        name: Compare pyproject version with existing tags (PEP 440)
        run: |
          set -euo pipefail
          version=$(uv version --short)
          tag="v$version"
          if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
            echo "Tag $tag already exists; nothing to release."
            echo "release=false" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          # PEP 440 comparison against every existing v* tag (git's version sort mis-orders rc tags).
          read -r newer pre latest < <(git tag --list 'v*' | uv run --no-project --quiet --with packaging python -c '
          import sys
          from packaging.version import InvalidVersion, Version
          new = Version(sys.argv[1])
          tags = []
          for line in sys.stdin:
              try:
                  tags.append(Version(line.strip().removeprefix("v")))
              except InvalidVersion:
                  pass
          latest = max(tags, default=None)
          print(str(latest is None or new > latest).lower(), str(new.is_prerelease).lower(), latest or "none")
          ' "$version")
          if [ "$newer" != "true" ]; then
            echo "::error file=pyproject.toml::Version $version is not greater than the latest released v$latest"
            exit 1
          fi
          { echo "release=true"; echo "version=$version"; echo "tag=$tag"; echo "prerelease=$pre"; } >> "$GITHUB_OUTPUT"
          echo "Will release $tag (prerelease=$pre, previous: $latest)"

  ci:
    name: CI gate
    needs: detect
    if: needs.detect.outputs.release == 'true'
    uses: ./.github/workflows/ci.yml

  release:
    name: Build and publish GitHub Release
    needs: [detect, ci]
    if: needs.detect.outputs.release == 'true'
    runs-on: ubuntu-latest       # pure-Python wheel (uv_build): platform-independent, cheap
    permissions:
      contents: write            # create tag + release; only this job can write
    steps:
      - uses: actions/checkout@v7
        with:
          persist-credentials: false
      - uses: astral-sh/setup-uv@v10
        with:
          version-file: pyproject.toml
      - name: Build wheel and sdist
        run: uv build --out-dir dist
      - name: Check artifact versions
        env:
          VERSION: ${{ needs.detect.outputs.version }}
        run: |
          ls dist
          test -f "dist/qwn-${VERSION}-py3-none-any.whl"
          test -f "dist/qwn-${VERSION}.tar.gz"
      - name: Create tag and GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
          TAG: ${{ needs.detect.outputs.tag }}
          PRERELEASE: ${{ needs.detect.outputs.prerelease }}
        run: |
          flags=(--repo "$GITHUB_REPOSITORY" --target "$GITHUB_SHA" --title "$TAG" --generate-notes)
          [ "$PRERELEASE" = "true" ] && flags+=(--prerelease)
          gh release create "$TAG" dist/* "${flags[@]}"
```

### Design notes

- **"Version bump" means "no tag for this version yet",** not "the diff touched the version line".
  This handles squash merges and multi-commit pushes, makes re-runs safe (an existing tag means a
  no-op), and lets `workflow_dispatch` retry a failed release. The `paths` filter keeps it from
  running on unrelated pushes.
- **Mistakes fail loudly:** a version that isn't greater than the newest tag (by PEP 440, including
  rc/dev ordering) fails `detect` with an annotation on `pyproject.toml`. A hand-edited version
  without a lock update fails CI's `uv sync --locked`.
- **The CI gate is the same `ci.yml`** (via `workflow_call`), so a release can't skip a check that
  PRs run. This means CI runs twice on a bump commit (once from `ci.yml`'s own push trigger); that's
  accepted for simplicity.
- **Least privilege:** read-only at the top level. Only the `release` job gets `contents: write`,
  and it uses the built-in `GITHUB_TOKEN`; no secrets. Tags created with `GITHUB_TOKEN` don't
  trigger other workflows, so there are no loops.
- **Prototyped on 2026-10-04** in a scratch git repo:
  - Detection: no tags → release `v0.1.0`; tag exists → no-op; `0.3.0rc2` while `v0.3.0` exists
    → error; `0.3.1` → release; `0.4.0.dev1` → prerelease. Invalid tags such as `vjunk` are ignored
    (git's own sort would have picked it as "latest").
  - Bumping and build: `uv version --bump minor` updated `uv.lock`, and `uv build` produced
    `qwn-0.2.0-py3-none-any.whl` and `qwn-0.2.0.tar.gz`.
- **Optional later:** add `.github/release.yml` to group generated notes by PR label, and a `pypi`
  job (Trusted Publishing, `environment: pypi`) between `ci` and `release`.

## Development workflow

```bash
uv sync                          # create .venv from uv.lock
uv run ruff format .             # format
uv run ruff check --fix .        # lint
uv run ty check                  # type check
uv run pytest -m "not slow"      # fast tests (no model downloads)
uv run pytest                    # everything, including real-model tests
uv run qwn ui                    # launch Streamlit
```

## Risks and mitigations

- **mlx-vlm embed/rerank Python API isn't documented.** Phase 0 reads the server handlers and
  copies their calls. Fallback: `mlx-embeddings` 0.1.0 behind the same adapters. It's GPL-3.0, so
  adopting it needs the user's OK; the alternative is a minimal in-house adapter (last-token pooling,
  yes/no logit difference).
- **mlx-community 8-bit conversions may not load.** Fallback: the original `Qwen/…` bf16 repos
  (~4.5 GB each), which still fit.
- **mlx-embeddings / mlx-vlm embedding numbers may drift from the reference.** The `slow` test
  compares against the PyTorch reference (cos ≥ 0.99) on 5 samples.
- **Long contexts eat memory.** `max_context=16384`, `max_images=4`, `max_pixels` cap.
- **sqlite-vec is pre-1.0 (0.1.9).** Pin it in `uv.lock`. The numpy-BLOB fallback is described under Data model.
- **Libraries change fast.** Pin the exact versions in `uv.lock` after phase 0. All library calls
  live in `adapters/`.
