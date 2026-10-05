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
| Libraries | `mlx-vlm` for generation, embedding **and** reranking; `mlx-lm` for Guard; `mlx-embeddings` as fallback. |
| Index | LanceDB, 3 tables (`documents`, `chunks`, `meta`), per-file update by content hash. |
| UI | Streamlit app with 3 pages: Chat, Library, System. |
| CLI | `ingest`, `search`, `ask`, `eval`, `status`, `ui`. |
| Testing | 3 tiers + fakes: unit, integration, UI (AppTest), plus `slow` real-model tests. |
| Eval | recall@1/5/10 + MRR (with/without rerank), latency by stage, citation hit. JSON results + committed baseline. |
| Config | `pydantic-settings`: defaults → `qwn.toml` → `QWN_*` env → CLI flags / UI sliders. |
| Tooling | uv (Python 3.12), ruff, ty, pytest (already set up). |

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
| `lancedb` | `>=0.39` | Embedded vector store |
| `pymupdf` | `>=1.28` | PDF → PNG + text |
| `pillow` | latest | Image loading/resizing |
| `typer` | `>=0.27` | CLI |
| `streamlit` | `>=1.65` | UI (`numpy<3`, OK) |
| `pydantic-settings` | latest | Config, TOML + env |
| `mlx-embeddings` | `==0.1.0` | **Fallback only**, added only if mlx-vlm embed/rerank fails phase 0 |

These pins overlap, so a single environment should resolve. Phase 0 confirms it with
`uv lock`.

## Data flow

```
INGEST   PDF ─► PyMuPDF ─► page PNG (150 dpi) + page text ─┐
         image / screenshot ─► normalised PNG copy ─────────┼─► Embedder ─► 1024-d, L2-norm ─► LanceDB
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

## Data model (LanceDB at `index_dir`)

```
documents  doc_id      str   sha256(abs path)                     primary key
           path        str   absolute path
           sha256      str   content hash
           mtime       float
           kind        str   "pdf" | "image" | "text"
           n_chunks    int
           indexed_at  str   ISO-8601

chunks     chunk_id    str   sha256(f"{doc_id}:{page}:{chunk_idx}")
           doc_id      str
           kind        str   "pdf_page" | "image" | "text"
           page        int?  1-based for PDF pages
           chunk_idx   int   0 for pages/images
           text        str   page text / chunk text / "" for images
           image_path  str?  index_dir/pages/<doc_id>/<page>.png (or copied image)
           vector      float32[embed_dim]

meta       key/value   embed_model, embed_dim, schema_version
```

- **Ingest algorithm**: walk the paths, skipping anything not in {pdf, png, jpg, jpeg, webp, md,
  txt}. For each file, compute sha256. If it matches `documents`, skip. Otherwise delete its
  chunks, render/chunk, embed in batches (8 images or 32 texts), and insert. Update `documents`
  last, so a crash midway leaves a document marked as "not indexed".
- **Removed files**: `qwn ingest --prune` deletes rows for files that no longer exist.
- **Rebuild trigger**: if `meta.embed_model`, `meta.embed_dim` or `meta.schema_version` differs
  from the settings, stop with a message suggesting `--reindex`.
- **Search**: brute-force cosine (dot product on normalised vectors) for `top_k`. Add an ANN
  index only if the corpus goes over ~100k chunks.

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
tests/integration/   tmp_path LanceDB; PyMuPDF-generated 2-page PDF; PIL image; markdown file
                     ingest → re-ingest skips → modify one file → only it re-indexes → --prune
tests/ui/            streamlit.testing.v1.AppTest smoke test per page, with fakes injected
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
| 0 | Env + checks | Add runtime deps; `interfaces.py`; adapters; `scripts/smoke_test.py`; `tests/slow/` contract tests | `uv lock` resolves; 4 core models loaded together under 24 GB; VL-8B ≥ 35 tok/s; slow tests pass (or the fallback library is adopted) |
| 1 | Retrieval | `config`, `ingest`, `index`, `retrieve`; `qwn ingest/search/status`; unit + integration tests | recall@5 ≥ 0.8 on 20 queries; rerank beats embedding alone |
| 2 | Answering | `prompts`, `answer`; `qwn ask`; `qwn eval` with citation_hit | Correct page cited in most of the 20 queries |
| 3 | Safety | `guard` wired into ask/chat; controversial policy | Known unsafe prompts blocked; normal prompts pass; unparseable output → warn |
| 4 | Voice | ASR/TTS adapters (mlx-audio already installed via mlx-vlm); `st.audio_input` + playback | About 3 s or less from end of speech to first audio |
| 5 | Streamlit UI | 3-page app; AppTest smoke tests; `qwn ui` | Full flow usable from the browser |

Phase 5 may begin once phase 2 is done; the Chat page doesn't need Guard or voice.

## Repo layout (target)

```
qwn/
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
├─ tests/                   # fakes.py, unit/, integration/, ui/, slow/
└─ data/  index/            # gitignored
```

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
  copies their calls. Fallback: `mlx-embeddings` 0.1.0 behind the same adapters.
- **mlx-community 8-bit conversions may not load.** Fallback: the original `Qwen/…` bf16 repos
  (~4.5 GB each), which still fit.
- **mlx-embeddings / mlx-vlm embedding numbers may drift from the reference.** The `slow` test
  compares against the PyTorch reference (cos ≥ 0.99) on 5 samples.
- **Long contexts eat memory.** `max_context=16384`, `max_images=4`, `max_pixels` cap.
- **Libraries change fast.** Pin the exact versions in `uv.lock` after phase 0. All library calls
  live in `adapters/`.
