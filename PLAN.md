# Plan: Local multimodal RAG + voice assistant (M2 Max, 32 GB, all-MLX)

## Goal

Ask questions — typed or spoken — over your own PDFs, slides, screenshots and photos. Answers come
from **Qwen3-VL-8B**, which reads the actual page images and cites file and page. Prompts and
answers pass through Qwen3Guard. Everything runs locally on Apple Silicon with MLX, behind a
Streamlit UI.

## Stack and memory budget

| Role | Repo (`mlx-community/…`) | Library | Weights |
|---|---|---|---|
| Brain | `Qwen3-VL-8B-Instruct-4bit` | `mlx-vlm` | 5.8 GB |
| Embedding | `Qwen3-VL-Embedding-2B-8bit` | `mlx-embeddings` | 2.7 GB |
| Reranker | `Qwen3-VL-Reranker-2B-8bit` | `mlx-embeddings` | 2.7 GB |
| Guard | `Qwen3Guard-Gen-0.6B-MLX` | `mlx-lm` | 1.2 GB |
| ASR (phase 4) | `Qwen3-ASR-1.7B-8bit` | `mlx-audio` | 2.5 GB |
| TTS (phase 4) | `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | `mlx-audio` | 2.9 GB |
| **Total** | | | **≈ 18 GB weights + 3–5 GB activations/KV ≈ 22 GB** |

- Raise the GPU wired limit if needed: `sudo sysctl iogpu.wired_limit_mb=26000` (resets on reboot).
- ASR/TTS are lazy-loaded, so text-only use stays around 13 GB.
- The mlx-community README snippets for the Embedding/Reranker repos are copy-paste boilerplate
  (SigLIP). Use the `mlx-embeddings` GitHub README API instead (`model.process(...)`).

## Data flow

```
INGEST   PDF ─► PyMuPDF ─► page PNG (≈150 dpi) + page text ─┐
         image / screenshot ────────────────────────────────┼─► Embedding-2B ─► MRL-truncate 1024-d, L2-norm ─► LanceDB
         .md / .txt ─► ~800-token chunks ───────────────────┘                     (path, page, kind, text)

QUERY    (voice ─► ASR) ─► text ─► Guard(prompt) ─► Embedding(+instruction) ─► top-50
                                                     ─► Reranker ─► top-5 ─► VL-8B (≤4 page images + text)
                                                     ─► answer with [file p.N] citations ─► Guard(response) ─► (TTS)
```

## Key design decisions

1. **Single Python process, lazy model registry.** Every model is MLX on the same Metal device, so
   servers add nothing. Split speech into a separate process only if dependency pins conflict.
2. **Streamlit UI** caches models with `st.cache_resource`, so they load once per server process,
   not on every rerun. Core logic stays in `qwn.*` modules; the UI module only handles layout and state.
3. **Visual RAG on page images.** Index each PDF page as an image (plus text as a fallback), so
   charts, tables and scans work without OCR.
4. **LanceDB** as an embedded on-disk vector store with metadata. Brute force is fine up to ~100k items.
5. **1024-d Matryoshka (MRL) truncation.** Halves storage for under 1 point of quality lost, and the reranker recovers precision.
6. **Image budget.** At most ~4 page images per answer, with `max_pixels` capped to protect the context.

## Phases

| # | Phase | Deliverable | Exit criterion |
|---|---|---|---|
| 0 | Env + smoke test | Runtime deps; `scripts/smoke_test.py` loads each model and reports `mx.get_peak_memory()` and tok/s | 4 core models co-resident under 24 GB; VL-8B at least 35 tok/s |
| 1 | Retrieval core | `qwn ingest <dir>`, `qwn search "<q>"` | recall@5 ≥ 0.8 on 20 hand-written queries; reranker beats embedding alone |
| 2 | Answering | `qwn ask "<q>"`: retrieve, rerank, VL-8B answer with citations | Correct page cited in most of the 20 queries |
| 3 | Safety | Guard on input and output (Controversial warns, Unsafe blocks) | Red-team prompts blocked; normal prompts pass |
| 4 | Voice | Push-to-talk: ASR → … → TTS | About 3 s or less from end of speech to first audio |
| 5 | Streamlit UI | `streamlit run src/qwn/ui/app.py`: upload/ingest, chat, page thumbnails with citations, mic input (`st.audio_input`), audio playback | Full flow usable from the browser |

## Repo layout (target)

```
qwn/
├─ pyproject.toml          # uv project; dev group: ruff, ty, pytest
├─ PLAN.md
├─ src/qwn/
│  ├─ config.py            # repo IDs, dims, top-k, image caps
│  ├─ models.py            # lazy registry + memory logging
│  ├─ ingest.py  index.py  retrieve.py  answer.py  guard.py  voice.py
│  ├─ cli.py               # typer CLI
│  └─ ui/app.py            # Streamlit app
├─ scripts/smoke_test.py
├─ eval/queries.jsonl      # 20 questions + expected file/page
├─ tests/                  # fast unit tests; real-model tests marked `slow`
└─ data/  index/           # gitignored
```

## Development workflow

```bash
uv sync                          # create .venv from uv.lock
uv run ruff format .             # format
uv run ruff check --fix .        # lint
uv run ty check                  # type check
uv run pytest -m "not slow"      # fast tests (no model downloads)
uv run pytest                    # everything, including real-model tests
```

## Risks and mitigations

- **Dependency pin conflicts** between `mlx-vlm`, `mlx-audio`, `mlx-embeddings` and `streamlit`
  (mlx / transformers / numpy versions). Phase 0 checks this. Fallback: speech in a sidecar process.
- **mlx-community 8-bit embedding/reranker may not load in `mlx-embeddings`.** Its README loads the
  original `Qwen/…` repos. Fallback: the bf16 originals (~4.5 GB each), which still fit.
- **`mlx-embeddings` is young (0.1.0).** In phase 0, compare its embeddings with the PyTorch reference on 5 samples (cosine ≥ 0.99).
- **Long contexts eat memory.** Cap the generation context at ~16K tokens.

## Decisions (resolved open questions)

1. **Corpus:** mixed. PDFs/slides (page images + text), images/screenshots, and markdown/text.
   Video is out of scope.
2. **Voice:** deferred to phase 4. v1 is text-only (~13 GB) and doesn't depend on `mlx-audio`.
3. **Interface:** Typer CLI for ingesting, evaluating and scripting, plus Streamlit for everyday use.
   Both are thin layers over the same `qwn.*` modules.
