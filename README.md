# qwn

Private, local multimodal RAG on Apple Silicon: ask questions about your PDFs, slides (exported as
PDF), images and notes, and get answers with page citations. Everything runs on-device with MLX.

Work in progress: see `PLAN.md` for the phases.

## Setup

```bash
uv sync
uv run qwn models pull      # the only command that uses the network (~13 GB)
uv run qwn models status
```

After `qwn models pull`, qwn runs fully offline (`HF_HUB_OFFLINE=1`). Run `qwn` from this folder,
or set `QWN_HOME` to it.

## Use

```bash
uv run qwn ui                         # Streamlit app on localhost: Chat, Library, System
uv run qwn ingest ~/Documents/papers  # or index folders in place from the CLI
uv run qwn ask "What was APAC revenue in Q3?"
uv run qwn eval review                # turn 👍/👎 Chat ratings into private eval questions
```

The UI loads the models (~13 GB) in the background when it starts. Only one qwn process can hold
the models, so close the UI before running `ask`, `search`, `ingest` or `eval` from the CLI.

## Models and licenses

qwn's code is Apache-2.0 (see `LICENSE`). Model weights aren't redistributed with qwn:
`qwn models pull` downloads them from Hugging Face under their own licenses, at the revisions
pinned in `src/qwn/models_lock.py`.

| Role | Model (MLX conversion by mlx-community) | Upstream | License |
|---|---|---|---|
| Answers | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | Qwen3-VL-8B-Instruct (Qwen team, Alibaba Cloud) | Apache-2.0 |
| Embedding | `mlx-community/Qwen3-VL-Embedding-2B-8bit` | Qwen3-VL-Embedding-2B | Apache-2.0 |
| Reranking | `mlx-community/Qwen3-VL-Reranker-2B-8bit` | Qwen3-VL-Reranker-2B | Apache-2.0 |
| Safety | `mlx-community/Qwen3Guard-Gen-0.6B-MLX` | Qwen3Guard-Gen-0.6B | Apache-2.0 |
| Speech to text (phase 4) | `mlx-community/Qwen3-ASR-1.7B-8bit` | Qwen3-ASR-1.7B | Apache-2.0 |
| Text to speech (phase 4) | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | Qwen3-TTS-0.6B | Apache-2.0 |
| Voice activity (phase 4) | `mlx-community/silero-vad` | Silero VAD (Silero Team) | MIT (upstream) |

Thanks to the Qwen team for the models, the Silero team for Silero VAD, and the
[MLX](https://github.com/ml-explore/mlx) and mlx-community maintainers for the Apple Silicon ports.
