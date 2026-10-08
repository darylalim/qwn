"""Make reference embeddings with the official Qwen3-VL-Embedding model (run once, in phase 0).

The slow test `tests/slow/test_embedding_reference.py` checks that qwn's mlx-vlm embedder agrees
with these (cosine ≥ 0.99). torch never becomes a project dependency; run this in a throwaway env:

    uv run --no-project --with sentence-transformers --with torch --with pillow \
        python scripts/make_embedding_reference.py

It downloads Qwen/Qwen3-VL-Embedding-2B (~4.5 GB, bf16) and writes
tests/slow/fixtures/embedding_reference.json plus the two PNGs it embeds. Synthetic content only.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "slow" / "fixtures"
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DIM = 1024

# Must match qwn.prompts (copied, so this script runs without the project installed).
QUERY = "Retrieve images or text relevant to the user's query."
DOC = "Represent the user's input."

PAGE_TEXT = "Quarterly report. Q3 revenue grew 12% year over year, driven by subscriptions."


def make_images() -> tuple[Path, Path]:
    font = ImageFont.load_default(size=28)
    chart = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(chart)
    for i, h in enumerate([120, 200, 160, 320]):
        d.rectangle([80 + i * 130, 420 - h, 160 + i * 130, 420], fill=(30, 90, 160))
    d.text((80, 30), "Revenue by quarter", fill="black", font=font)
    page = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(page)
    d.multiline_text((40, 60), PAGE_TEXT.replace(". ", ".\n"), fill="black", font=font)
    chart_path, page_path = OUT / "ref_chart.png", OUT / "ref_page.png"
    chart.save(chart_path)
    page.save(page_path)
    return chart_path, page_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chart, page = make_images()
    samples = [
        {"text": "How much did revenue grow in Q3?", "image": None, "is_query": True},
        {"text": PAGE_TEXT, "image": None, "is_query": False},
        {"text": None, "image": chart.name, "is_query": False},
        {"text": PAGE_TEXT, "image": page.name, "is_query": False},
        {"text": "a bar chart of revenue", "image": None, "is_query": True},
    ]
    model = SentenceTransformer(MODEL, device="cpu")
    vectors = []
    for s in samples:
        if s["image"] is None:
            inp: object = s["text"]
        else:
            parts: dict[str, object] = {"image": Image.open(OUT / str(s["image"])).convert("RGB")}
            if s["text"]:
                parts["text"] = s["text"]
            inp = parts
        v = model.encode([inp], prompt=QUERY if s["is_query"] else DOC)[0][:DIM]
        vectors.append((v / np.linalg.norm(v)).astype(float).round(7).tolist())
    meta = {
        "model": MODEL,
        "dim": DIM,
        "python": sys.version.split()[0],
        "note": "first `dim` values of the official embedding, L2-renormalised",
    }
    out = OUT / "embedding_reference.json"
    out.write_text(json.dumps({"meta": meta, "samples": samples, "vectors": vectors}) + "\n")
    print(f"wrote {out} ({len(vectors)} x {DIM})")


if __name__ == "__main__":
    main()
