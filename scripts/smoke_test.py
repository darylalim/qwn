"""Phase 0 smoke test: load the 4 core models together and exercise each once.

    uv run python scripts/smoke_test.py

Exit criteria checked here (PLAN.md → Phases, phase 0): the core models load together through the
Registry's memory check (≈13 GB active), everything runs offline, and VL-8B generates ≥ 35 tok/s.
Uses synthetic content only.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # before anything imports huggingface_hub

import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from qwn.config import Settings
from qwn.interfaces import Item, Source
from qwn.models import CORE_ROLES, Registry

MIN_TPS = 35.0
PAGE = "Northwind quarterly report. Q3 revenue grew 12% year over year to 4.2 million euros."
CAKE = "A recipe for lemon cake: butter, sugar, eggs, flour and two lemons."


def text_image(folder: Path, name: str, text: str) -> Path:
    """A synthetic A4 page (150 dpi) showing `text`."""
    img = Image.new("RGB", (1240, 1754), "white")
    font = ImageFont.load_default(size=36)
    ImageDraw.Draw(img).multiline_text(
        (100, 150), text.replace(". ", ".\n"), fill="black", font=font, spacing=16
    )
    path = folder / f"{name}.png"
    img.save(path)
    return path


def page_image(folder: Path) -> Path:
    return text_image(folder, "page", PAGE)


def main() -> int:
    reg = Registry(Settings())
    failures: list[str] = []

    for role in CORE_ROLES:
        t = time.perf_counter()
        getattr(reg, role)()
        mem = reg.memory_gb()
        print(f"loaded {role:9} in {time.perf_counter() - t:5.1f} s  active {mem['active']:.1f} GB")
    mem = reg.memory_gb()
    print(f"core loaded: active {mem['active']:.1f} GB of {mem['recommended']:.1f} GB recommended")

    with tempfile.TemporaryDirectory() as tmp:
        image = page_image(Path(tmp))
        cake_image = text_image(Path(tmp), "cake", CAKE)

        # The relevant page must win in every input mode the index uses.
        q = "How much did revenue grow in Q3?"
        query_vec = reg.embedder().embed([Item(text=q)], is_query=True)[0]
        for kind, docs in [
            ("text", [Item(text=PAGE), Item(text=CAKE)]),
            ("image", [Item(image_path=image), Item(image_path=cake_image)]),
            ("image+text", [Item(PAGE, image), Item(CAKE, cake_image)]),
        ]:
            sims = reg.embedder().embed(docs, is_query=False) @ query_vec
            print(f"embed {kind:10}: page={sims[0]:.3f} cake={sims[1]:.3f}")
            if not sims[0] > sims[1]:
                failures.append(f"embedder ranks the cake above the page ({kind})")

        scores = reg.reranker().score(Item(text=q), [Item(PAGE, image), Item(text=CAKE)])
        print(f"rerank: page={scores[0]:.3f} cake={scores[1]:.3f}")
        if not scores[0] > scores[1]:
            failures.append("reranker ranks the cake above the page")

        verdict = reg.guard().check_prompt(q)
        print(f"guard: {verdict.label} {verdict.categories}")
        if verdict.label != "Safe":
            failures.append(f"guard flagged a harmless question: {verdict}")

        source = Source("S1", "c1", "report.pdf", 1, PAGE, image, 1.0)
        gen = reg.generator()
        q_long = "Summarise the page in detail, quoting every figure, in about 150 words."
        short = gen.answer(q, [source], greedy=True)  # also warms up
        print(f"answer: cited {short.cited}")
        if short.cited != ["S1"]:
            failures.append("generator didn't cite S1")
        t = time.perf_counter()
        answer = gen.answer(q_long, [source], greedy=True)
        elapsed = time.perf_counter() - t
        tps = getattr(gen, "last_tps", 0.0)  # MlxVlmGenerator extra: decode speed, no prefill
        print(f"generate: {answer.completion_tokens} tokens in {elapsed:.1f} s, {tps:.1f} tok/s")
        if tps < MIN_TPS:
            failures.append(f"VL-8B generates {tps:.1f} tok/s < {MIN_TPS}")

    mem = reg.memory_gb()
    print(f"peak {mem['peak']:.1f} GB, active {mem['active']:.1f} GB")
    for f in failures:
        print(f"FAIL: {f}")
    print("smoke test passed" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
