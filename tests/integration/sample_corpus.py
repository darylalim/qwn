"""Shared helpers for the integration tests: a small corpus and a counting embedder."""

from pathlib import Path

from fakes import FakeEmbedder
from pdf_fixture import write_pdf
from PIL import Image, ImageDraw

MD = """\
# Pricing

## Starter

Five users, nine euros per user.

## Enterprise

Single sign-on and an audit log. Order code PO-48213.
"""


class CountingEmbedder(FakeEmbedder):
    def __init__(self, dim: int = 1024) -> None:
        super().__init__(dim)
        self.calls = 0
        self.items = 0

    def embed(self, items, *, is_query):
        self.calls += 1
        self.items += len(items)
        return super().embed(items, is_query=is_query)


def make_corpus(root: Path) -> Path:
    docs = root / "docs"
    write_pdf(
        docs / "report.pdf",
        [["Quarterly report", "Revenue grew 12 percent."], ["Outlook", "Hiring continues."]],
    )
    img = Image.new("RGB", (320, 200), "white")
    ImageDraw.Draw(img).text((10, 10), "Bar chart", fill="black")
    img.save(docs / "chart.png")
    (docs / "pricing.md").write_text(MD)
    return docs
