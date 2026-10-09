"""md/txt chunking, contextual headers, embed inputs and the ingest hash."""

from pathlib import Path

from qwn.config import Settings
from qwn.ingest import (
    HEADING_SEP,
    chunk_text,
    context_header,
    embed_input,
    ingest_hash,
    sections,
    split,
)

MD = """\
# Pricing

Intro paragraph.

## Starter

Five users.

## Enterprise

### SSO

Included in all plans.

```
# not a heading inside a fence
```

# Empty

## Next
Body.
"""


def test_sections_track_heading_path():
    got = sections(MD, markdown=True)
    paths = [p for p, _ in got]
    assert paths == [
        "Pricing",
        f"Pricing{HEADING_SEP}Starter",
        f"Pricing{HEADING_SEP}Enterprise{HEADING_SEP}SSO",
        f"Empty{HEADING_SEP}Next",
    ]


def test_heading_inside_code_fence_is_body():
    sso = dict(sections(MD, markdown=True))[f"Pricing{HEADING_SEP}Enterprise{HEADING_SEP}SSO"]
    assert "# not a heading inside a fence" in sso


def test_sections_include_heading_line_in_body():
    starter = dict(sections(MD, markdown=True))[f"Pricing{HEADING_SEP}Starter"]
    assert starter.startswith("## Starter")


def test_plain_text_has_no_headings():
    assert sections("# just text\n\nmore", markdown=False) == [("", "# just text\n\nmore")]


def test_split_packs_paragraphs_up_to_limit():
    body = "\n\n".join(["a" * 40] * 5)
    chunks = split(body, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).replace("\n", "") == "a" * 200


def test_split_breaks_long_paragraph_at_sentence():
    para = "First sentence here. " * 20
    chunks = split(para, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert all(c.endswith(".") for c in chunks[:-1])


def test_chunk_text_respects_chunk_tokens():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(20))
    chunks = chunk_text(text, chunk_tokens=100, markdown=False)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for _, c in chunks)


def test_context_header():
    sso = f"Enterprise{HEADING_SEP}SSO"
    assert context_header("pricing.md", sso) == f"pricing.md{HEADING_SEP}{sso}"
    assert context_header("notes.txt", "") == "notes.txt"


def test_header_goes_to_embedder_not_stored_text(home):
    s = Settings()
    item = embed_input("text", "Included in all plans.", None, "Enterprise", "pricing.md", s)
    assert item.text == f"pricing.md{HEADING_SEP}Enterprise\n\nIncluded in all plans."
    plain = embed_input(
        "text", "Included", None, "Enterprise", "pricing.md", Settings(chunk_context=False)
    )
    assert plain.text == "Included"


def test_pdf_page_input_follows_pdf_embed(home):
    img = Path("1.webp")
    both = embed_input("pdf_page", "x" * 3000, img, "", "a.pdf", Settings())
    assert both.image_path == img and both.text == "x" * 2000
    image_only = embed_input("pdf_page", "text", img, "", "a.pdf", Settings(pdf_embed="image"))
    assert image_only.text is None and image_only.image_path == img
    no_text_layer = embed_input("pdf_page", "", img, "", "a.pdf", Settings())
    assert no_text_layer.text is None


def test_ingest_hash_tracks_stored_settings_only(home):
    base = ingest_hash(Settings())
    assert ingest_hash(Settings(pdf_dpi=200)) != base
    assert ingest_hash(Settings(chunk_context=False)) != base
    assert ingest_hash(Settings(pdf_embed="image")) != base
    assert ingest_hash(Settings(top_k=7, hybrid=False)) == base  # query-time settings
