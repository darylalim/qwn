"""pypdfium2: PDF pages → text + WebP render (PLAN.md → Data model, Runtime robustness).

PDFium isn't thread-safe and can crash the process, not just raise, so every call here holds one
module-level lock from opening a document until it's closed.
"""

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from qwn.adapters.images import save_webp

_LOCK = threading.Lock()
POINTS_PER_INCH = 72


class PdfRejected(ValueError):
    """The PDF can't be indexed; the message is the reason shown to the user."""


@dataclass(frozen=True)
class RenderedPage:
    number: int  # 1-based
    text: str
    image_path: Path


def render(
    src: Path, out_dir: Path, *, dpi: int, max_pages: int, max_image_pixels: int
) -> list[RenderedPage]:
    """Extract each page's text layer and render it to `out_dir/<page>.webp`."""
    with _LOCK:
        try:
            doc = pdfium.PdfDocument(src)
        except pdfium.PdfiumError as e:
            if e.err_code == pdfium_c.FPDF_ERR_PASSWORD:
                raise PdfRejected("password-protected (not supported)") from None
            raise PdfRejected("unreadable PDF") from None
        try:
            n = len(doc)
            if n == 0:
                raise PdfRejected("PDF has no pages")
            if n > max_pages:
                raise PdfRejected(f"too many pages ({n} > max_pdf_pages {max_pages})")
            pages: list[RenderedPage] = []
            for i in range(n):
                page = doc[i]
                try:
                    textpage = page.get_textpage()
                    text = textpage.get_text_range().replace("\r\n", "\n").strip()
                    textpage.close()
                    w, h = page.get_size()
                    # pdf_dpi, unless the page is huge (posters): stay under max_image_pixels.
                    scale = dpi / POINTS_PER_INCH
                    if (w * scale) * (h * scale) > max_image_pixels:
                        scale = math.sqrt(max_image_pixels / (w * h))
                    image = page.render(scale=scale).to_pil()
                finally:
                    page.close()
                path = out_dir / f"{i + 1}.webp"
                save_webp(image, path)
                pages.append(RenderedPage(i + 1, text, path))
            return pages
        finally:
            doc.close()
