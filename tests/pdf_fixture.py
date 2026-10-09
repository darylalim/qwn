"""A tiny dependency-free PDF writer: text pages in Helvetica, one or more lines per page.

Used by the integration tests and by eval/public/build_corpus.py. ASCII text only.
"""

from pathlib import Path


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# A standard security handler whose /U doesn't match the empty password: PDFium asks for one.
_ENCRYPT = b"/Encrypt << /Filter /Standard /V 1 /R 2 /O <%s> /U <%s> /P -4 >> /ID [<%s><%s>] " % (
    b"ab" * 32,
    b"cd" * 32,
    b"0f" * 16,
    b"0f" * 16,
)


def write_pdf(
    path: Path, pages: list[list[str]], *, size: int = 12, encrypted: bool = False
) -> Path:
    """Write `pages` (each a list of lines) as an A4 PDF with a real text layer.

    `encrypted=True` marks it password-protected (the content itself isn't encrypted).
    """
    leading = round(size * 1.4)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # Pages, filled in below once the kids are known
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for lines in pages:
        ops = " T* ".join(f"({_escape(line)}) Tj" for line in lines)
        stream = f"BT /F1 {size} Tf {leading} TL 72 770 Td {ops} ET".encode("ascii")
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content
        )
        kids.append(len(objects))
    refs = " ".join(f"{k} 0 R" for k in kids).encode()
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (refs, len(kids))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    encrypt = _ENCRYPT if encrypted else b""
    out += b"trailer\n<< %s/Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        encrypt,
        len(objects) + 1,
        xref,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path
