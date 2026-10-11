"""Build the hard eval corpus: the whole public corpus plus documents built to be hard to tell
apart or hard to read (PLAN.md → Evaluation → Harder tier).

    uv run python eval/hard/build_corpus.py [OUT_DIR]   # default: data/eval-hard
    uv run python eval/hard/build_corpus.py --regions   # rewrite "region" in queries.jsonl

The public documents are written first, by the public generator, unchanged: they act as
distractors. The hard documents use their own seed, so the public files stay byte-identical.
Every fact is invented. The questions in queries.jsonl are written against the text below, so
change both together. Groups (tags): `version` (near-identical editions), `degraded` (poor scans,
small print, a dense timetable, charts without value labels), `long` (a 50-page manual of
templated sections); `multi`, `paraphrase` and `near-miss` are kinds of question, not documents.

Hardened once (PLAN.md → Harder tier results): the first run had no misses on `long`, `multi`
or `near-miss`, so the manual gained decoy sections (near-twins of the asked-about areas, and a
planned Cold store E), and lures were added next to the facts the near-miss questions ask for.
"""

import importlib.util
import json
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "public_build_corpus", ROOT / "eval" / "public" / "build_corpus.py"
)
assert _spec is not None and _spec.loader is not None
public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(public)

from pdf_fixture import write_pdf  # noqa: E402  # ty: ignore[unresolved-import]

SEED_HARD = 20261010
SEED_DECOYS = SEED_HARD + 1  # its own stream, so the decoys leave the other facts alone
Box = public.Box
Layout = public.Layout

# version: twins of public documents, same layout and wording, different figures
TEXT_PDFS: dict[str, list[list[str]]] = {
    "reports/annual-review-2024.pdf": [
        [
            "Halvorsen Marine - Annual Review 2024",
            "Chief executive Ingrid Solberg presented the results for the year.",
            "Total revenue reached 196.9 million euros, up 4 percent on the year before.",
            "Headcount ended the year at 1,104 employees across three offices.",
            "Restated figures for 2023 are published separately.",  # near-miss lure
        ],
        [
            "Fleet operations",
            "The company operated 35 vessels during the year.",
            "Average fuel consumption fell to 22.8 tonnes per day after hull coating upgrades.",
            "The newest ship, MV Storm Petrel, entered service in September.",
        ],
        [
            "Safety",
            "The lost time injury frequency was 0.61 per million hours worked.",
            "Three emergency drills per quarter are mandatory on every vessel.",
            "This year's safety award went to the crew of MV Lomvi.",
        ],
        [
            "Outlook",
            "The board approved a dividend of 1.45 euros per share.",
            "Capital spending for 2025 is budgeted at 39 million euros,",
            "mostly for one LNG-ready ferry.",
        ],
    ],
    # the public agreement has no response time: the amendment adds one and changes the fee
    "contracts/service-agreement-2026.pdf": [
        [
            "Service agreement between Brightwater Foods and Nordlys Cleaning - 2026 amendment",
            "Contract reference SA-7713. Monthly fee 3,650 euros.",
            "Emergency call-out response time: 4 hours.",
            "Fees for 2027 will be agreed in December 2026.",  # near-miss lure
        ],
        [
            "Order PO-48362",
            "Deep cleaning of the Leeds warehouse, total 5,480 euros.",
            "Scheduled for the first week of March.",
            "This order replaces the cancelled order PO-48214.",  # near-miss lure
        ],
    ],
}

TABLES: dict[str, tuple[str, list[str], list[list[str]]]] = {
    # near-miss lure: a roster with weekday shifts, but none on Friday
    "tables/weekday-roster.png": (
        "Weekday staff roster (Monday to Thursday)",
        ["Shift", "Staff"],
        [
            ["Monday evening", "Annika"],
            ["Tuesday evening", "Rafael"],
            ["Wednesday evening", "Leila"],
            ["Thursday evening", "Bogdan"],
        ],
    ),
    "tables/price-list-winter.png": (
        "Corner Cafe winter price list",
        ["Item", "Price (pounds)"],
        [
            ["Espresso", "2.20"],
            ["Flat white", "3.25"],
            ["Oat latte", "3.60"],
            ["Chai", "3.45"],
            ["Croissant", "2.95"],
        ],
    ),
}

# long: a cover page, then one templated section per page (section N on page N + 1)
MANUAL = "manuals/operations-manual.pdf"
MANUAL_AREAS = [
    *(f"Cold store {c}" for c in "ABCD"),
    *(f"Loading bay {n}" for n in range(1, 7)),
    "Boiler room North",
    "Boiler room South",
    *(f"Pump house {n}" for n in range(1, 4)),
    "Compressor room 1",
    "Compressor room 2",
    "Forklift charging bay East",
    "Forklift charging bay West",
    *(f"Packing line {n}" for n in range(1, 6)),
    "Mezzanine store North",
    "Mezzanine store South",
    "Chemical store",
    "Workshop",
    "Server room",
    "Canteen kitchen",
    "Roof plant deck",
    "Goods-in office",
    "Despatch office",
    "Generator room",
    "Sprinkler valve room",
    "Electrical substation",
    "Waste compactor yard",
    "Cleaning store",
    "Gatehouse",
]
MANUAL_INTERVALS = [7, 10, 14, 21, 28, 30, 45, 60, 90]
MANUAL_PEOPLE = [
    "Oskar Lind",
    "Mei Tanaka",
    "Ruth Adeyemi",
    "Callum Price",
    "Ines Duarte",
    "Pavel Novak",
    "Hana Sato",
    "Dev Malhotra",
]
MANUAL_TEAMS = ["Facilities", "Maintenance", "Health and safety", "Operations"]


def manual_sections(rng: random.Random) -> list[tuple[str, int, str, str, str]]:
    """(area, interval in days, person, team, isolation point) per section, in order."""
    codes = rng.sample(range(100, 1000), len(MANUAL_AREAS))
    return [
        (
            area,
            rng.choice(MANUAL_INTERVALS),
            rng.choice(MANUAL_PEOPLE),
            rng.choice(MANUAL_TEAMS),
            f"IP-{code}",
        )
        for area, code in zip(MANUAL_AREAS, codes, strict=True)
    ]


# decoys, appended after section 39 so every earlier page keeps its number: near-twins of the
# areas the questions ask about, then a planned store with nothing to report
MANUAL_DECOYS = {  # decoy -> the section it twins
    "Cold store A annex": "Cold store A",
    "Cold store B annex": "Cold store B",
    "Cold store C annex": "Cold store C",
    "Loading bay 3A": "Loading bay 3",
    "Boiler room North annex": "Boiler room North",
    "Boiler room South annex": "Boiler room South",
    "Packing line 4B": "Packing line 4",
    "Server room annex": "Server room",
    "Generator room annex": "Generator room",
}


def manual_decoys(
    rng: random.Random, sections: list[tuple[str, int, str, str, str]]
) -> list[tuple[str, int, str, str, str]]:
    """A decoy per entry of MANUAL_DECOYS, every value different from its twin's."""
    by_area = {s[0]: s for s in sections}
    used = {s[4] for s in sections}
    out = []
    for area, twin in MANUAL_DECOYS.items():
        _, days, person, team, _ = by_area[twin]
        code = rng.choice(sorted(set(range(100, 1000)) - {int(c[3:]) for c in used}))
        used.add(f"IP-{code}")
        out.append(
            (
                area,
                rng.choice([d for d in MANUAL_INTERVALS if d != days]),
                rng.choice([p for p in MANUAL_PEOPLE if p != person]),
                rng.choice([t for t in MANUAL_TEAMS if t != team]),
                f"IP-{code}",
            )
        )
    return out


MANUAL_PLANNED = [  # near-miss lure: Cold store E is mentioned, but has no interval
    "Section 49: Cold store E",
    "Status: planned, not yet in service.",
    "Inspection interval, responsible team and isolation point will be added to this manual "
    "when the store opens.",
]


def manual_pages(sections: list[tuple[str, int, str, str, str]]) -> list[list[str]]:
    cover = [
        "Kestrel Packaging - Site operations manual",
        "Revision 7, issued 3 February 2026.",
        "Each section covers one area of the Leeds site: how often it is inspected, who is "
        "responsible and where to isolate it.",
    ]
    return [
        cover,
        *(
            [
                f"Section {n}: {area}",
                f"Inspection interval: every {days} days.",
                f"Responsible: {person}, {team} team.",
                f"Isolation point: {code}.",
                "Record each inspection in the site log and report any defect to the duty "
                "manager on the same day.",
            ]
            for n, (area, days, person, team, code) in enumerate(sections, start=1)
        ),
        MANUAL_PLANNED,
    ]


# degraded: a faded low-resolution scan, and a page of small print
FADED_INVOICE = "scans/faded-invoice.pdf"
FADED_LINES = [
    "Tidewater Plumbing",
    "Invoice INV-83317",
    "Replace kitchen mixer tap and isolation valve",
    "Labour: 2.5 hours",
    "Total due: 173.40 pounds",
    "Payment due by 28 November",
    "Late payment charges apply as set out in our terms.",  # near-miss lure
]

SMALL_PRINT = "scans/small-print.pdf"
SMALL_PRINT_PARAS = [
    "Fernhill Self Storage - Terms and conditions",
    "1. These terms apply to every unit let by Fernhill Self Storage Ltd. The customer named on "
    "the booking form is responsible for the unit, for anything kept in it and for anyone given "
    "a key or access code. Units are let for a minimum of two weeks and are charged weekly in "
    "advance.",
    "2. Rent is payable by direct debit on the first working day of each week. If a payment "
    "fails, a late payment charge of 12.50 pounds is added to the account and access may be "
    "suspended until the balance is cleared. Persistent arrears are handled under clause 7.",
    "3. Units can be accessed between 6 am and 10 pm every day of the year, using the personal "
    "code issued at check-in. The office is staffed on weekdays only. Vehicles must not be left "
    "in the loading area for more than 30 minutes.",
    "4. Customers must insure the goods they store. Fernhill offers contents cover from 2 pounds "
    "per week; customers who arrange their own cover must show the policy on request. Fernhill "
    "is not liable for loss or damage caused by damp, pests or the customer's own lock failing.",
    "5. Either party may end the agreement by giving 28 days written notice. Rent paid for any "
    "period after the agreement ends is refunded within 14 days, less any amount still owed. "
    "The unit must be left empty and clean, and the lock removed.",
    "6. Hazardous, perishable or illegal goods must not be stored. Fernhill may open a unit "
    "without notice in an emergency, or when required by law, and will fit a new lock "
    "afterwards at the customer's cost.",
    "7. If rent is unpaid for 60 days, Fernhill may sell the contents after giving notice in "
    "writing to the last known address, and will pay any surplus to the customer.",
]


def degraded_scan(
    paragraphs: list[str],
    rng: random.Random,
    *,
    dpi: int,
    sizes: tuple[int, int],  # title, body (pixels)
    width: int,  # characters per line
    ink: int,
    specks: int,
    angle: float,
    blur: float,
) -> tuple[Image.Image, list[tuple[str, Box]]]:
    """An A4 page at `dpi`: like the public scans, with every knob turned (rotation in degrees,
    anticlockwise)."""
    w, h = round(8.27 * dpi), round(11.69 * dpi)
    margin, y = round(0.8 * dpi), round(1.07 * dpi)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    drawn: list[tuple[str, Box]] = []
    for i, para in enumerate(paragraphs):
        size = sizes[0] if i == 0 else sizes[1]
        parts = []
        for part in textwrap.wrap(para, width):
            draw.text((margin, y), part, fill=ink, font=public.font(size))
            parts.append(draw.textbbox((margin, y), part, font=public.font(size)))
            y += round(size * 1.35)
        drawn.append((para, public.union(parts)))
        y += round(size * 0.6)
    for _ in range(specks):
        draw.point((rng.randrange(w), rng.randrange(h)), fill=rng.randrange(60, 200))
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    boxes = [(text, public.relative(public.rotated(box, angle, w, h), w, h)) for text, box in drawn]
    return img.filter(ImageFilter.GaussianBlur(blur)), boxes


def save_scan_pdf(img: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(
        path,
        "PDF",
        resolution=float(dpi),
        creationDate=public.PDF_DATE,
        modDate=public.PDF_DATE,
    )


# degraded: a dense timetable with thin rules
TIMETABLE = "tables/bus-timetable.png"
TIMETABLE_STOPS = [
    "Bus station",
    "Market Street",
    "Westgate",
    "St Anne's Church",
    "Harbour Road",
    "Fish Quay",
    "Lighthouse Lane",
    "Coastguard Cottages",
    "Bramley Park",
    "Hospital",
    "College",
    "Orchard Way",
    "Millbrook",
    "Hilltop Farm",
    "Kingsmead",
    "Ferry Terminal",
    "Sandhaven",
    "Retail Park",
]
TIMETABLE_BUSES = 5
TIMETABLE_NOTE = "Bus 6 runs on Saturdays only and is not shown."  # near-miss lure


def timetable_rows(rng: random.Random) -> list[list[str]]:
    """Each bus leaves 25 minutes after the one before; the stops are 2 to 5 minutes apart."""
    gaps = [0, *(rng.randint(2, 5) for _ in TIMETABLE_STOPS[1:])]
    rows = []
    for s, stop in enumerate(TIMETABLE_STOPS):
        offset = sum(gaps[: s + 1])
        times = [10 * 60 + 5 + 25 * b + offset for b in range(TIMETABLE_BUSES)]
        rows.append([stop, *(f"{t // 60:02d}:{t % 60:02d}" for t in times)])
    return rows


def thin_table(
    title: str, header: list[str], rows: list[list[str]], note: str
) -> tuple[Image.Image, list[tuple[str, Box]]]:
    """A dense table: small text, 1 px light rules, a wide first column, a note underneath. One
    box per body cell, and one for the note."""
    widths = [300] + [130] * (len(header) - 1)
    row_h, pad, y0 = 34, 40, 100
    w = pad * 2 + sum(widths)
    h = y0 + row_h * (len(rows) + 1) + pad + 40
    lefts = [pad + sum(widths[:c]) for c in range(len(widths) + 1)]
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((pad, 35), title, fill="black", font=public.font(30))
    for r, cells in enumerate([header, *rows]):
        y = y0 + r * row_h
        for c, cell in enumerate(cells):
            d.text(
                (lefts[c] + 10, y + row_h / 2),
                cell,
                fill="black",
                font=public.font(20),
                anchor="lm",
            )
        d.line([(pad, y + row_h), (w - pad, y + row_h)], fill="#c8c8c8", width=1)
    for x in lefts:
        d.line([(x, y0), (x, y0 + row_h * (len(rows) + 1))], fill="#c8c8c8", width=1)
    d.line([(pad, y0), (w - pad, y0)], fill="#c8c8c8", width=1)
    note_at = (pad, y0 + row_h * (len(rows) + 1) + 15)
    d.text(note_at, note, fill="#444444", font=public.font(20))
    note_box = public.relative(d.textbbox(note_at, note, font=public.font(20)), w, h)
    boxes = [
        (
            cell,
            public.relative((lefts[c], y0 + r * row_h, lefts[c + 1], y0 + (r + 1) * row_h), w, h),
        )
        for r, cells in enumerate(rows, start=1)
        for c, cell in enumerate(cells)
    ]
    return img, [*boxes, (note, note_box)]


# degraded: charts with gridlines every 10 and no value labels; values sit between gridlines
CHARTS: dict[str, tuple[str, str, str, list[tuple[str, int]], str]] = {
    "charts/rainfall-unlabelled.png": (
        "Monthly rainfall, Thornbury 2025",
        "mm",
        "bar",
        [
            ("Jan", 87),
            ("Feb", 64),
            ("Mar", 76),
            ("Apr", 46),
            ("May", 53),
            ("Jun", 38),
            ("Jul", 27),
            ("Aug", 34),
        ],
        "September to December: no readings, the rain gauge was under repair.",  # lure
    ),
    "charts/occupancy-unlabelled.png": (
        "Hotel occupancy, Seaview Hotel 2025",
        "percent of rooms",
        "line",
        [
            ("Jan", 54),
            ("Feb", 58),
            ("Mar", 66),
            ("Apr", 73),
            ("May", 77),
            ("Jun", 86),
            ("Jul", 93),
            ("Aug", 88),
            ("Sep", 75),
            ("Oct", 64),
            ("Nov", 57),
            ("Dec", 69),
        ],
        "",
    ),
}


def unlabelled_chart(
    title: str, ylabel: str, kind: str, data: list[tuple[str, int]], note: str
) -> tuple[Image.Image, list[tuple[str, Box]]]:
    """A 0-100 chart with a labelled gridline every 10, no value labels and an optional note
    under the axis. One box per column: its data mark (the top edge of the bar, or the point),
    labelled with the category only, since the value isn't drawn; and one for the note."""
    w, h = 1200, 800
    left, right, top, bottom = 140, 60, 130, 120
    plot_w, plot_h = w - left - right, h - top - bottom
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((left, 40), title, fill="black", font=public.font(40))
    d.text((20, top - 50), ylabel, fill="#444444", font=public.font(24))
    for tick in range(0, 101, 10):
        y = h - bottom - plot_h * tick / 100
        d.line([(left, y), (w - right, y)], fill="#d0d0d0", width=1)
        d.text((left - 15, y), str(tick), fill="#444444", font=public.font(22), anchor="rm")
    d.line([(left, top), (left, h - bottom), (w - right, h - bottom)], fill="black", width=3)
    step = plot_w / len(data)
    points = []
    boxes: list[tuple[str, Box]] = []
    for i, (label, value) in enumerate(data):
        cx = left + step * (i + 0.5)
        y = h - bottom - plot_h * value / 100
        points.append((cx, y))
        if kind == "bar":
            d.rectangle([cx - step * 0.3, y, cx + step * 0.3, h - bottom], fill="#2f6f8f")
            mark = (cx - step * 0.3, y - 8, cx + step * 0.3, y + 8)
        else:
            mark = (cx - 10, y - 10, cx + 10, y + 10)
        d.text((cx, h - bottom + 20), label, fill="black", font=public.font(26), anchor="mt")
        boxes.append((label, public.relative(mark, w, h)))
    if kind == "line":
        d.line(points, fill="#b0452f", width=4)
        for x, y in points:
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill="#b0452f")
    if note:
        d.text((left, h - 50), note, fill="#444444", font=public.font(22))
        box = d.textbbox((left, h - 50), note, font=public.font(22))
        boxes.append((note, public.relative(box, w, h)))
    return img, boxes


MARKDOWN: dict[str, str] = {  # near-miss lure: sprint 15 is planned, its velocity isn't known
    "docs/sprint-15-plan.md": """\
# Sprint 15 plan

## Dates

Sprint 15 starts on 19 November and ends on 2 December.

## Goal

Ship offline sync to the beta group.

## Reporting

Velocity and open issues are reported at the sprint review.
""",
}


def build_hard(out: Path, layout: Layout | None = None) -> list[Path]:
    """Write only the hard documents under `out` (and their layout)."""
    rng = random.Random(SEED_HARD)
    written: list[Path] = []
    drawn: Layout = layout if layout is not None else {}
    # the facts first, so `manual_sections` and `timetable_rows` alone reproduce them
    sections, timetable = manual_sections(rng), timetable_rows(rng)
    decoys = manual_decoys(random.Random(SEED_DECOYS), sections)
    text_pdfs = {**TEXT_PDFS, MANUAL: manual_pages(sections + decoys)}
    for rel, pages in text_pdfs.items():
        lines = [[part for para in page for part in textwrap.wrap(para, 80)] for page in pages]
        written.append(write_pdf(out / rel, lines))
        for n, page in enumerate(pages, start=1):
            drawn[(rel, n)] = public.text_page_boxes(page)

    img, boxes = degraded_scan(
        FADED_LINES,
        rng,
        dpi=100,
        sizes=(31, 24),
        width=50,
        ink=120,
        specks=9000,
        angle=rng.choice([-1, 1]) * rng.uniform(3.0, 4.0),
        blur=1.2,
    )
    save_scan_pdf(img, out / FADED_INVOICE, 100)
    drawn[(FADED_INVOICE, 1)] = boxes
    img, boxes = degraded_scan(
        SMALL_PRINT_PARAS,
        rng,
        dpi=150,
        sizes=(30, 21),
        width=95,
        ink=40,
        specks=4000,
        angle=rng.uniform(-1.2, 1.2),
        blur=0.6,
    )
    save_scan_pdf(img, out / SMALL_PRINT, 150)
    drawn[(SMALL_PRINT, 1)] = boxes
    written += [out / FADED_INVOICE, out / SMALL_PRINT]

    header = ["Stop", *(f"Bus {b}" for b in range(1, TIMETABLE_BUSES + 1))]
    for rel, (img, boxes) in [
        *((rel, public.table(*spec)) for rel, spec in TABLES.items()),
        (TIMETABLE, thin_table("Route 7 weekday timetable", header, timetable, TIMETABLE_NOTE)),
        *((rel, unlabelled_chart(*spec)) for rel, spec in CHARTS.items()),
    ]:
        public.save_png(img, out / rel)
        drawn[(rel, None)] = boxes
        written.append(out / rel)
    for rel, text in MARKDOWN.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
        written.append(out / rel)
    return written


def build(out: Path, layout: Layout | None = None) -> list[Path]:
    """The whole hard corpus: the public documents, then the hard ones."""
    return public.build(out, layout) + build_hard(out, layout)


def write_regions(queries_path: Path, scratch: Path) -> int:
    """Set (or clear) each query's `region` from the layout. Returns how many have one."""
    layout: Layout = {}
    build(scratch, layout)
    rows = [json.loads(line) for line in queries_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row.pop("region", None)
        if (region := public.find_region(layout, row)) is not None:
            row["region"] = region
    queries_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return sum("region" in row for row in rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["--regions"]:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            n = write_regions(Path(__file__).with_name("queries.jsonl"), Path(tmp))
        print(f"{n} queries have a region")
        sys.exit()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "eval-hard"
    files = build(out)
    print(f"wrote {len(files)} documents to {out}")
