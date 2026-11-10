"""Record the demo video's shots with real models (PLAN.md → Demo video → Record step).

    uv run python scripts/demo_video/record.py --out <folder>     # ~10 min

Runs `qwn ui` on the hard eval corpus and drives it in Chrome (Playwright), one JPEG screenshot
per step (~24 fps, native 1920x1080: Gate C), then `qwn chat` in a pty, then renders the
captions, labels, end card and thumbnail title as local HTML pages. Writes frames, audio and
events.json to <folder>; `assemble.py` cuts the video from it. It saves no conversation.

Every answer on screen is checked against its eval query, the refusal against ABSTAIN_TEXT, the
voice transcript against the query's words, and every frame for the home folder's path: the
step fails, naming the shot and why.
"""

import argparse
import base64
import codecs
import json
import os
import re
import select
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import ViewportSize

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import timeline  # noqa: E402

VIEW: "ViewportSize" = {"width": 1280, "height": 720}  # x1.5 = 1920x1080
SCALE = 1.5
CHAT_INPUT = '[data-testid="stChatInput"] textarea'
MESSAGE = '[data-testid="stChatMessage"]'
STATUS = '[data-testid="stStatusWidget"]'
PASSAGE = re.compile(r"Supporting passage \(approximate\)|Couldn't pinpoint")
WARMUP = "hard-025"  # an answer no shot uses
PROMPT = "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} "
THEME = {"bg": "#1B1A18", "panel": "#262420", "text": "#ECE8E1", "muted": "#A49E93",
         "border": "#3B3833", "link": "#5CC9BC"}  # fmt: skip  # .streamlit/config.toml dark

# Rewrites the home folder to ~ as the page changes; __qwnLeak() names any text that still
# holds it, so a frame with the path in it is never kept.
HOME_SCRIPT = """
(() => {
  const HOME = %s;
  const ATTRS = ["title", "alt", "aria-label"];
  const fix = (n) => {
    if (n.nodeType === 3) {
      if (n.nodeValue.includes(HOME)) n.nodeValue = n.nodeValue.split(HOME).join("~");
      return;
    }
    if (n.nodeType !== 1) return;
    for (const a of ATTRS) {
      const v = n.getAttribute(a);
      if (v && v.includes(HOME)) n.setAttribute(a, v.split(HOME).join("~"));
    }
    for (const c of n.childNodes) fix(c);
  };
  const watch = () => {
    fix(document.documentElement);
    new MutationObserver((ms) => {
      for (const m of ms) {
        if (m.type === "characterData") fix(m.target);
        else if (m.type === "attributes") fix(m.target);
        else m.addedNodes.forEach(fix);
      }
    }).observe(document.documentElement, {
      subtree: true, childList: true, characterData: true, attributes: true,
      attributeFilter: ATTRS,
    });
  };
  if (document.documentElement) watch();
  else document.addEventListener("DOMContentLoaded", watch);
  window.__qwnLeak = () => {
    if (document.title.includes(HOME)) return "title: " + document.title;
    const w = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_ALL);
    for (let n = w.currentNode; n; n = w.nextNode()) {
      if (n.nodeType === 3 && n.nodeValue.includes(HOME)) return n.nodeValue.slice(0, 120);
      if (n.nodeType === 1)
        for (const a of ATTRS) {
          const v = n.getAttribute(a);
          if (v && v.includes(HOME)) return a + ": " + v.slice(0, 120);
        }
    }
    return null;
  };
})();
"""


class RecordError(RuntimeError):
    pass


def words(text: str) -> set[str]:
    return set(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def queries(name: str) -> dict[str, dict]:
    lines = (ROOT / "eval" / name / "queries.jsonl").read_text().splitlines()
    return {(q := json.loads(line))["id"]: q for line in lines if line.strip()}


# --- capture -----------------------------------------------------------------------------------


class Recorder:
    """Back-to-back screenshots of one shot; waits poll their condition between frames."""

    def __init__(self, page, out: Path, *, check_home: bool = True) -> None:
        self.page, self.out, self.check_home = page, out, check_home
        self.shots: dict[str, dict] = {}

    def start(self, shot_id: str) -> None:
        self.id = shot_id
        self.dir = self.out / "frames" / shot_id
        self.dir.mkdir(parents=True)
        self.frames: list[float] = []
        self.events: dict[str, float] = {}
        self.framing: Callable[[], None] | None = None  # camera work before each frame
        self.t0 = time.monotonic()
        self.shoot()
        self.mark("start")

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 4)

    def shoot(self) -> None:
        if self.check_home and (leak := self.page.evaluate("window.__qwnLeak?.() ?? null")):
            raise RecordError(f"shot {self.id}: the home folder is on screen: {leak!r}")
        if self.framing is not None:
            self.framing()
        before = time.monotonic()
        data = self.page.screenshot(type="jpeg", quality=95)
        t = (before + time.monotonic()) / 2 - self.t0
        (self.dir / f"{len(self.frames):05d}.jpg").write_bytes(data)
        self.frames.append(round(t, 4))

    def mark(self, name: str) -> None:
        self.events[name] = self.now()

    def until(self, done: Callable[[], bool], what: str, timeout: float = 300) -> None:
        deadline = time.monotonic() + timeout
        while not done():
            if time.monotonic() > deadline:
                raise RecordError(f"shot {self.id}: timed out waiting for {what}")
            self.shoot()

    def hold(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.shoot()

    def type(self, text: str, delay: float) -> None:
        for ch in text:
            nxt = time.monotonic() + delay
            self.page.keyboard.type(ch)
            self.shoot()
            while time.monotonic() < nxt:
                self.shoot()

    def finish(self, **extra) -> None:
        self.mark("end")
        self.shoot()
        self.shots[self.id] = {"frames": self.frames, "events": self.events, **extra}


# --- the UI shots ------------------------------------------------------------------------------


def wait_ready(page) -> None:
    page.wait_for_function(
        "() => { const t = document.querySelector('[data-testid=stChatInput] textarea');"
        " return t && !t.disabled; }",
        timeout=600_000,
    )


def answers(page) -> int:
    return page.locator(MESSAGE).get_by_text("Searched in:").count()


def last_text(page) -> str:
    return page.locator(MESSAGE).last.inner_text()


def new_conversation(page) -> None:
    btn = page.get_by_role("button", name="New conversation")
    if btn.count():
        btn.click()
        page.wait_for_function(
            "() => !document.querySelector('[data-testid=stChatMessage]')", timeout=30_000
        )
    wait_ready(page)
    reset = page.get_by_role("button", name="Reset", exact=True)
    if reset.count():  # the last recording's waveform: every shot starts clean
        reset.first.click()
        time.sleep(0.5)
    page.mouse.move(5, 5)
    time.sleep(0.5)


def frame_latest(page) -> None:
    """Camera work: scroll the latest question to just under the caption, so its answer shows."""
    page.evaluate(
        """() => {
          const m = [...document.querySelectorAll('[data-testid=stChatMessage]')];
          const el = m[Math.max(m.length - 2, 0)];
          if (!el) return;
          el.style.scrollMarginTop = '84px';
          el.scrollIntoView({block: 'start'});
        }"""
    )


def check_answer(shot_id: str, text: str, q: dict) -> None:
    from qwn.prompts import ABSTAIN_TEXT

    if not q["expected"]:
        if ABSTAIN_TEXT not in text:
            raise RecordError(f"shot {shot_id}: expected the refusal, got {text[:200]!r}")
        return
    missing = [s for s in q["answer_contains"] if s.lower() not in text.lower()]
    if missing:
        raise RecordError(f"shot {shot_id}: the answer lacks {missing}: {text[:200]!r}")


def ask(rec: Recorder, page, question: str, tl, *, typed: bool = True) -> None:
    page.locator(CHAT_INPUT).click()
    if typed:
        rec.type(question, tl.type_delay_s)
    else:
        page.locator(CHAT_INPUT).fill(question)
        rec.shoot()
    page.keyboard.press("Enter")
    rec.shoot()


def chat_shot(rec, page, shot, q, tl) -> None:
    new_conversation(page)
    rec.start(shot.id)
    rec.hold(shot.lead_s)
    ask(rec, page, q["query"], tl)
    rec.mark("sent")
    rec.until(lambda: answers(page) >= 1, "the answer")
    rec.mark("answer")
    rec.framing = lambda: frame_latest(page)
    rec.hold(shot.hold_s)
    check_answer(shot.id, last_text(page), q)
    rec.finish()


def frame_audio(page) -> None:
    """Camera work: keep the spoken answer's player above the chat input."""
    page.evaluate(
        """() => {
          const m = [...document.querySelectorAll('[data-testid=stChatMessage]')];
          const a = m.length && m[m.length - 1].querySelector('audio');
          if (!a) return;
          a.style.scrollMarginBottom = '150px';
          a.scrollIntoView({block: 'end'});
        }"""
    )


def open_page_shot(rec, page, shot, q) -> None:
    rec.start(shot.id)
    rec.hold(shot.lead_s)
    page.locator(MESSAGE).last.get_by_role("button", name="Open page").first.click()
    rec.mark("click")
    dialog = page.get_by_role("dialog")
    rec.until(
        lambda: dialog.get_by_text(PASSAGE).count() > 0 and not page.locator(STATUS).is_visible(),
        "the supporting passage",
    )
    rec.mark("dialog")
    page.mouse.move(5, 5)
    if dialog.get_by_text("Supporting passage (approximate)").count() == 0:
        raise RecordError(f"shot {shot.id}: the dialog couldn't pinpoint the passage")
    # the zoom's focus: the page image's middle (the whole row, its label too) at the height of
    # the answer's region (the eval's, which the outline overlaps)
    box = dialog.locator('img[alt$="outlined"]').first.bounding_box()
    if box is None or "region" not in q:
        raise RecordError(f"shot {shot.id}: no page image or region to zoom on")
    _, y0, _, y1 = q["region"]
    focus = [
        round((box["x"] + box["width"] / 2) * SCALE, 1),
        round((box["y"] + box["height"] * (y0 + y1) / 2) * SCALE, 1),
    ]
    rec.hold(shot.hold_s)
    rec.finish(focus=focus, image=[round(box[k] * SCALE, 1) for k in ("x", "y", "width", "height")])
    page.keyboard.press("Escape")
    dialog.wait_for(state="hidden", timeout=10_000)


def followup_shot(rec, page, shot, turn, first, tl) -> None:
    new_conversation(page)
    rec.start(shot.id)
    rec.hold(shot.lead_s)
    ask(rec, page, turn["history"][0]["question"], tl, typed=False)
    rec.mark("sent1")
    rec.until(lambda: answers(page) >= 1, "the first answer")
    rec.mark("answer1")
    rec.framing = lambda: frame_latest(page)
    check_answer(shot.id, last_text(page), first)
    rec.hold(shot.mid_s)
    ask(rec, page, turn["query"], tl)
    rec.mark("sent2")
    rec.until(lambda: answers(page) >= 2, "the follow-up's answer")
    rec.mark("answer2")
    rec.framing = lambda: frame_latest(page)
    rec.hold(shot.hold_s)
    text = last_text(page)
    check_answer(shot.id, text, turn)
    if "Searched for:" not in text:
        raise RecordError(f"shot {shot.id}: the follow-up shows no rewrite (Searched for:)")
    rec.finish()


def spoken(rec: Recorder | None, page, wav_s: float) -> None:
    """Record the fake microphone's file: Record, the file's length, Stop recording."""
    page.get_by_role("button", name="Record", exact=True).click()
    if rec is not None:
        rec.mark("rec_start")
        rec.hold(wav_s + 0.3)
    else:
        time.sleep(wav_s + 0.3)
    page.get_by_role("button", name="Stop recording").click()


def voice_shot(rec, page, shot, q, wav_s: float, url: str, out: Path) -> None:
    new_conversation(page)
    rec.start(shot.id)
    rec.hold(shot.lead_s)
    spoken(rec, page, wav_s)
    rec.mark("rec_stop")
    rec.until(lambda: answers(page) >= 1, "the spoken question's answer")
    rec.mark("answer")
    rec.framing = lambda: frame_latest(page)
    audio = page.locator(MESSAGE).last.locator("audio")
    rec.until(lambda: audio.count() > 0 and bool(audio.first.get_attribute("src")), "the speech")
    rec.mark("audio")
    rec.framing = lambda: frame_audio(page)
    src = audio.first.get_attribute("src") or ""
    resp = page.request.get(urllib.parse.urljoin(url + "/", src))
    if not resp.ok:
        raise RecordError(f"shot {shot.id}: couldn't fetch the spoken answer ({resp.status})")
    (out / "answer.wav").write_bytes(resp.body())
    with wave.open(str(out / "answer.wav")) as w:
        answer_s = w.getnframes() / w.getframerate()
    rec.hold(answer_s)
    rec.mark("audio_end")
    rec.hold(shot.hold_s)
    texts = [m.inner_text() for m in page.locator(MESSAGE).all()]
    user = next((t for t in texts if "Spoken question" in t), "")
    if missing := words(q["query"]) - words(user.replace("Spoken question", "")):
        raise RecordError(
            f"shot {shot.id}: the transcript {user!r} lacks {sorted(missing)}: make the WAV "
            "again (make_question.py) or pick another preset"
        )
    check_answer(shot.id, texts[-1], q)
    rec.finish(audio={"question": "question.wav", "answer": "answer.wav"})


def library_shot(rec, page, shot) -> None:
    page.get_by_role("link", name="Library").click()
    page.locator('[data-testid="stDataFrame"]').first.wait_for(timeout=60_000)
    page.mouse.move(5, 5)
    time.sleep(1.5)
    rec.start(shot.id)
    rec.hold(shot.hold_s)
    rec.finish()


# --- local pages: overlays, cards, the terminal replay -----------------------------------------


def font_face() -> str:
    import streamlit

    media = Path(streamlit.__file__).parent / "static" / "static" / "media"
    fonts = sorted(media.glob("SourceSansVF-Upright*.woff2"))
    if not fonts:
        raise RecordError(f"Streamlit's Source Sans isn't in {media}")
    data = base64.b64encode(fonts[0].read_bytes()).decode()  # file:// fonts are cross-origin
    return (
        "@font-face { font-family: 'Source Sans'; "
        f"src: url(data:font/woff2;base64,{data}) format('woff2'); font-weight: 200 900; }}"
    )


ICONS = {
    "label": '<svg viewBox="0 0 24 24" width="22" height="22"><path fill="currentColor" '
    'd="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/></svg>',
    "tag": '<svg viewBox="0 0 24 24" width="22" height="22"><path fill="currentColor" '
    'd="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 '
    '6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>',
}


def overlay_html(style: str, text: str, center_x: float) -> str:
    from html import escape

    common = (
        f"position:absolute;display:flex;align-items:center;gap:10px;color:{THEME['text']};"
        "background:rgba(21,20,18,.86);border:1px solid rgba(236,232,225,.16);"
        "border-radius:12px;font-family:'Source Sans';"
    )
    if style == "caption":
        box = (
            f"{common}top:18px;left:{center_x}px;transform:translateX(-50%);max-width:880px;"
            "padding:9px 22px;font-size:30px;font-weight:600;white-space:nowrap;"
        )
        body = escape(text)
    else:
        bottom = 30 if style == "label" else 78
        box = (
            f"{common}left:28px;bottom:{bottom}px;padding:7px 14px;font-size:20px;font-weight:600;"
        )
        body = ICONS[style] + f"<span>{escape(text)}</span>"
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{font_face()}"
        "html,body{margin:0;background:transparent;width:1280px;height:720px;overflow:hidden}"
        f'</style></head><body><div style="{box}">{body}</div></body></html>'
    )


def card_html(shot) -> str:
    from html import escape

    lines = "".join(f"<p>{escape(line)}</p>" for line in shot.lines)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{font_face()}
html,body{{margin:0;width:1280px;height:720px;background:{THEME["bg"]};color:{THEME["text"]};
font-family:'Source Sans';display:flex;align-items:center;justify-content:center}}
main{{text-align:center}} h1{{font-size:120px;font-weight:700;margin:0 0 18px;letter-spacing:-2px}}
p{{font-size:32px;margin:8px 0;color:{THEME["text"]}}}
.url{{margin-top:40px;font-size:30px;color:{THEME["link"]};font-weight:600}}
</style></head><body><main><h1>qwn</h1>{lines}<p class="url">{escape(shot.url)}</p></main>
</body></html>"""


def title_html(tl) -> str:
    from html import escape

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{font_face()}
html,body{{margin:0;width:1280px;height:720px;background:transparent;overflow:hidden}}
div{{position:absolute;left:0;right:0;bottom:0;height:300px;padding:0 64px;
display:flex;flex-direction:column;justify-content:center;font-family:'Source Sans';
color:{THEME["text"]};background:linear-gradient(to top,rgba(21,20,18,.97) 60%,rgba(21,20,18,0))}}
h1{{font-size:112px;font-weight:700;margin:40px 0 0;letter-spacing:-2px}}
p{{font-size:44px;margin:0;font-weight:600}}
</style></head><body><div><h1>{escape(tl.title)}</h1><p>{escape(tl.subtitle)}</p></div>
</body></html>"""


ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]")


def clean_log(log: list[tuple[float, bytes]]) -> list[tuple[float, str]]:
    """The pty log as text: ANSI codes out, CRLF to LF; one entry per original chunk."""
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    out, pending = [], ""
    for t, chunk in log:
        text = pending + dec.decode(chunk)
        pending = ""
        if text.endswith(("\x1b", "\r")):  # may continue in the next chunk
            pending, text = text[-1], text[:-1]
        text = ANSI.sub("", text).replace("\r\n", "\n")
        if text:
            out.append((t, text))
    if pending:
        out.append((log[-1][0], pending))
    return out


def terminal_html(banner: str, events: list[tuple[float, str]]) -> str:
    data = json.dumps({"banner": banner, "events": events})
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:1280px;height:720px;background:{THEME["bg"]};overflow:hidden}}
.win{{position:absolute;left:150px;right:150px;top:96px;bottom:40px;background:#141311;
border:1px solid {THEME["border"]};border-radius:12px;overflow:hidden}}
.bar{{height:34px;background:{THEME["panel"]};display:flex;align-items:center;gap:8px;
padding:0 14px;color:{THEME["muted"]};font:13px Menlo,monospace}}
.dot{{width:12px;height:12px;border-radius:50%;background:{THEME["border"]}}}
.bar span{{margin-left:auto;margin-right:auto}}
pre{{margin:0;padding:16px 20px;color:{THEME["text"]};font:17px/1.5 Menlo,monospace;
white-space:pre-wrap;word-break:break-word}}
.cur{{background:{THEME["text"]};color:{THEME["bg"]}}}
</style></head><body><div class="win"><div class="bar"><i class="dot"></i><i class="dot"></i>
<i class="dot"></i><span>qwn chat</span></div><pre id="t"></pre></div>
<script>
const D = {data};
let lines = [""], i = 0, t0 = null;
function feed(s) {{
  for (const c of s) {{
    if (c === "\\n") lines.push("");
    else if (c === "\\r") lines[lines.length - 1] = "";
    else if (c === "\\b") lines[lines.length - 1] = lines[lines.length - 1].slice(0, -1);
    else lines[lines.length - 1] += c;
  }}
}}
function draw() {{
  const el = document.getElementById("t");
  el.textContent = lines.join("\\n");
  const cur = document.createElement("span");
  cur.className = "cur"; cur.textContent = " ";
  el.appendChild(cur);
}}
function tick() {{
  const t = (performance.now() - t0) / 1000;
  while (i < D.events.length && D.events[i][0] <= t) feed(D.events[i++][1]);
  draw();
  if (i < D.events.length) requestAnimationFrame(tick);
  else window.done = true;
}}
feed(D.banner); draw();
window.start = () => {{ t0 = performance.now(); requestAnimationFrame(tick); }};
</script></body></html>"""


# --- qwn chat in a pty -------------------------------------------------------------------------


class Pty:
    def __init__(self, env: dict) -> None:
        import pty

        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(
            ["uv", "run", "qwn", "chat"],
            cwd=ROOT,
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
        )
        os.close(slave)
        self.log: list[tuple[float, bytes]] = []
        self.t0 = time.monotonic()

    def text(self, start: int = 0, end: int | None = None) -> str:
        raw = b"".join(c for _, c in self.log[start:end])
        return ANSI.sub("", raw.decode("utf-8", "replace"))

    def read(self, timeout: float = 0.05) -> bool:
        r, _, _ = select.select([self.master], [], [], timeout)
        if not r:
            return True
        try:
            chunk = os.read(self.master, 65536)
        except OSError:
            return False
        if not chunk:
            return False
        self.log.append((time.monotonic() - self.t0, chunk))
        return True

    def until(self, done: Callable[[], bool], what: str, timeout: float = 600) -> None:
        deadline = time.monotonic() + timeout
        while not done():
            if time.monotonic() > deadline or self.proc.poll() is not None:
                raise RecordError(f"qwn chat: gave up waiting for {what}:\n{self.text()[-800:]}")
            self.read()

    def send(self, data: str) -> None:
        os.write(self.master, data.encode())

    def close(self) -> None:
        self.send("\x04")  # Ctrl-D: quit (nothing is saved, so nothing to confirm)
        deadline = time.monotonic() + 60
        while self.proc.poll() is None and time.monotonic() < deadline:
            if not self.read():
                break
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
        self.proc.wait(timeout=30)
        os.close(self.master)


def run_terminal(env: dict, q: dict, warm: dict, delay: float) -> dict:
    """Ask the warm-up off the record, then the shot's question; return the replay data."""
    term = Pty(env)
    term.until(lambda: PROMPT in term.text(), "the prompt")
    banner_end = len(term.log)
    term.send(warm["query"] + "\r")
    term.until(lambda: term.text(banner_end).count(PROMPT) >= 1, "the warm-up answer")
    shot_start = len(term.log)
    t_type = time.monotonic() - term.t0
    for ch in q["query"]:
        term.send(ch)
        end = time.monotonic() + delay
        while time.monotonic() < end:
            term.read(max(end - time.monotonic(), 0.001))
    t_sent = time.monotonic() - term.t0
    term.send("\r")
    want = q["answer_contains"][0]
    term.until(lambda: want in term.text(shot_start), "the answer")
    t_answer = next(  # the chunk that completes the answer's text
        term.log[k][0]
        for k in range(shot_start, len(term.log))
        if want in term.text(shot_start, k + 1)
    )
    term.until(lambda: term.text(shot_start).count(PROMPT) >= 1, "the next prompt")
    shot_end = next(  # the chunk that draws the next prompt: the replay ends there
        k + 1 for k in range(shot_start, len(term.log)) if PROMPT in term.text(shot_start, k + 1)
    )
    term.close()
    text = term.text(shot_start)
    missing = [s for s in q["answer_contains"] if s not in text]
    if missing:
        raise RecordError(f"terminal: the answer lacks {missing}: {text[:300]!r}")
    banner = "".join(s for _, s in clean_log(term.log[:banner_end]))
    events = [(round(t - t_type, 4), s) for t, s in clean_log(term.log[shot_start:shot_end])]
    return {
        "banner": banner,
        "events": events,
        "sent": t_sent - t_type,
        "answer": t_answer - t_type,
        "last": events[-1][0] if events else 0.0,
    }


# --- main --------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def preflight(index: Path, out: Path) -> None:
    from qwn.models import ModelsInUse, ProcessLock

    busy = subprocess.run(
        ["pgrep", "-f", r"qwn/ui/app\.py|bin/qwn (ui|chat)|qwn (ui|chat)$"],
        capture_output=True,
        text=True,
    ).stdout.split()
    if busy:
        raise RecordError(f"qwn ui or qwn chat is running (pid {', '.join(busy)}): close it first")
    lock = ProcessLock(index / "models.lock")
    try:
        lock.acquire()
    except ModelsInUse as e:
        raise RecordError(str(e)) from None
    lock.release()
    conv = index / "conversations"
    if conv.exists() and any(conv.iterdir()):
        raise RecordError(f"{conv} isn't empty: the sidebar would show saved conversations")
    if out.exists() and any(out.iterdir()):
        raise RecordError(f"{out} isn't empty")


def hard_index() -> Path:
    found = sorted(p for p in (ROOT / "index" / "eval-hard").glob("*") if p.is_dir())
    if len(found) != 1:
        raise RecordError(f"pass --index: index/eval-hard has {len(found)} indexes")
    return found[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--index", type=Path, default=None, help="default: the one in index/eval-hard")
    p.add_argument("--question-wav", type=Path, default=ROOT / "docs" / "video" / "question.wav")
    p.add_argument("--timeline", type=Path, default=ROOT / "docs" / "video" / "timeline.toml")
    p.add_argument("--only", help="record these shots only (comma-separated ids; for iterating)")
    args = p.parse_args()
    try:
        record(args)
    except (RecordError, timeline.TimelineError) as e:
        sys.exit(f"record: {e}")


def record(args) -> None:
    from playwright.sync_api import sync_playwright

    tl = timeline.load(args.timeline)
    index = (args.index or hard_index()).resolve()
    out = args.out.resolve()
    preflight(index, out)
    out.mkdir(parents=True, exist_ok=True)
    wav = args.question_wav.resolve()
    shutil.copyfile(wav, out / "question.wav")
    with wave.open(str(wav)) as w:
        wav_s = w.getnframes() / w.getframerate()
    hard, turns = queries("hard"), queries("turns")
    by_text = {q["query"]: q for q in hard.values()}
    only = set(args.only.split(",")) if args.only else None
    wanted = [s for s in tl.shots if only is None or s.id in only]
    home = str(Path.home())

    port = free_port()
    url = f"http://localhost:{port}"
    env = {
        **os.environ,
        "QWN_DATA_DIR": str(ROOT / "data" / "eval-hard"),
        "QWN_INDEX_DIR": str(index),
        "STREAMLIT_SERVER_PORT": str(port),
        "STREAMLIT_SERVER_HEADLESS": "true",
    }
    say = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)  # noqa: E731
    ui_kinds = {"chat", "open_page", "followup", "voice", "refusal", "library"}
    shots: dict[str, dict] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome",
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav}%noloop",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        ctx = browser.new_context(viewport=VIEW, device_scale_factor=SCALE, color_scheme="dark")
        ctx.add_init_script(HOME_SCRIPT % json.dumps(home))

        if any(s.kind in ui_kinds for s in wanted):
            ctx.grant_permissions(["microphone"], origin=url)
            log = (out / "ui.log").open("w")
            ui = subprocess.Popen(
                ["uv", "run", "qwn", "ui"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                for _ in range(180):
                    try:
                        urllib.request.urlopen(url, timeout=1)
                        break
                    except OSError:
                        time.sleep(1)
                page = ctx.new_page()
                page.goto(url)
                say("qwn ui up; warming up (typed, then spoken)")
                wait_ready(page)
                page.locator(CHAT_INPUT).fill(hard[WARMUP]["query"])
                page.keyboard.press("Enter")
                page.locator(MESSAGE).get_by_text("Searched in:").first.wait_for(timeout=600_000)
                new_conversation(page)
                spoken(None, page, wav_s)
                page.locator(MESSAGE).last.locator("audio").first.wait_for(
                    state="attached", timeout=600_000
                )
                rec = Recorder(page, out)
                for shot in tl.shots:
                    if shot.kind not in ui_kinds or shot not in wanted:
                        continue
                    say(f"shot {shot.id} ({shot.kind})")
                    q = (turns if shot.kind == "followup" else hard).get(shot.query or "")
                    if shot.kind in ("chat", "refusal"):
                        chat_shot(rec, page, shot, q, tl)
                    elif shot.kind == "open_page":
                        prev = tl.shots[tl.shots.index(shot) - 1]
                        open_page_shot(rec, page, shot, hard[prev.query or ""])
                    elif shot.kind == "followup":
                        assert q is not None
                        first = by_text[q["history"][0]["question"]]
                        followup_shot(rec, page, shot, q, first, tl)
                    elif shot.kind == "voice":
                        voice_shot(rec, page, shot, q, wav_s, url, out)
                    elif shot.kind == "library":
                        library_shot(rec, page, shot)
                shots.update(rec.shots)
                page.close()
            finally:
                os.killpg(ui.pid, signal.SIGTERM)
                ui.wait(timeout=60)
                log.close()
            say("qwn ui stopped")

        pages = out / "pages"
        pages.mkdir(exist_ok=True)
        page = ctx.new_page()

        def render(name: str, html: str, *, transparent: bool) -> None:
            src = pages / (Path(name).stem + ".html")
            src.write_text(html)
            page.goto(src.as_uri())
            page.evaluate("document.fonts.ready")
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out / name), omit_background=transparent)

        kinds = {s.id: s.kind for s in tl.shots}
        for name, (style, text) in timeline.overlays(tl).items():
            shot_id = Path(name).stem.split("-", 1)[1] if style != "label" else ""
            center = 640 if kinds.get(shot_id) in ("terminal", "card") else 790
            render(name, overlay_html(style, text, center), transparent=True)
        render("overlays/title.png", title_html(tl), transparent=True)
        say("overlays rendered")

        for shot in tl.shots:
            if shot not in wanted:
                continue
            if shot.kind == "card":
                render(f"pages/{shot.id}.png", card_html(shot), transparent=False)
                rec = Recorder(page, out, check_home=False)
                rec.start(shot.id)
                rec.events["end"] = shot.hold_s
                rec.shots[shot.id] = {"frames": rec.frames, "events": rec.events}
                shots.update(rec.shots)
            elif shot.kind == "terminal":
                say(f"shot {shot.id}: qwn chat in a pty")
                assert shot.query is not None
                term_env = {k: v for k, v in env.items() if not k.startswith("STREAMLIT_")}
                data = run_terminal(term_env, hard[shot.query], hard[WARMUP], tl.type_delay_s)
                src = pages / "terminal.html"
                src.write_text(terminal_html(data["banner"], data["events"]))
                page.goto(src.as_uri())
                rec = Recorder(page, out)
                rec.start(shot.id)
                rec.hold(shot.lead_s)
                page.evaluate("window.start()")
                base = rec.now()
                rec.until(lambda: page.evaluate("window.done === true"), "the replay", 900)
                rec.events["sent"] = round(base + data["sent"], 4)
                rec.events["answer"] = round(base + data["answer"], 4)
                rec.hold(shot.hold_s)
                rec.finish()
                shots.update(rec.shots)
        browser.close()

    events = {"shots": shots, "meta": {"index": index.name, "question_s": round(wav_s, 3)}}
    (out / "events.json").write_text(json.dumps(events, indent=1))
    if only is None:
        timeline.build_plan(tl, events)  # every event a cut needs is there
    say(f"recorded {len(shots)} shots to {out}")


if __name__ == "__main__":
    main()
