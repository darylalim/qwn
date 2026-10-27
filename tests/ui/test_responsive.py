"""Responsive layout: a real Streamlit server (fakes injected) and Chrome via Playwright.

Slow: too heavy for CI; run locally with the merge checklist (PLAN.md → Responsive layout).
Screenshots go to tests/ui/screenshots/ (gitignored) for the visual review.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.slow

ROOT = Path(__file__).parents[2]
FAKE_APP = Path(__file__).with_name("fake_app.py")
SHOTS = Path(__file__).with_name("screenshots")
WIDTHS = (2560, 1728, 1512, 1024, 760, 390)
HEIGHT = 1000
SCHEMES = ("light", "dark")
PAGES = {"chat": "", "library": "library", "system": "system"}
READY = {  # wait for a page-specific element, not a fixed delay
    "chat": '[data-testid="stChatInput"]',
    "library": '[data-testid="stDataFrame"]',
    "system": '[data-testid="stTable"]',
}
CHAT_MAX_PX = 736
QUESTION = "How much did revenue grow?"
FOLLOW_UP = "And in 2025?"
WARM_SECONDS = 12


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _serve(home: Path, index_dir: Path, **env: str):
    """Start the fake app on a free port with its own QWN_HOME; yield its URL, then stop it."""
    (home / "qwn.example.toml").write_text("")
    shutil.copytree(index_dir, home / "index")
    port = _free_port()
    env = {
        **os.environ,
        "QWN_HOME": str(home),
        "QWN_LOG_DIR": str(home / "logs"),
        "HF_HUB_OFFLINE": "1",
        **env,
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(FAKE_APP),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        cwd=ROOT,  # .streamlit/config.toml (theme, localhost only) is read from here
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1) as r:
                if r.status == 200:
                    break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        out = proc.stdout.read().decode() if proc.stdout else ""
        pytest.fail(f"streamlit didn't start:\n{out}")
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def server(tmp_path_factory, corpus_index):
    yield from _serve(tmp_path_factory.mktemp("ui-home"), corpus_index[1])


@pytest.fixture
def warming_server(tmp_path_factory, corpus_index):
    """A fresh server whose warm load takes WARM_SECONDS: the models load once per process."""
    home = tmp_path_factory.mktemp("ui-home-warm")
    yield from _serve(
        home, corpus_index[1], QWN_FAKE_WARM_SECONDS=str(WARM_SECONDS), QWN_FAKE_ANSWER_SECONDS="10"
    )


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")  # the installed Chrome: no browser download
        yield b
        b.close()


def _overflow(page) -> tuple[int, int]:
    return page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("name", PAGES)
def test_layout(server, browser, name, width, scheme):
    SHOTS.mkdir(exist_ok=True)
    context = browser.new_context(viewport={"width": width, "height": HEIGHT}, color_scheme=scheme)
    page = context.new_page()
    try:
        page.goto(f"{server}/{PAGES[name]}")
        page.wait_for_selector(READY[name], timeout=20_000)
        if name == "chat":
            box = page.locator('[data-testid="stChatInput"] textarea')
            box.fill(QUESTION)
            box.press("Enter")
            page.get_by_role("button", name="Open page").first.wait_for(timeout=20_000)
            box.fill(FOLLOW_UP)  # a rewritten follow-up: "Searched for" and "New conversation"
            box.press("Enter")
            page.get_by_text("Searched for:").wait_for(timeout=20_000)
            page.get_by_role("button", name="New conversation").wait_for(timeout=5_000)
            page.locator("button").filter(has_text="Export").wait_for(timeout=5_000)
            page.get_by_role("button", name="Save").click()  # the button row, then its caption
            page.get_by_text("Saved ", exact=False).first.wait_for(timeout=10_000)
            if width > 760:  # the sidebar's list (collapsed on narrow screens)
                page.get_by_text("Saved conversations").first.wait_for(timeout=10_000)
        page.wait_for_timeout(300)  # let the layout settle after the last element arrives
        page.screenshot(path=SHOTS / f"{name}-{width}-{scheme}.png", full_page=True)

        scroll, inner = _overflow(page)
        assert scroll <= inner, f"horizontal overflow: {scroll} > {inner}"
        sidebar = page.locator('[data-testid="stSidebar"]')
        if width <= 760:
            assert sidebar.get_attribute("aria-expanded") == "false"
        if name == "chat":
            main = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
            assert main is not None and main["width"] <= CHAT_MAX_PX
            page.get_by_role("button", name="Open page").first.click()
            dialog = page.locator('[role="dialog"]')
            dialog.locator("img").first.wait_for(timeout=10_000)
            page.wait_for_timeout(300)
            img = dialog.locator("img").first.bounding_box()
            assert img is not None and img["x"] >= 0 and img["x"] + img["width"] <= width
            page.screenshot(path=SHOTS / f"chat-dialog-{width}-{scheme}.png")
    finally:
        context.close()


def test_a_saved_conversation_reopens_from_the_sidebar(server, browser):
    """Save, then open it in a new session: AppTest can't show that the sidebar's clicks land."""
    context = browser.new_context(viewport={"width": 1512, "height": HEIGHT})
    page = context.new_page()
    try:
        page.goto(server)
        box = page.locator('[data-testid="stChatInput"] textarea')
        box.wait_for(timeout=20_000)
        box.fill(QUESTION)
        box.press("Enter")
        page.get_by_role("button", name="Save").click()
        page.get_by_text("Saved ", exact=False).first.wait_for(timeout=10_000)

        page = context.new_page()  # a new session: nothing on screen
        page.goto(server)
        page.locator('[data-testid="stChatInput"]').wait_for(timeout=20_000)
        sidebar = page.locator('[data-testid="stSidebar"]')
        sidebar.locator('[data-testid="stSelectbox"]').click()
        page.get_by_role("option").first.click()
        # by its text: the selectbox's own arrow is also a button named "Open"
        sidebar.locator("button").filter(has_text="Open").click()
        page.get_by_text("Searched in:").first.wait_for(timeout=10_000)
        assert page.get_by_role("button", name="Open page").count() >= 1
    finally:
        context.close()


def test_export_downloads_and_rename_lands_in_the_sidebar(server, browser, tmp_path):
    """The download and the Rename dialog in a real browser: AppTest can't click either."""
    context = browser.new_context(viewport={"width": 1512, "height": HEIGHT}, accept_downloads=True)
    page = context.new_page()
    try:
        page.goto(server)
        box = page.locator('[data-testid="stChatInput"] textarea')
        box.wait_for(timeout=20_000)
        box.fill(QUESTION)
        box.press("Enter")
        page.get_by_role("button", name="Open page").first.wait_for(timeout=20_000)
        with page.expect_download(timeout=10_000) as info:
            page.locator("button").filter(has_text="Export").click()
        download = info.value
        assert download.suggested_filename.startswith("qwn-conversation-")
        target = tmp_path / download.suggested_filename
        download.save_as(target)
        text = target.read_text()
        assert text.startswith(f"# {QUESTION}\n") and "· not saved" in text
        assert "[report.pdf p.1]" in text

        page.get_by_role("button", name="Save").click()
        page.get_by_text("Saved ", exact=False).first.wait_for(timeout=10_000)
        sidebar = page.locator('[data-testid="stSidebar"]')
        sidebar.locator('[data-testid="stSelectbox"]').click()
        page.get_by_role("option").filter(has_text=QUESTION).first.click()
        sidebar.locator("button").filter(has_text="Rename").click()
        dialog = page.locator('[role="dialog"]')
        field = dialog.get_by_label("Title")
        field.fill("Renamed in the browser")
        field.press("Enter")  # commit the text input before the click
        dialog.locator("button").filter(has_text="Rename").click()
        dialog.wait_for(state="hidden", timeout=10_000)
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "chat-renamed-1512.png", full_page=True)
        selected = sidebar.locator('[data-testid="stSelectbox"] input').input_value()
        assert selected.startswith("Renamed in the browser · "), selected
        sidebar.locator('[data-testid="stSelectbox"]').click()
        page.get_by_role("option").first.wait_for(timeout=5_000)
        options = page.get_by_role("option").all_inner_texts()
        assert any(o.startswith("Renamed in the browser · ") for o in options), options
    finally:
        context.close()


def test_a_question_during_the_warm_load_leaves_the_chat_box_usable(warming_server, browser):
    """Ask as soon as the page loads, while the models are still loading. The sidebar's 1 s
    status timer must stop once they're ready: left running, it kept the chat box disabled
    after the answer with real models (the fakes are too quick to show that part)."""
    context = browser.new_context(viewport={"width": 1512, "height": HEIGHT})
    page = context.new_page()
    sent = []  # the browser's messages: a timer tick sends one a second
    page.on("websocket", lambda ws: ws.on("framesent", lambda f: sent.append(f)))
    try:
        page.goto(warming_server)
        box = page.locator('[data-testid="stChatInput"] textarea')
        box.wait_for(timeout=20_000)
        box.fill(QUESTION)
        box.press("Enter")
        page.get_by_role("button", name="Open page").first.wait_for(
            timeout=(WARM_SECONDS + 20) * 1000
        )
        page.get_by_text("Guard on").wait_for(timeout=10_000)  # the warm load has finished
        page.wait_for_timeout(2000)  # the last tick's rerun lands
        before = len(sent)
        page.wait_for_timeout(3000)
        assert len(sent) == before, f"the status timer still ticks: {len(sent) - before} in 3 s"
        expect(box).to_be_enabled(timeout=10_000)
        box.fill(FOLLOW_UP)
        box.press("Enter")
        page.get_by_text("Searched for:").wait_for(timeout=20_000)
    finally:
        context.close()
