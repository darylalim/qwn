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


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory, corpus_index):
    _docs, index_dir = corpus_index
    home = tmp_path_factory.mktemp("ui-home")
    (home / "qwn.example.toml").write_text("")
    shutil.copytree(index_dir, home / "index")
    port = _free_port()
    env = {
        **os.environ,
        "QWN_HOME": str(home),
        "QWN_LOG_DIR": str(home / "logs"),
        "HF_HUB_OFFLINE": "1",
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
