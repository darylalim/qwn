"""Real-model fixtures. Everything here is @slow: it loads ~13 GB of MLX models."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from samples import CAKE, PAGE

from qwn.config import Settings
from qwn.models import Registry


def pytest_collection_modifyitems(items):
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def registry():
    reg = Registry(Settings())
    reg.load_core()
    yield reg
    reg.process_lock.release()


def _text_image(path: Path, text: str) -> Path:
    img = Image.new("RGB", (1240, 1754), "white")  # A4 at 150 dpi
    font = ImageFont.load_default(size=36)
    ImageDraw.Draw(img).multiline_text(
        (100, 150), text.replace(". ", ".\n"), fill="black", font=font, spacing=16
    )
    img.save(path)
    return path


@pytest.fixture(scope="session")
def page_image(tmp_path_factory) -> Path:
    return _text_image(tmp_path_factory.mktemp("pages") / "page.png", PAGE)


@pytest.fixture(scope="session")
def cake_image(tmp_path_factory) -> Path:
    return _text_image(tmp_path_factory.mktemp("pages") / "cake.png", CAKE)
