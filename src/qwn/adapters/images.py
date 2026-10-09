"""Pillow: normalise images and write page renders as WebP q85 (PLAN.md → Data model → Images)."""

import math
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

WEBP_QUALITY = 85


class ImageRejected(ValueError):
    """The image can't be indexed; the message is the reason shown to the user."""


def fit(size: tuple[int, int], max_pixels: int) -> tuple[int, int]:
    """Largest size with the same aspect ratio and at most `max_pixels` pixels."""
    w, h = size
    if w * h <= max_pixels:
        return w, h
    scale = math.sqrt(max_pixels / (w * h))
    return max(1, int(w * scale)), max(1, int(h * scale))


def save_webp(image: Image.Image, dst: Path, *, max_pixels: int | None = None) -> None:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    if max_pixels is not None and image.width * image.height > max_pixels:
        image = image.resize(fit(image.size, max_pixels), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, "WEBP", quality=WEBP_QUALITY)


def normalize(src: Path, dst: Path, *, max_pixels: int, max_image_pixels: int) -> None:
    """Re-encode `src` as WebP at `dst`, upright, aspect kept, capped at `max_pixels`.

    Pillow only warns between 1x and 2x its pixel limit, so the warning is turned into an error
    here; anything over `max_image_pixels` is rejected before it's decoded.
    """
    if src.stat().st_size == 0:
        raise ImageRejected("empty file")
    Image.MAX_IMAGE_PIXELS = max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(src) as img:
                img.load()
                upright = ImageOps.exif_transpose(img)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ImageRejected(f"image too large (over {max_image_pixels:,} pixels)") from None
    except UnidentifiedImageError:
        raise ImageRejected("unreadable image") from None
    except OSError as e:  # truncated or corrupt data
        raise ImageRejected(f"unreadable image ({e})") from None
    save_webp(upright, dst, max_pixels=max_pixels)
