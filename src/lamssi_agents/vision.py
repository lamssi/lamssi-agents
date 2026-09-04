"""Encodes images (numpy array, PIL image, file path, bytes, or data: URL) into base64 data: URLs for vision-capable LLM calls."""

from __future__ import annotations

import base64
import io
from typing import Any, List

# Rough per-image token cost for history-budget estimation; downscaled images run
# a few hundred to ~1k tokens, so we estimate high to avoid under-counting.
IMAGE_TOKEN_EST = 1_000

_DEFAULT_MAX_PX = 1024
_DEFAULT_QUALITY = 80


def to_image_urls(image: Any, **kw) -> List[str]:
    """Normalise an ``image=`` argument to a list of ``data:`` URLs.

    Accepts a single image (any form :func:`encode_image` takes) or a list
    of them. ``None`` becomes ``[]``.
    """
    if image is None:
        return []
    if isinstance(image, (list, tuple)):
        return [encode_image(im, **kw) for im in image]
    return [encode_image(image, **kw)]


def encode_image(
    image: Any,
    *,
    max_px: int = _DEFAULT_MAX_PX,
    quality: int = _DEFAULT_QUALITY,
) -> str:
    """Return a ``data:image/...;base64,...`` URL for *image*, downscaled to *max_px*.

    Args:
        image: A ``data:``/``http(s)://`` URL (returned as-is), a file path, raw
            encoded image bytes, a numpy array (uint8 or float, min-max
            stretched), or a PIL ``Image``.
    """
    if isinstance(image, str) and image.startswith(("data:", "http://", "https://")):
        return image

    img = _to_pil(image)
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))

    # PNG keeps an alpha channel; JPEG (smaller) for everything else.
    if img.mode in ("RGBA", "LA", "P"):
        fmt, mime = "PNG", "image/png"
        out = img
    else:
        fmt, mime = "JPEG", "image/jpeg"
        out = img.convert("RGB")

    buf = io.BytesIO()
    save_kw = {"quality": quality} if fmt == "JPEG" else {}
    out.save(buf, format=fmt, **save_kw)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _to_pil(image: Any):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            'sending an image needs Pillow: pip install "lamssi-agents[vision]"'
        ) from exc

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):                       # file path
        return Image.open(image)
    if isinstance(image, (bytes, bytearray)):        # encoded image bytes
        return Image.open(io.BytesIO(bytes(image)))
    return Image.fromarray(_as_uint8_array(image))   # numpy array


def _as_uint8_array(arr: Any):
    import numpy as np

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype("float64")
        if a.size:
            lo, hi = float(a.min()), float(a.max())
        else:
            lo, hi = 0.0, 1.0
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
        a = a.clip(0, 255).astype("uint8")
    # Drop a trailing singleton channel: HxWx1 becomes HxW (PIL wants HxW for 'L').
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return a
