"""Top-down map slices for the phone app: grayscale histogram images of a map's points, one per height.

The phone draws the map near its own height. Instead of the point cloud (~5 MB for a house), it downloads only the
one or two slice images around its height (tens of KB each) and blends them.

  * A slice is a top-down image of the map's x/z plane (OpenCV map frame, the frame localizations are returned in):
    image column = +x, image row = +z, CELL metres per pixel.
  * Slices are every STEP metres along the map's vertical axis (y); slice k at y = y0 + k * STEP holds the points
    within ±HALF_HEIGHT of it.
  * Pixel value: the number of points in the cell, on a log scale (0 = none, 255 = the 99th percentile of the
    non-empty cells or more), so walls and sparse floors both show.

Built once per reconstruction and cached under CACHE/<reconstruction id>/: meta.json and <k>.png.
"""
from __future__ import annotations

import json
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

CACHE = Path.home() / "pf_data" / "map_slices"
CELL = 0.05  # metres per pixel
STEP = 0.25  # metres between slices
HALF_HEIGHT = 1.0  # points within this of a slice's height are in it
MARGIN = 1.0  # metres around the map's extent
MAX_PIXELS = 4000  # per side (200 m at 5 cm)


def cached(reconstruction_id: str) -> dict | None:
    meta = CACHE / reconstruction_id / "meta.json"
    return json.loads(meta.read_text()) if meta.exists() else None


def build(reconstruction_id: str, points_payload: bytes) -> dict:
    """Slices from Placeframe's /reconstructions/{id}/points payload (uint32 count, float32 x y z, uint8 r g b)."""
    (n,) = struct.unpack_from("<I", points_payload, 0)
    xyz = np.frombuffer(points_payload, dtype="<f4", count=3 * n, offset=4).reshape(n, 3).astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    # The extent from robust percentiles: a few stray points far away must not make the images huge.
    lo, hi = np.percentile(xyz, 0.5, axis=0) - MARGIN, np.percentile(xyz, 99.5, axis=0) + MARGIN
    width = int(min(MAX_PIXELS, np.ceil((hi[0] - lo[0]) / CELL)))
    height = int(min(MAX_PIXELS, np.ceil((hi[2] - lo[2]) / CELL)))
    y0 = np.floor(lo[1] / STEP) * STEP
    levels = int(np.ceil((hi[1] - y0) / STEP)) + 1

    # Per-band histograms (one band per STEP), then each slice sums the bands within ±HALF_HEIGHT.
    u = np.floor((xyz[:, 0] - lo[0]) / CELL).astype(np.int64)
    v = np.floor((xyz[:, 2] - lo[2]) / CELL).astype(np.int64)
    b = np.floor((xyz[:, 1] - y0) / STEP + 0.5).astype(np.int64)  # nearest slice height
    keep = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (b >= 0) & (b < levels)
    bands = np.zeros((levels, height, width), dtype=np.int32)
    np.add.at(bands, (b[keep], v[keep], u[keep]), 1)
    reach = int(round(HALF_HEIGHT / STEP))
    cumulative = np.concatenate([np.zeros((1, height, width), np.int32), np.cumsum(bands, axis=0, dtype=np.int32)])
    slices = [cumulative[min(levels, k + reach + 1)] - cumulative[max(0, k - reach)] for k in range(levels)]

    nonzero = np.concatenate([s[s > 0] for s in slices]) if any((s > 0).any() for s in slices) else np.array([1])
    top = max(1.0, float(np.percentile(nonzero, 99)))
    out = CACHE / reconstruction_id
    out.mkdir(parents=True, exist_ok=True)
    sizes = []
    for k, s in enumerate(slices):
        img = np.clip(np.log1p(s) / np.log1p(top), 0, 1)
        buf = BytesIO()
        Image.fromarray((img * 255).astype(np.uint8), mode="L").save(buf, format="PNG", optimize=True)
        (out / f"{k}.png").write_bytes(buf.getvalue())
        sizes.append(len(buf.getvalue()))
    meta = {"reconstruction_id": reconstruction_id, "points": int(n), "cell": CELL, "x0": float(lo[0]),
            "z0": float(lo[2]), "width": width, "height": height, "y0": float(y0), "step": STEP,
            "half_height": HALF_HEIGHT, "levels": levels, "bytes": sizes,
            "frame": "OpenCV map frame; image column = +x, image row = +z; slice k at y = y0 + k * step"}
    tmp = out / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(out / "meta.json")
    return meta
