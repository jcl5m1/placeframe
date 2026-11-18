#!/usr/bin/env python3
"""Render a synthetic equirectangular walk through a textured 8x3x6 m room (camera at 1.6 m).

  uv run tests/render_synthetic_360.py OUT_DIR   -> OUT_DIR/walk.mp4 + OUT_DIR/gt.json
Used to sanity-check insta360_to_placeframe.py (pose accuracy, levelling, metric scale).
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, D = 8.0, 3.0, 6.0
PX = 0.004
EW, EH = 2048, 1024


def tex(w, h, seed):
    r = np.random.default_rng(seed)
    w, h = int(w / PX), int(h / PX)
    img = np.zeros((h, w, 3), np.float32)
    for s in [2, 8, 32, 128]:
        n = r.random((max(2, h // s), max(2, w // s), 3)).astype(np.float32)
        img += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) / 4
    img = (img * 255).clip(0, 255).astype(np.uint8)
    for _ in range(int(w * h / 4000)):
        c = tuple(int(x) for x in r.integers(0, 255, 3))
        p = (int(r.integers(0, w)), int(r.integers(0, h)))
        if r.random() < 0.5:
            cv2.circle(img, p, int(r.integers(5, 40)), c, -1)
        else:
            cv2.rectangle(img, p, (p[0] + int(r.integers(5, 60)), p[1] + int(r.integers(5, 60))), c, -1)
    return img


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic360")
    out.mkdir(parents=True, exist_ok=True)
    T = {"floor": tex(W, D, 1), "ceil": tex(W, D, 2), "x0": tex(D, H, 3), "x1": tex(D, H, 4),
         "z0": tex(W, H, 5), "z1": tex(W, H, 6)}
    lon = (np.arange(EW) + 0.5) / EW * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(EH) + 0.5) / EH * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    drig = np.stack([np.cos(LAT) * np.sin(LON), -np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
    faces = [(1, 0.0, "floor", 0, 2), (1, H, "ceil", 0, 2), (0, 0.0, "x0", 2, 1), (0, W, "x1", 2, 1),
             (2, 0.0, "z0", 0, 1), (2, D, "z1", 0, 1)]

    def render(pos, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        Rwr = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.diag([1.0, -1.0, 1.0])
        d = drig @ Rwr.T
        img = np.zeros((EH, EW, 3), np.uint8)
        best = np.full((EH, EW), np.inf)
        for ax, val, name, ua, va in faces:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (val - pos[ax]) / d[..., ax]
            m = (t > 0) & (t < best)
            if not m.any():
                continue
            p = pos + d * t[..., None]
            u, v = p[..., ua], p[..., va]
            if va == 1:
                v = H - v
            smp = cv2.remap(T[name], (u / PX).astype(np.float32), (v / PX).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img[m] = smp[m]
            best[m] = t[m]
        return img

    fps, n = 5, 150
    vw = cv2.VideoWriter(str(out / "walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (EW, EH))
    gt = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pos = np.array([4 + 2.5 * math.cos(a), 1.6, 3 + 1.8 * math.sin(a)])
        vw.write(render(pos, 0.3 * math.sin(3 * a) + a * 0.5))
        gt.append(pos.tolist())
    vw.release()
    (out / "gt.json").write_text(json.dumps({"fps": fps, "positions": gt}))
    print(f"wrote {out / 'walk.mp4'}")


if __name__ == "__main__":
    main()
