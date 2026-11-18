#!/usr/bin/env python3
"""Stitch a raw Insta360 dual-fisheye capture into gravity-levelled equirectangular frames.

Uses the lens calibration that insta360_dualfisheye_to_placeframe.py self-calibrated (work/sfm) and
the camera's own IMU (gyro + accelerometer, complementary filter) for per-frame gravity, so the output
is level and upright even when the camera was carried tilted or sideways. The stick/hand/operator
region is blacked out. Feed the result to insta360_to_placeframe.py:

  uv run stitch_dualfisheye.py VID.insv --work ~/pf_x5/out/work -o ~/pf_x5/equirect
  uv run insta360_to_placeframe.py ~/pf_x5/equirect --image-interval-ms 500 --room-height 2.44
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import telemetry_parser
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from insta360_dualfisheye_to_placeframe import LENSES, LensGeom, Ry, frame_poses, lens1_from_rig, operator_mask
from insta360_to_placeframe import pycolmap

# IMU axes -> lens0 (OpenCV) axes for the X5, found by matching accelerometer gravity to SfM rotations.
LENS0_FROM_IMU = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])


def fisheye_project(d: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OPENCV_FISHEYE projection valid past 90° (uses atan2). Returns COLMAP pixel coords (centre +0.5)."""
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    r = np.hypot(d[..., 0], d[..., 1])
    th = np.arctan2(r, d[..., 2])
    th2 = th * th
    thd = th * (1 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
    scale = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.0)
    return fx * d[..., 0] * scale + cx, fy * d[..., 1] * scale + cy


def imu_gravity(insv: Path, offset_ms: float, tau_s: float, gyro_sign: float):
    """Complementary filter. Returns (t_ms, up_imu[N,3]) — the unit 'up' vector in IMU axes over time."""
    imu = telemetry_parser.Parser(str(insv)).normalized_imu()
    t = np.array([s["timestamp_ms"] for s in imu]) - offset_ms
    acc = np.array([s["accl"] for s in imu], dtype=np.float64)
    gyr = np.radians(np.array([s["gyro"] for s in imu], dtype=np.float64)) * gyro_sign
    up = np.zeros_like(acc)
    u = acc[:200].mean(0)
    u /= np.linalg.norm(u)
    for i in range(len(t)):
        dt = 0.001 if i == 0 else max(0.0, min(0.01, (t[i] - t[i - 1]) / 1000))
        u = u - np.cross(gyr[i], u) * dt  # vector fixed in world, seen from a rotating body
        a = acc[i] / (np.linalg.norm(acc[i]) + 1e-9)
        w = dt / tau_s if abs(np.linalg.norm(acc[i]) - 9.81) < 2.0 else 0.0  # trust accel only when ~1 g
        u = (1 - w) * u + w * a
        u /= np.linalg.norm(u)
        up[i] = u
    return t, up


def down_in_lens0(t_imu, up_imu, ts_ms) -> dict[int, np.ndarray]:
    out = {}
    for ts in ts_ms:
        i = int(np.clip(np.searchsorted(t_imu, ts), 0, len(t_imu) - 1))
        d = -(LENS0_FROM_IMU @ up_imu[i])
        out[ts] = d / np.linalg.norm(d)
    return out


def level_rotation(down: np.ndarray) -> np.ndarray:
    """R_lens0_from_level: level frame y = down, z = lens0's forward projected onto the horizontal."""
    y = down
    fwd = np.array([0.0, 0.0, 1.0])
    if abs(fwd @ y) > 0.9:
        fwd = np.array([1.0, 0.0, 0.0])
    z = fwd - (fwd @ y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="raw .insv (for IMU)")
    ap.add_argument("--work", required=True, help="work dir from insta360_dualfisheye_to_placeframe.py")
    ap.add_argument("-o", "--out", required=True, help="output folder for equirect frames")
    ap.add_argument("--width", type=int, default=3072)
    ap.add_argument("--imu-offset-ms", type=float, default=None, help="IMU time minus video time (auto)")
    ap.add_argument("--tau", type=float, default=1.5, help="complementary filter time constant (s)")
    ap.add_argument("--bottom-mask", type=float, default=0.35)
    ap.add_argument("--seam-deg", type=float, default=3.0, help="blend width at the lens seam")
    args = ap.parse_args()

    work = Path(args.work).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((work / "meta.json").read_text())
    ts = meta["ts"]
    geoms = [LensGeom(**g) for g in meta["geoms"]]
    rec = pycolmap.Reconstruction(str(work / "sfm"))
    cams = {}
    for image in rec.images.values():
        cams.setdefault(image.name.split("/")[0], rec.cameras[image.camera_id])
    params = [np.asarray(cams[lens].params, dtype=np.float64) for lens in LENSES]
    R10, _ = lens1_from_rig(rec)
    poses = frame_poses(rec)

    # Calibrate IMU time offset + gyro sign against SfM: world gravity should be constant.
    print("[imu] fusing gyro + accelerometer")
    reg = sorted(poses)

    def spread(t_imu, up):
        g = np.array([poses[t][0] @ down_in_lens0(t_imu, up, [t])[t] for t in reg])
        return 1 - np.linalg.norm(g.mean(0)), g.mean(0) / np.linalg.norm(g.mean(0))

    best = None
    for sign in (1.0, -1.0):
        off = args.imu_offset_ms
        if off is None:
            t_raw, up_raw = imu_gravity(Path(args.input), 0.0, args.tau, sign)
            res = minimize_scalar(lambda o: spread(t_raw - o, up_raw)[0], bounds=(-500, 800), method="bounded",
                                  options={"xatol": 5})
            off = res.x
        t_imu, up = imu_gravity(Path(args.input), off, args.tau, sign)
        s, g = spread(t_imu, up)
        print(f"  gyro sign {sign:+.0f}: offset {off:.0f} ms, gravity spread {s:.4f} "
              f"(~{math.degrees(math.sqrt(2 * s)):.1f}° rms)")
        if best is None or s < best[0]:
            best = (s, sign, off, t_imu, up)
    _, sign, off, t_imu, up = best
    downs = down_in_lens0(t_imu, up, ts)

    # Equirect ray grid (level frame, OpenCV axes; image centre = horizontal heading of lens0).
    W, H = args.width, args.width // 2
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    d_level = np.stack([np.cos(LAT) * np.sin(LON), -np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1).astype(np.float64)

    sample = cv2.imread(str(work / "frames" / "lens0" / f"{ts[0]}.jpg"))
    h, w = sample.shape[:2]
    keeps = []
    for g in geoms:
        yy, xx = np.mgrid[0:h, 0:w]
        circle = np.hypot(xx - g.cx, yy - g.cy) < g.radius * 0.97
        keeps.append(np.where(circle, operator_mask((h, w), g, args.bottom_mask), 0).astype(np.uint8))

    seam = math.radians(args.seam_deg)
    print(f"[stitch] {len(ts)} frames -> {W}x{H} level equirect in {out}")
    for n, t in enumerate(ts):
        R = level_rotation(downs[t])  # lens0_from_level
        d0 = d_level @ R.T
        d1 = d0 @ R10.T
        imgs = [cv2.imread(str(work / "frames" / lens / f"{t}.jpg")) for lens in LENSES]
        layers, weights = [], []
        for d, p, img, keep in ((d0, params[0], imgs[0], keeps[0]), (d1, params[1], imgs[1], keeps[1])):
            u, v = fisheye_project(d, p)
            mx, my = (u - 0.5).astype(np.float32), (v - 0.5).astype(np.float32)
            layers.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0).astype(np.float32))
            valid = cv2.remap(keep, mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
            theta = np.arctan2(np.hypot(d[..., 0], d[..., 1]), d[..., 2])
            wgt = np.clip((math.pi / 2 + seam - theta) / (2 * seam), 0, 1) * valid
            weights.append(wgt.astype(np.float32))
        wsum = weights[0] + weights[1]
        pano = (layers[0] * weights[0][..., None] + layers[1] * weights[1][..., None]) / np.maximum(wsum, 1e-6)[..., None]
        pano[wsum < 1e-3] = 0
        cv2.imwrite(str(out / f"{t:08d}.jpg"), pano.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
        if n % 50 == 0:
            print(f"  {n}/{len(ts)}")
    (out / "stitch.json").write_text(json.dumps({"imu_offset_ms": off, "gyro_sign": sign, "frames": len(ts),
                                                 "interval_ms": ts[1] - ts[0] if len(ts) > 1 else 500}))
    print("done")


if __name__ == "__main__":
    main()
