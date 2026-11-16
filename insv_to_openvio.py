#!/usr/bin/env python3
"""Export a raw Insta360 dual-fisheye .insv as an OpenVIO dataset (EQUIRECTANGULAR camera model).

OpenVIO (MIT, github.com/93won/openvio) is a tightly coupled VIO with a two-lens 360 camera model that
estimates the camera-IMU time offset, extrinsics, inter-lens time offset and stitching error online.

Unlike stitch_dualfisheye.py (which levels every frame with the IMU), frames here are stitched in the fixed
camera frame OpenVIO expects: front lens (lens0) along +z, x right, y down; the back lens covers z <= 0.

  uv run insv_to_openvio.py VID.insv --calib ~/cap/fisheye/work -o ~/cap/openvio
  -> panocam/*.png, cam_timestamps.txt, imu_data.csv, x5.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import telemetry_parser

from insta360_dualfisheye_to_placeframe import LENSES, LensGeom, lens1_from_rig, operator_mask
from insta360_to_placeframe import pycolmap
from stitch_dualfisheye import fisheye_project
from vio_dualfisheye import calibrate_imu, detect_label_shift


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv")
    ap.add_argument("--calib", required=True, help="dual-fisheye converter work dir (sfm/ + meta.json + frames/)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--bottom-mask", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    (out / "panocam").mkdir(parents=True, exist_ok=True)
    work = Path(args.calib).expanduser()
    meta = json.loads((work / "meta.json").read_text())
    geoms = [LensGeom(**g) for g in meta["geoms"]]
    rec = pycolmap.Reconstruction(str(work / "sfm"))
    cams = {}
    for im in rec.images.values():
        cams.setdefault(im.name.split("/")[0], rec.cameras[im.camera_id])
    params = [np.asarray(cams[l].params, float) for l in LENSES]
    R10, _ = lens1_from_rig(rec)

    print("[imu] calibrating against the lens SfM")
    off_ms, lens0_from_imu, gscale = calibrate_imu(Path(args.insv), rec, 220.0)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=r_frame_rate,nb_frames", "-of", "json", args.insv],
                                      capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = map(int, probe["r_frame_rate"].split("/"))
    fps = num / den
    nframes = int(probe["nb_frames"]) if not args.max_frames else args.max_frames
    shift = detect_label_shift(args.insv, work / "frames", fps, meta["ts"])
    print(f"  IMU->label offset {off_ms:.1f} ms, label->video shift {shift * 1000:.0f} ms, gyro scale {gscale:.4f}")

    # IMU on the video clock: t_video = t_imu_raw - offset + shift
    imu = telemetry_parser.Parser(args.insv).normalized_imu()
    t = np.array([s["timestamp_ms"] for s in imu]) / 1000 - off_ms / 1000 + shift
    acc = np.array([s["accl"] for s in imu], float)
    gyr = np.radians(np.array([s["gyro"] for s in imu], float)) * gscale
    keep = np.concatenate([[True], np.diff(t) > 0]) & (t >= -0.5) & (t <= nframes / fps + 0.5)
    with open(out / "imu_data.csv", "w") as fh:
        fh.write("timestamp,ax,ay,az,gx,gy,gz\n")
        for ti, a, g in zip(t[keep], acc[keep], gyr[keep]):
            fh.write(f"{ti:.6f},{a[0]:.6f},{a[1]:.6f},{a[2]:.6f},{g[0]:.8f},{g[1]:.8f},{g[2]:.8f}\n")

    # Fixed stitch maps (OpenVIO ERP convention): u = W(0.5 + atan2(x,z)/2pi), v = H(0.5 + asin(y)/pi)
    W, H = args.width, args.width // 2
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 2 * np.pi
    lat = ((np.arange(H) + 0.5) / H - 0.5) * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    d0 = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
    d1 = d0 @ R10.T
    S = 1920
    maps, weights = [], []
    seam = math.radians(3)
    for d, p, g in ((d0, params[0], geoms[0]), (d1, params[1], geoms[1])):
        u, v = fisheye_project(d, p)
        mx, my = (u - 0.5).astype(np.float32), (v - 0.5).astype(np.float32)
        keep_px = np.where(np.hypot(*np.meshgrid(np.arange(S) - g.cx, np.arange(S) - g.cy)) < g.radius * 0.97,
                           operator_mask((S, S), g, args.bottom_mask), 0).astype(np.uint8)
        valid = cv2.remap(keep_px, mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
        th = np.arctan2(np.hypot(d[..., 0], d[..., 1]), d[..., 2])
        maps.append((mx, my))
        weights.append((np.clip((math.pi / 2 + seam - th) / (2 * seam), 0, 1) * valid).astype(np.float32))
    wsum = weights[0] + weights[1]
    masked = wsum < 1e-3
    rects = mask_rectangles(masked)
    print(f"  operator mask -> {len(rects)} rectangles covering {masked.mean():.0%} of the image")

    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", args.insv, "-filter_complex",
                             "[0:v:0]format=gray[a];[0:v:1]format=gray[b];[a][b]hstack", "-frames:v", str(nframes),
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    ts = []
    for k in range(nframes):
        buf = proc.stdout.read(S * 2 * S)
        if len(buf) < S * 2 * S:
            break
        img = np.frombuffer(buf, np.uint8).reshape(S, 2 * S)
        layers = [cv2.remap(img[:, L * S:(L + 1) * S], *maps[L], cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
                  for L in range(2)]
        pano = (layers[0] * weights[0] + layers[1] * weights[1]) / np.maximum(wsum, 1e-6)
        pano[masked] = 0
        cv2.imwrite(str(out / "panocam" / f"{k:06d}.png"), pano.clip(0, 255).astype(np.uint8),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        ts.append(k / fps)
        if k % 500 == 0:
            print(f"  stitched {k}/{nframes}")
    proc.kill()
    (out / "cam_timestamps.txt").write_text("".join(f"{x:.6f}\n" for x in ts))

    R_imu_cam = lens0_from_imu.T  # p_imu = R p_camera
    rows = [list(R_imu_cam[i]) + [0.0] for i in range(3)] + [[0, 0, 0, 1]]
    yaml = f"""%YAML:1.0
---
# Insta360 X5 (raw .insv, stitched in the fixed camera frame), generated by insv_to_openvio.py.
camera:
  model: EQUIRECTANGULAR
  width: {W}
  height: {H}
  params: [0.0]
  polar_exclusion_ratio: 0.12
  border_margin: 20
  mask_rectangles: {json.dumps([[round(v, 3) for v in r] for r in rects])}
imu:
  # Starting point: PanoAir's consumer 360-camera IMU. Replace with Allan-variance values from a static recording.
  gyro_noise: 8.28865836731e-05
  accel_noise: 0.000587880617676
  gyro_walk: 7.84852086053e-05
  accel_walk: 0.000184199881504
  gravity_magnitude: 9.805
rig:
  imu_from_camera:
""" + "".join(f"    - [{', '.join(f'{v:.10f}' for v in r)}]\n" for r in rows) + """  clock_offset: 0.0   # IMU already written on the video clock (calibrated against the lens SfM)
"""
    (out / "x5.yaml").write_text(yaml)
    print(json.dumps({"frames": len(ts), "fps": fps, "imu_samples": int(keep.sum()), "out": str(out)}, indent=2))


def mask_rectangles(masked: np.ndarray) -> list[list[float]]:
    """Cover the masked (operator) area with a few column-band rectangles [x0, x1, y0, y1] as image fractions."""
    H, W = masked.shape
    bands = 16
    rects = []
    for b in range(bands):
        x0, x1 = b * W // bands, (b + 1) * W // bands
        rows = np.where(masked[:, x0:x1].mean(axis=1) > 0.3)[0]
        if len(rows) == 0:
            continue
        # only the bottom-attached / contiguous blob, ignore tiny specks
        if len(rows) < H * 0.02:
            continue
        rects.append([x0 / W, x1 / W, rows.min() / H, (rows.max() + 1) / H])
    return rects


if __name__ == "__main__":
    main()
