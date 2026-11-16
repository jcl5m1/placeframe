#!/usr/bin/env python3
"""Estimate metric scale (metres per SfM unit) by matching SfM accelerations to the Insta360's IMU.

The accelerometer measures true m/s²; the SfM trajectory's second derivative is in arbitrary units.
Both are band-limited identically (Butterworth) and related by a scaled rotation (Umeyama fit).

  uv run estimate_imu_scale.py VID.insv OUT_DIR/work/sfm [--imu-offset-ms 220]
Then:  uv run insta360_to_placeframe.py FRAMES --reuse-frames --reuse-sfm --keep-work --scale <printed>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import telemetry_parser
from scipy.interpolate import make_smoothing_spline
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from insta360_to_placeframe import frame_poses, pycolmap


def imu_world_linear_accel(insv: Path, offset_ms: float):
    imu = telemetry_parser.Parser(str(insv)).normalized_imu()
    t = (np.array([s["timestamp_ms"] for s in imu]) - offset_ms) / 1000
    acc = np.array([s["accl"] for s in imu], dtype=np.float64)
    gyr = np.radians(np.array([s["gyro"] for s in imu], dtype=np.float64))
    # Mahony-style attitude: integrate gyro, pull tilt toward the accelerometer (heading left free).
    R = Rotation.align_vectors([[0, 0, 1.0]], [acc[:200].mean(0)])[0]
    quats = np.zeros((len(t), 4))
    for i in range(len(t)):
        dt = 0.001 if i == 0 else min(0.01, max(0.0, t[i] - t[i - 1]))
        n = np.linalg.norm(acc[i])
        err = np.cross(acc[i] / n, R.inv().apply([0, 0, 1.0])) if abs(n - 9.81) < 2 else np.zeros(3)
        R = R * Rotation.from_rotvec((gyr[i] + 0.5 * err) * dt)
        quats[i] = R.as_quat()
    return t, Rotation.from_quat(quats).apply(acc) - np.array([0, 0, 9.81])


def lowpass(x, fs, fc):
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv")
    ap.add_argument("sfm_dir", help="work/sfm from insta360_to_placeframe.py --keep-work (largest model is used)")
    ap.add_argument("--imu-offset-ms", type=float, default=None, help="IMU minus video time (default: stitch.json or 220)")
    ap.add_argument("--fc", type=float, default=0.7, help="low-pass cutoff in Hz (keep below frame-rate Nyquist)")
    ap.add_argument("--lam", type=float, default=0.0005, help="SfM trajectory spline smoothing")
    args = ap.parse_args()

    sfm_dir = Path(args.sfm_dir).expanduser()
    models = sorted(sfm_dir.glob("*/images.bin")) or sorted(sfm_dir.glob("images.bin"))
    rec = max((pycolmap.Reconstruction(str(p.parent)) for p in models), key=lambda r: r.num_reg_frames())
    offset = args.imu_offset_ms
    if offset is None:
        stitch = next(sfm_dir.parents[2].glob("**/stitch.json"), None) if len(sfm_dir.parents) > 2 else None
        offset = json.loads(stitch.read_text())["imu_offset_ms"] if stitch else 220.0

    poses = frame_poses(rec)
    ts = np.array(sorted(poses))
    t = ts / 1000
    C = np.array([poses[x][1] for x in ts])
    t_imu, a_imu = imu_world_linear_accel(Path(args.insv), offset)

    fs = 20
    xs, ys = [], []
    for seg in np.split(np.arange(len(t)), np.where(np.diff(t) > 1.01)[0] + 1):
        if len(seg) < 12:
            continue
        tt = np.arange(t[seg[0]] + 1.0, t[seg[-1]] - 1.0, 1 / fs)
        if len(tt) < 20:
            continue
        a_sfm = np.stack([make_smoothing_spline(t[seg], C[seg, k], lam=args.lam).derivative(2)(tt)
                          for k in range(3)], 1)
        xs.append(lowpass(a_sfm, fs, args.fc))
        grid = np.arange(tt[0] - 2, tt[-1] + 2, 0.005)
        a_grid = lowpass(np.stack([np.interp(grid, t_imu, a_imu[:, k]) for k in range(3)], 1), 200, args.fc)
        ys.append(np.stack([np.interp(tt, grid, a_grid[:, k]) for k in range(3)], 1))
    if not xs:
        raise SystemExit("Not enough continuous registered trajectory to estimate scale.")

    X, Y = np.vstack(xs), np.vstack(ys)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Yc.T @ Xc)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (Xc**2).sum())
    r2 = 1 - ((Yc - s * Xc @ R.T) ** 2).sum() / (Yc**2).sum()
    seg_s = [float(((x @ R.T) * y).sum() / ((x @ R.T) ** 2).sum()) for x, y in zip(xs, ys)]
    print(f"IMU offset {offset:.0f} ms, {len(xs)} segments, {len(X)} samples")
    print(f"per-segment scales: {np.round(seg_s, 3).tolist()}")
    print(f"fit R^2 = {r2:.3f}  (below ~0.6 means don't trust it)")
    print(f"scale = {s:.4f} m per SfM unit  (segment spread ±{np.std(seg_s):.3f})")


if __name__ == "__main__":
    main()
