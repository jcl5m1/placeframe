#!/usr/bin/env python3
"""Open-loop visual-inertial odometry for a raw Insta360 dual-fisheye .insv — one pose per video frame.

Loosely coupled, no loop closure, no global optimisation (drift is left visible on purpose):
  * rotation    — camera gyro integrated at ~1 kHz (Mahony filter; accelerometer only corrects tilt)
  * translation — KLT feature tracks on both fisheye lenses -> unit bearings (self-calibrated lens
                  models from insta360_dualfisheye_to_placeframe.py); with rotation known, each frame
                  is a translation-only fix against triangulated points; keyframes add new points.
                  If vision is lost the pose coasts at constant velocity.
  * scale       — metric scale from the accelerometer, re-fit in sliding windows (monocular scale drifts)

  uv run vio_dualfisheye.py VID.insv --calib ~/pf_x5/out/work -o ~/pf_x5/vio
Writes vio.csv: frame,t_s,t_label_s,x,y,z,qx,qy,qz,qw,status (t_label_s = converter/map timestamp) (world: metres, z up; status tracked|reinit|coast)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import telemetry_parser
from scipy.interpolate import make_smoothing_spline
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from insta360_dualfisheye_to_placeframe import LENSES, LensGeom, lens1_from_rig
from insta360_to_placeframe import pycolmap
from stitch_dualfisheye import LENS0_FROM_IMU

DOWN = 2  # decode at half resolution (960x960 per lens) for speed


# --------------------------------------------------------------------------- geometry


def unproject(uv: np.ndarray, p: np.ndarray) -> np.ndarray:
    """OPENCV_FISHEYE pixels (COLMAP convention, full-res) -> unit bearings in lens frame. Valid past 90°."""
    fx, fy, cx, cy, k1, k2, k3, k4 = p
    x, y = (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy
    thd = np.hypot(x, y)
    th = thd.copy()
    for _ in range(8):  # Newton on thd = th(1 + k1 th^2 + ...)
        t2 = th * th
        f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        th = th - f / df
    s = np.where(thd > 1e-9, np.sin(th) / np.maximum(thd, 1e-9), 1.0)
    return np.stack([x * s, y * s, np.cos(th)], 1)


def skew_rows(b: np.ndarray) -> np.ndarray:
    z = np.zeros(len(b))
    return np.stack([np.stack([z, -b[:, 2], b[:, 1]], 1), np.stack([b[:, 2], z, -b[:, 0]], 1),
                     np.stack([-b[:, 1], b[:, 0], z], 1)], 1)  # (N,3,3)


def solve_center(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Camera centre c with known world bearings b to points X: [b]x c = [b]x X (least squares)."""
    S = skew_rows(b)
    A = S.reshape(-1, 3)
    y = np.einsum("nij,nj->ni", S, X).reshape(-1)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def ang_err(b: np.ndarray, X: np.ndarray, c: np.ndarray) -> np.ndarray:
    v = X - c
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return np.degrees(np.arccos(np.clip(np.sum(v * b, 1), -1, 1)))


def ransac_center(b, X, thr_deg, rng, iters=60):
    best = None
    n = len(b)
    for _ in range(iters):
        i = rng.choice(n, 2, replace=False)
        c = solve_center(b[i], X[i])
        inl = ang_err(b, X, c) < thr_deg
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 6:
        return None, None
    c = solve_center(b[best], X[best])
    inl = ang_err(b, X, c) < thr_deg
    return solve_center(b[inl], X[inl]), inl


def ransac_direction(b1, b2, rng, iters=150, thr_deg=0.6):
    """Translation direction between two views with world-frame bearings (rotation known).

    Hypotheses from pairs of epipolar-plane normals (b1 x b2) of well-separated rays; residual is the
    angle of b2 from the epipolar plane spanned by t and b1, which stays well-conditioned for distant
    (low-parallax) points."""
    par = np.degrees(np.arccos(np.clip(np.sum(b1 * b2, 1), -1, 1)))
    cand = np.where(par > 0.8)[0]
    if len(cand) < 8:
        return None, None
    thr = math.sin(math.radians(thr_deg))

    def residual(t):
        n_p = np.cross(t, b1)
        n_p /= np.linalg.norm(n_p, axis=1, keepdims=True) + 1e-12
        return np.abs(np.sum(n_p * b2, 1))

    best, bt = None, None
    for _ in range(iters):
        i, j = rng.choice(cand, 2, replace=False)
        n1, n2 = np.cross(b1[i], b2[i]), np.cross(b1[j], b2[j])
        t = np.cross(n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2))
        if np.linalg.norm(t) < 1e-3:
            continue
        t /= np.linalg.norm(t)
        inl = residual(t) < thr
        if best is None or inl.sum() > best.sum():
            best, bt = inl, t
    if best is None or best.sum() < 8:
        return None, None
    n_ = np.cross(b1[best], b2[best])
    w = np.linalg.norm(n_, axis=1)
    _, _, vt = np.linalg.svd(n_ / np.maximum(w, 1e-9)[:, None] * np.minimum(w / np.median(w), 3)[:, None])
    t = vt[2] if vt[2] @ bt >= 0 else -vt[2]
    return t, residual(t) < thr


def triangulate(cA, bA, cB, bB):
    """Midpoint triangulation, vectorised. Returns points, depthA, depthB, parallax(deg)."""
    w0 = cA - cB
    a = np.sum(bA * bA, 1)
    bb = np.sum(bA * bB, 1)
    c = np.sum(bB * bB, 1)
    d = bA @ w0
    e = bB @ w0
    den = a * c - bb * bb
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    sA = (bb * e - c * d) / den
    sB = (a * e - bb * d) / den
    P = 0.5 * ((cA + sA[:, None] * bA) + (cB + sB[:, None] * bB))
    par = np.degrees(np.arccos(np.clip(np.sum(bA * bB, 1), -1, 1)))
    return P, sA, sB, par


# --------------------------------------------------------------------------- IMU


def imu_attitude(insv: Path, offset_ms: float, gyro_scale: float = 1.0, kp: float = 0.05):
    imu = telemetry_parser.Parser(str(insv)).normalized_imu()
    t = (np.array([s["timestamp_ms"] for s in imu]) - offset_ms) / 1000
    acc = np.array([s["accl"] for s in imu], dtype=np.float64)
    gyr = np.radians(np.array([s["gyro"] for s in imu], dtype=np.float64))
    R = Rotation.align_vectors([[0, 0, 1.0]], [acc[:200].mean(0)])[0]
    q = np.zeros((len(t), 4))
    for i in range(len(t)):
        dt = 0.001 if i == 0 else min(0.01, max(0.0, t[i] - t[i - 1]))
        n = np.linalg.norm(acc[i])
        err = np.cross(acc[i] / n, R.inv().apply([0, 0, 1.0])) if abs(n - 9.81) < 2 else np.zeros(3)
        R = R * Rotation.from_rotvec((gyr[i] * gyro_scale + kp * err) * dt)
        q[i] = R.as_quat()
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], Rotation.from_quat(q[keep]), acc[keep]


def calibrate_imu(insv: Path, rec, offset0_ms: float):
    """Refine IMU->lens0 mount rotation, time offset and gyro scale against SfM relative rotations."""
    from scipy.optimize import least_squares

    from insta360_dualfisheye_to_placeframe import frame_poses

    poses = frame_poses(rec)
    ts = sorted(poses)
    pairs = [(a, b) for a, b in zip(ts[:-1], ts[1:]) if b - a <= 600]
    imu = telemetry_parser.Parser(str(insv)).normalized_imu()
    T = np.array([s["timestamp_ms"] for s in imu]) / 1000
    G = np.radians(np.array([s["gyro"] for s in imu]))
    dt = np.clip(np.diff(T, prepend=T[0]), 0, 0.01)
    q = np.zeros((len(T), 4))
    R = Rotation.identity()
    for i in range(len(T)):
        R = R * Rotation.from_rotvec(G[i] * dt[i])
        q[i] = R.as_quat()
    keep = np.concatenate([[True], np.diff(T) > 0])
    sl = Slerp(T[keep], Rotation.from_quat(q[keep]))

    def resid(x):
        off = x[0] / 1000
        C = Rotation.from_rotvec(x[1:4]) * Rotation.from_matrix(LENS0_FROM_IMU)
        out = []
        for a, b in pairs:
            ra, rb = (sl(np.clip(v / 1000 + off, T[0], T[-1])) for v in (a, b))
            rel = C * Rotation.from_rotvec((ra.inv() * rb).as_rotvec() * (1 + x[4])) * C.inv()
            out.append((Rotation.from_matrix(poses[a][0].T @ poses[b][0]).inv() * rel).as_rotvec())
        return np.concatenate(out)

    x0 = np.array([offset0_ms, 0, 0, 0, 0.0])
    r = least_squares(resid, x0, x_scale=[50, 0.05, 0.05, 0.05, 0.01], loss="soft_l1", f_scale=0.02)
    e0 = np.degrees(np.linalg.norm(resid(x0).reshape(-1, 3), axis=1))
    e1 = np.degrees(np.linalg.norm(resid(r.x).reshape(-1, 3), axis=1))
    lens0_from_imu = (Rotation.from_rotvec(r.x[1:4]) * Rotation.from_matrix(LENS0_FROM_IMU)).as_matrix()
    print(f"  rotation error vs SfM: median {np.median(e0):.2f}° -> {np.median(e1):.2f}°  "
          f"(offset {r.x[0]:.1f} ms, mount {np.degrees(np.linalg.norm(r.x[1:4])):.2f}°, gyro scale {1 + r.x[4]:.4f})")
    return r.x[0], lens0_from_imu, 1 + r.x[4]


def detect_label_shift(insv: str, frames_dir: Path, fps: float, ts_labels: list[int]) -> float:
    """Seconds to add to converter frame labels to get true video time (ffmpeg's fps filter shifts them)."""
    S = 480
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", insv, "-filter_complex",
                          f"[0:v:0]scale={S}:{S},format=gray", "-frames:v", "90", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    F = np.frombuffer(raw, np.uint8).reshape(-1, S, S).astype(np.float32)
    shifts = []
    for lab in ts_labels[:3]:
        ref = cv2.resize(cv2.imread(str(frames_dir / "lens0" / f"{lab}.jpg"), cv2.IMREAD_GRAYSCALE), (S, S))
        j = int(np.argmin([np.mean(np.abs(f - ref)) for f in F]))
        shifts.append(j / fps - lab / 1000)
    return float(np.median(shifts))


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv")
    ap.add_argument("--calib", required=True, help="work dir with sfm/ + meta.json from the dual-fisheye converter")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--imu-offset-ms", type=float, default=220.0)
    ap.add_argument("--max-theta", type=float, default=88.0)
    ap.add_argument("--bottom-mask", type=float, default=0.35)
    ap.add_argument("--corners", type=int, default=350, help="features per lens")
    ap.add_argument("--kf-parallax", type=float, default=2.5, help="median parallax (deg) to add a keyframe")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--scale-window", type=float, default=10.0, help="seconds per IMU scale window")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.calib).expanduser()
    meta = json.loads((work / "meta.json").read_text())
    geoms = [LensGeom(**g) for g in meta["geoms"]]
    rec = pycolmap.Reconstruction(str(work / "sfm"))
    cams = {}
    for im in rec.images.values():
        cams.setdefault(im.name.split("/")[0], rec.cameras[im.camera_id])
    params = [np.asarray(cams[l].params, float) for l in LENSES]
    R10, _ = lens1_from_rig(rec)  # lens1_from_lens0
    lens_to_rig = [np.eye(3), R10.T]

    t0 = time.time()
    print("[imu] calibrating IMU against lens SfM, then integrating gyro")
    off, lens0_from_imu, gscale = calibrate_imu(Path(args.insv), rec, args.imu_offset_ms)
    t_imu, R_wi, acc = imu_attitude(Path(args.insv), off, gscale)
    slerp = Slerp(t_imu, R_wi)
    imu_from_rig = lens0_from_imu.T

    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=r_frame_rate,nb_frames", "-of", "json", args.insv],
                                      capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = map(int, probe["r_frame_rate"].split("/"))
    fps = num / den
    nframes = int(probe["nb_frames"])
    if args.max_frames:
        nframes = min(nframes, args.max_frames)
    shift = detect_label_shift(args.insv, work / "frames", fps, meta["ts"])
    print(f"  converter frame labels lag true video time by {shift * 1000:.0f} ms; IMU looked up on label time")
    S = 1920 // DOWN
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", args.insv, "-filter_complex",
         f"[0:v:0]scale={S}:{S}[a];[0:v:1]scale={S}:{S}[b];[a][b]hstack,format=gray",
         "-frames:v", str(nframes), "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

    # detection masks (half-res)
    masks = []
    for g, p in zip(geoms, params):
        yy, xx = np.mgrid[0:S, 0:S] * DOWN
        r = np.hypot(xx - g.cx, yy - g.cy)
        f = 0.5 * (p[0] + p[1])
        m = (r < f * math.radians(args.max_theta)) & (yy < g.cy + args.bottom_mask * g.radius)
        masks.append((m * 255).astype(np.uint8))

    rng = np.random.default_rng(0)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # track state per lens: pts (N,2 half-res), ids (N,)
    tracks = [{"pts": np.zeros((0, 2), np.float32), "ids": np.zeros(0, int)} for _ in LENSES]
    birth: dict[int, tuple[np.ndarray, Rotation]] = {}  # id -> (rig bearing at birth, camera rotation at birth)
    static_mask = [np.zeros((S, S), np.uint8) for _ in LENSES]  # >0: recently rig-static (operator) area
    n_static = 0
    next_id = 0
    points3d: dict[int, np.ndarray] = {}
    kf = None  # {"c":, "bear": {id: world bearing}}
    centers = np.zeros((nframes, 3))
    quats = np.zeros((nframes, 4))
    status = ["coast"] * nframes
    prev = [None, None]
    vel = np.zeros(3)
    have_scale = False
    stats = {"tracked": 0, "reinit": 0, "coast": 0}

    for k in range(nframes):
        buf = proc.stdout.read(S * 2 * S)
        if len(buf) < S * 2 * S:
            nframes = k
            break
        img = np.frombuffer(buf, np.uint8).reshape(S, 2 * S)
        lens_imgs = [img[:, :S], img[:, S:]]
        t = k / fps
        Rw = (slerp(np.clip(t - shift, t_imu[0], t_imu[-1])) * Rotation.from_matrix(imu_from_rig))
        quats[k] = Rw.as_quat()
        Rwm = Rw.as_matrix()

        # --- track + replenish
        ids_all, bear_all = [], []
        for L in range(2):
            tr = tracks[L]
            if prev[L] is not None and len(tr["pts"]):
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev[L], lens_imgs[L], tr["pts"], None, **lk)
                p0, st0, _ = cv2.calcOpticalFlowPyrLK(lens_imgs[L], prev[L], p1, None, **lk)
                good = (st1[:, 0] == 1) & (st0[:, 0] == 1) & (np.linalg.norm(p0 - tr["pts"], axis=1) < 1.0)
                pi = np.clip(p1.astype(int), 0, S - 1)
                good &= masks[L][pi[:, 1], pi[:, 0]] > 0
                tr["pts"], tr["ids"] = p1[good], tr["ids"][good]
            static_mask[L] = np.maximum(static_mask[L].astype(int) - 1, 0).astype(np.uint8)
            if len(tr["pts"]) < args.corners * 0.6:
                m = masks[L].copy()
                m[static_mask[L] > 0] = 0
                for x, y in tr["pts"].astype(int):
                    cv2.circle(m, (int(x), int(y)), 12, 0, -1)
                new = cv2.goodFeaturesToTrack(lens_imgs[L], args.corners - len(tr["pts"]), 0.01, 12, mask=m)
                if new is not None:
                    new = new.reshape(-1, 2).astype(np.float32)
                    tr["pts"] = np.vstack([tr["pts"], new])
                    tr["ids"] = np.concatenate([tr["ids"], np.arange(next_id, next_id + len(new))])
                    next_id += len(new)
            prev[L] = lens_imgs[L]
            if len(tr["pts"]):
                d = unproject(tr["pts"].astype(float) * DOWN + 0.5, params[L]) @ lens_to_rig[L].T
                # Reject operator / stick features: fixed in the rig frame while the camera rotates.
                drop = np.zeros(len(d), bool)
                for j, i in enumerate(tr["ids"].tolist()):
                    if i not in birth:
                        birth[i] = (d[j], Rw)
                        continue
                    b0, R0 = birth[i]
                    cam_rot = np.degrees((R0.inv() * Rw).magnitude())
                    if cam_rot > 6:
                        moved = np.degrees(np.arccos(np.clip(d[j] @ b0, -1, 1)))
                        if moved < 0.25 * cam_rot:
                            drop[j] = True
                if drop.any():
                    n_static += int(drop.sum())
                    for x, y in tr["pts"][drop].astype(int):
                        cv2.circle(static_mask[L], (int(x), int(y)), 25, 90, -1)  # ~3 s exclusion
                    for i in tr["ids"][drop].tolist():
                        points3d.pop(int(i), None)
                    keepm = ~drop
                    tr["pts"], tr["ids"], d = tr["pts"][keepm], tr["ids"][keepm], d[keepm]
                ids_all.append(tr["ids"])
                bear_all.append(d @ Rwm.T)
        ids = np.concatenate(ids_all) if ids_all else np.zeros(0, int)
        bw = np.vstack(bear_all) if bear_all else np.zeros((0, 3))

        # --- pose
        c_pred = centers[k - 1] + vel if k else np.zeros(3)
        c = None
        have = np.array([i in points3d for i in ids], bool)
        if have.sum() >= 12:
            X = np.array([points3d[i] for i in ids[have]])
            c_est, inl = ransac_center(bw[have], X, 1.0, rng)
            if c_est is not None and inl.sum() >= 12 and np.linalg.norm(c_est - c_pred) < 0.5 + 5 * np.linalg.norm(vel):
                c = c_est
                status[k] = "tracked"
                bad = ids[have][~inl]
                for i in bad:
                    points3d.pop(int(i), None)

        if kf is None:
            kf = {"k": k, "c": np.zeros(3), "bear": dict(zip(ids.tolist(), bw))}
            centers[k] = 0
            continue

        common = [i for i in ids.tolist() if i in kf["bear"]]
        if common:
            idx = {i: j for j, i in enumerate(ids.tolist())}
            b1 = np.array([kf["bear"][i] for i in common])
            b2 = np.array([bw[idx[i]] for i in common])
            par = np.median(np.degrees(np.arccos(np.clip(np.sum(b1 * b2, 1), -1, 1))))
        else:
            par = 0.0

        if c is None and common and par > args.kf_parallax and len(common) >= 20:
            # (re)initialise translation from two-view geometry with known rotation
            tdir, inl = ransac_direction(b1, b2, rng)
            if tdir is not None:
                step = np.linalg.norm(vel) * (k - kf["k"]) if have_scale else 1.0
                for sgn in (1, -1):
                    cB = kf["c"] + sgn * tdir * max(step, 1e-3)
                    _, sA, sB, _ = triangulate(kf["c"], b1[inl], cB, b2[inl])
                    if np.mean((sA > 0) & (sB > 0)) > 0.7:
                        c = cB
                        status[k] = "reinit"
                        have_scale = True
                        break
        if c is None:
            c = c_pred  # coast
        centers[k] = c
        stats[status[k]] += 1
        if k > 1 and status[k] != "coast":
            vel = 0.8 * vel + 0.2 * (centers[k] - centers[k - 1])

        # --- keyframe: triangulate new points
        if status[k] != "coast" and common and par > args.kf_parallax:
            P, sA, sB, pr = triangulate(kf["c"], b1, c, b2)
            ok = (sA > 0) & (sB > 0) & (pr > 1.5)
            for i, p, o in zip(common, P, ok):
                if o and i not in points3d:
                    points3d[i] = p
            kf = {"k": k, "c": c.copy(), "bear": dict(zip(ids.tolist(), bw))}
            if len(points3d) > 20000:
                live = set(ids.tolist())
                points3d = {i: p for i, p in points3d.items() if i in live}
        elif not common or len(common) < 30:
            kf = {"k": k, "c": c.copy(), "bear": dict(zip(ids.tolist(), bw))}

        if k % 300 == 0:
            print(f"  frame {k}/{nframes} t={t:6.1f}s  tracks={len(ids)}  3d={len(points3d)}  "
                  f"tracked={stats['tracked']} reinit={stats['reinit']} coast={stats['coast']}  operator-rejected={n_static}  "
                  f"{(time.time() - t0) / 60:.1f} min")
    proc.kill()
    centers, quats, status = centers[:nframes], quats[:nframes], status[:nframes]
    ts = np.arange(nframes) / fps

    # --- metric scale from the accelerometer, re-estimated in sliding windows. Monocular VO loses scale
    # over time (it shrinks badly in low-texture stretches); the accelerometer makes it observable, so
    # each window's displacement is rescaled to match measured acceleration, then re-integrated.
    print("[scale] sliding-window fit of VIO accelerations to the accelerometer")
    a_world = R_wi.apply(acc) - np.array([0, 0, 9.81])
    tl = ts - shift
    b, a = butter(2, 1.5 / (fps / 2))
    acc_v = np.stack([make_smoothing_spline(tl, centers[:, j], lam=1e-4).derivative(2)(tl) for j in range(3)], 1)
    acc_i = np.stack([np.interp(tl, t_imu, a_world[:, j]) for j in range(3)], 1)
    xv, yi = filtfilt(b, a, acc_v, axis=0), filtfilt(b, a, acc_i, axis=0)
    win, hop = int(args.scale_window * fps), int(2 * fps)
    wc, ws, wr = [], [], []
    for c in range(win // 2, nframes - win // 2, hop):
        x, y = xv[c - win // 2:c + win // 2], yi[c - win // 2:c + win // 2]
        sc = float(np.sum(x * y) / np.sum(x * x))
        r2 = 1 - np.sum((y - sc * x) ** 2) / np.sum((y - y.mean(0)) ** 2)
        wc.append(tl[c]); ws.append(sc); wr.append(r2)
    wc, ws, wr = np.array(wc), np.array(ws), np.array(wr)
    ok = (wr > 0.2) & (ws > 0)
    if ok.sum() < 2:
        raise SystemExit("IMU scale fit failed: not enough motion with usable accelerometer signal")
    s_t = np.interp(tl, wc[ok], ws[ok])
    steps = np.diff(centers, axis=0) * s_t[1:, None]
    centers = np.vstack([np.zeros((1, 3)), np.cumsum(steps, axis=0)])
    s = float(np.median(ws[ok]))
    r2 = float(np.median(wr[ok]))
    print(f"  {ok.sum()}/{len(ws)} windows usable, median fit R^2 {r2:.2f}, "
          f"local scale range {np.percentile(ws[ok], 10):.3g}-{np.percentile(ws[ok], 90):.3g} m/unit")

    with open(out / "vio.csv", "w") as fh:
        fh.write("frame,t_s,t_label_s,x,y,z,qx,qy,qz,qw,status\n")
        for k in range(nframes):
            q = quats[k]
            fh.write(f"{k},{ts[k]:.4f},{ts[k] - shift:.4f},{centers[k,0]:.4f},{centers[k,1]:.4f},{centers[k,2]:.4f},"
                     f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f},{status[k]}\n")
    summ = {"frames": nframes, "fps": fps, **{f"frames_{k}": v for k, v in stats.items()},
            "scale_median_m_per_unit": s, "scale_fit_median_r2": r2,
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))),
            "elapsed_min": round((time.time() - t0) / 60, 1)}
    (out / "vio.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
