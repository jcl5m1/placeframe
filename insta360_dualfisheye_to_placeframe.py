#!/usr/bin/env python3
"""Convert a raw Insta360 dual-fisheye .insv (X3/X4/X5...) into a Placeframe capture-session tar.

No Insta360 Studio export needed. The lenses are self-calibrated from the footage:

  1. Extract both lens streams at --fps (ffmpeg).
  2. Mask: outside the usable fisheye cone, and the bottom of each lens (selfie stick / hand / operator).
  3. Rig-aware COLMAP SfM on the raw fisheye images (OPENCV_FISHEYE per lens, lens1 = lens0 rotated
     180° about y), refining focal / centre / distortion and the lens-to-lens extrinsic.
  4. Level the world with the floor plane, set metric scale from --camera-height or --room-height.
  5. Re-project each lens into 5 pinhole views (operator blacked out) and write manifest.json +
     rig0/frames.csv + rig0/cameraK/<ms>.jpg as capture.tar.

  uv run insta360_dualfisheye_to_placeframe.py data/.../VID_xxx.insv --camera-height 1.4
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from insta360_to_placeframe import _cuda_device_count, pycolmap, vocab_tree_path  # noqa: F401  (pycolmap after CUDA preload)

RIG_ID = "rig0"
LENSES = ("lens0", "lens1")


def Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# --------------------------------------------------------------------------- 1. frames


def extract_frames(video: Path, fps: float, frames_dir: Path, max_seconds: float | None = None) -> list[int]:
    staging = frames_dir / "_ffmpeg"
    shutil.rmtree(staging, ignore_errors=True)  # leftovers of an interrupted run would be renamed as this run's frames
    for lens in LENSES:
        (frames_dir / lens).mkdir(parents=True, exist_ok=True)
        (staging / lens).mkdir(parents=True, exist_ok=True)
    print(f"[1/5] extracting both lenses at {fps} fps")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *(["-t", str(max_seconds)] if max_seconds else []), "-i", str(video),
         "-filter_complex", f"[0:v:0]fps={fps}[a];[0:v:1]fps={fps}[b]",
         "-map", "[a]", "-q:v", "2", str(staging / "lens0" / "%06d.jpg"),
         "-map", "[b]", "-q:v", "2", str(staging / "lens1" / "%06d.jpg")],
        check=True,
    )
    # Pair frames by index, name them by millisecond timestamp; unpaired trailing frames are dropped.
    n = min(len(list((staging / lens).glob("*.jpg"))) for lens in LENSES)
    ts = []
    for i in range(1, n + 1):
        t = round((i - 1) * 1000 / fps)
        for lens in LENSES:
            (staging / lens / f"{i:06d}.jpg").rename(frames_dir / lens / f"{t}.jpg")
        ts.append(t)
    shutil.rmtree(staging)
    print(f"  {n} frame pairs")
    return ts


# --------------------------------------------------------------------------- 2. masks


@dataclass
class LensGeom:
    cx: float
    cy: float
    radius: float  # image-circle radius in px


def fit_circle(frames_dir: Path, lens: str, ts: list[int]) -> LensGeom:
    acc = None
    for t in ts[:: max(1, len(ts) // 40)]:
        g = cv2.imread(str(frames_dir / lens / f"{t}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        acc = g if acc is None else acc + g
    mean = acc / max(1, len(ts[:: max(1, len(ts) // 40)]))
    m = mean > 12
    ys, xs = np.nonzero(m)
    return LensGeom(float(xs.mean()), float(ys.mean()), float(math.sqrt(m.sum() / math.pi)))


def operator_mask(shape, geom: LensGeom, bottom_frac: float) -> np.ndarray:
    """255 = keep. Removes the bottom cap of the fisheye where the stick / hand / operator sit."""
    h, w = shape
    yy = np.arange(h)[:, None]
    keep = np.broadcast_to(yy < geom.cy + bottom_frac * geom.radius, (h, w))
    return (keep * 255).astype(np.uint8)


def write_sfm_masks(frames_dir, masks_dir, ts, geoms, f_init, max_theta_deg, bottom_frac):
    for lens, geom in zip(LENSES, geoms):
        h, w = cv2.imread(str(frames_dir / lens / f"{ts[0]}.jpg"), cv2.IMREAD_GRAYSCALE).shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(xx - geom.cx, yy - geom.cy)
        cone = r < f_init * math.radians(max_theta_deg)
        mask = np.where(cone, operator_mask((h, w), geom, bottom_frac), 0).astype(np.uint8)
        (masks_dir / lens).mkdir(parents=True, exist_ok=True)
        for t in ts:
            cv2.imwrite(str(masks_dir / lens / f"{t}.jpg.png"), mask)


# --------------------------------------------------------------------------- 3. SfM


def run_sfm(args, frames_dir: Path, masks_dir: Path, work: Path, geoms, f_init) -> pycolmap.Reconstruction:
    db_path = work / "database.db"
    db_path.unlink(missing_ok=True)
    sfm_dir = work / "sfm"
    shutil.rmtree(sfm_dir, ignore_errors=True)
    sfm_dir.mkdir(parents=True)

    use_gpu = pycolmap.has_cuda and not args.cpu and _cuda_device_count() > 0
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    print(f"[3/5] fisheye SfM self-calibration ({'GPU' if use_gpu else 'CPU'})")

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = args.max_features
    extraction.use_gpu = use_gpu
    extraction.max_image_size = 2000
    for lens, geom in zip(LENSES, geoms):
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "OPENCV_FISHEYE"
        reader.camera_params = f"{f_init},{f_init},{geom.cx + 0.5},{geom.cy + 0.5},0,0,0,0"
        reader.mask_path = str(masks_dir)
        names = sorted(f"{lens}/{p.name}" for p in (frames_dir / lens).glob("*.jpg"))
        pycolmap.extract_features(
            str(db_path), str(frames_dir), image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="OPENCV_FISHEYE", reader_options=reader, extraction_options=extraction, device=device,
        )

    cam0 = pycolmap.RigConfigCamera()
    cam0.image_prefix = "lens0/"
    cam0.ref_sensor = True
    cam1 = pycolmap.RigConfigCamera()
    cam1.image_prefix = "lens1/"
    cam1.ref_sensor = False
    cam1.cam_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(Ry(math.pi)), np.zeros(3))
    rig = pycolmap.RigConfig()
    rig.cameras = [cam0, cam1]
    db = pycolmap.Database.open(str(db_path))
    pycolmap.apply_rig_config([rig], db)
    db.close()

    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = use_gpu
    matching.skip_image_pairs_in_same_frame = True
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = args.overlap
    pairing.quadratic_overlap = True
    pairing.expand_rig_images = True
    pairing.loop_detection = args.loop_closure
    if args.loop_closure:
        pairing.vocab_tree_path = vocab_tree_path()
    print("  matching")
    pycolmap.match_sequential(str(db_path), matching_options=matching, pairing_options=pairing, device=device)

    print("  incremental mapping (refining lens intrinsics + lens-to-lens extrinsic)")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True
    opts.ba_refine_principal_point = True
    opts.ba_refine_extra_params = True
    opts.ba_refine_sensor_from_rig = True
    recs = pycolmap.incremental_mapping(str(db_path), str(frames_dir), str(sfm_dir), options=opts)
    if not recs:
        sys.exit("SfM failed: no reconstruction.")
    best = max(recs.values(), key=lambda r: r.num_reg_frames())
    if len(recs) > 1:
        sizes = sorted((r.num_reg_frames() for r in recs.values()), reverse=True)
        print(f"  ! {len(recs)} disconnected models (frames: {sizes}); keeping the largest")
    best.write(str(sfm_dir))
    return best


# --------------------------------------------------------------------------- 4. level + scale


def frame_poses(rec) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for image in rec.images.values():
        if not image.name.startswith("lens0/") or not image.has_pose:
            continue
        frame = rec.frames[image.frame_id]
        rfw = frame.rig_from_world() if callable(frame.rig_from_world) else frame.rig_from_world
        R = rfw.rotation.matrix().T
        out[int(Path(image.name).stem.split(".")[0])] = (R, -R @ np.asarray(rfw.translation).reshape(3))
    return out


def lens1_from_rig(rec) -> tuple[np.ndarray, np.ndarray]:
    for rig in rec.rigs.values():
        for sensor_id, sensor_from_rig in rig.non_ref_sensors.items():
            if sensor_from_rig is not None:
                return sensor_from_rig.rotation.matrix(), np.asarray(sensor_from_rig.translation).reshape(3)
    raise RuntimeError("no lens1 extrinsic in reconstruction")


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3) if a @ b > 0 else Rx(math.pi)
    return Rotation.from_rotvec(axis / s * math.atan2(s, a @ b)).as_matrix()


def level_and_scale(args, rec, poses):
    """Returns (R_align, origin, scale, info). World after: y = down (gravity), origin = first frame."""
    pts = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3])
    centers = np.array([C for _, C in poses.values()])
    # Rough gravity: the stick/handle (+y of both lenses) points down on average.
    g0 = np.mean([R @ np.array([0.0, 1.0, 0.0]) for R, _ in poses.values()], axis=0)
    g0 /= np.linalg.norm(g0)

    # Refine with the dominant horizontal plane below the cameras (the floor) via RANSAC.
    rng = np.random.default_rng(0)
    cam_ref = np.median(centers, axis=0)
    below = pts[(pts - cam_ref) @ g0 > 0]
    extent = np.percentile(np.abs((pts - cam_ref) @ g0), 90)
    tol = 0.01 * extent
    best = (0, g0, None)
    for _ in range(3000):
        s = below[rng.choice(len(below), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ g0 < 0:
            n = -n
        if n @ g0 < math.cos(math.radians(25)):
            continue
        inl = np.abs((below - s[0]) @ n) < tol
        if inl.sum() > best[0]:
            best = (int(inl.sum()), n, below[inl])
    g = g0
    if best[2] is not None and best[0] > 200:
        P = best[2]
        _, _, vt = np.linalg.svd(P - P.mean(0))
        g = vt[2] if vt[2] @ g0 > 0 else -vt[2]
    tilt = math.degrees(math.acos(np.clip(g @ g0, -1, 1)))

    R_align = rot_between(g, np.array([0.0, 1.0, 0.0]))
    first = min(poses)
    origin = poses[first][1].copy()
    h = ((pts - origin) @ R_align.T)[:, 1]  # +down
    cam_h = np.median(((centers - origin) @ R_align.T)[:, 1])
    rel = h - cam_h

    def peak(vals):
        if len(vals) < 200:
            return None
        hist, edges = np.histogram(vals, bins=150, range=(0, np.percentile(vals, 98)))
        k = int(np.argmax(hist))
        sel = vals[(vals >= edges[max(0, k - 1)]) & (vals <= edges[min(len(edges) - 1, k + 2)])]
        return float(np.median(sel))

    floor_d = peak(rel[rel > 0])
    ceil_d = peak(-rel[rel < 0])
    info = {"floor_plane_inliers": best[0], "gravity_vs_handle_deg": round(tilt, 1),
            "floor_below_camera_sfm": floor_d, "ceiling_above_camera_sfm": ceil_d}
    if args.scale:
        s, src = args.scale, "explicit --scale"
    elif args.camera_height:
        if not floor_d:
            sys.exit("Could not find the floor; pass --scale or --room-height.")
        s, src = args.camera_height / floor_d, f"camera height {args.camera_height} m"
    elif floor_d and ceil_d:
        s, src = args.room_height / (floor_d + ceil_d), f"room height {args.room_height} m (assumed)"
    else:
        s, src = 1.0, "unscaled"
    info |= {"scale": s, "scale_source": src}
    if floor_d:
        info["camera_height_m"] = round(s * floor_d, 3)
    if floor_d and ceil_d:
        info["room_height_m"] = round(s * (floor_d + ceil_d), 3)
    return R_align, origin, s, info


# --------------------------------------------------------------------------- 5. pinhole views + tar


@dataclass
class View:
    name: str
    lens: int
    yaw: float
    pitch: float

    def lens_from_view(self) -> np.ndarray:
        return Ry(math.radians(self.yaw)) @ Rx(math.radians(self.pitch))


def build_views(tilt: float, include_down: bool) -> list[View]:
    views = []
    for lens in (0, 1):
        dirs = [(0, 0), (-tilt, 0), (tilt, 0), (0, tilt)] + ([(0, -tilt)] if include_down else [])
        for yaw, pitch in dirs:
            views.append(View(f"camera{len(views)}", lens, yaw, pitch))
    return views


def view_remap(view: View, cam, size: int, hfov: float):
    f = (size / 2) / math.tan(math.radians(hfov) / 2)
    c = (size - 1) / 2
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    d = np.stack([(u - c) / f, (v - c) / f, np.ones_like(u)], -1).reshape(-1, 3) @ view.lens_from_view().T
    valid = d[:, 2] > 0.02
    xy = np.full((len(d), 2), -1e4)
    xy[valid] = cam.img_from_cam(d[valid]) - 0.5  # COLMAP pixel centres at +0.5
    mx = xy[:, 0].reshape(size, size).astype(np.float32)
    my = xy[:, 1].reshape(size, size).astype(np.float32)
    return mx, my, f, c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="raw dual-fisheye .insv video")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second used for SfM (default 2)")
    ap.add_argument("--out-fps", type=float, default=1.0, help="frames per second written to the tar (default 1)")
    ap.add_argument("--lens-half-fov", type=float, default=103.0, help="initial half-FOV at the image circle edge")
    ap.add_argument("--max-theta", type=float, default=85.0, help="SfM uses rays within this angle of the lens axis")
    ap.add_argument("--bottom-mask", type=float, default=0.35,
                    help="mask pixels below cy + this * radius (stick/hand/operator); 1.0 disables")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--loop-closure", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--size", type=int, default=1024, help="pinhole view size (px)")
    ap.add_argument("--hfov", type=float, default=70.0, help="pinhole view FOV (deg)")
    ap.add_argument("--tilt", type=float, default=45.0, help="angle of the off-axis views (deg)")
    ap.add_argument("--include-down", action="store_true", help="also emit the downward (operator-side) views")
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--camera-height", type=float, help="lens height above the floor (m)")
    sc.add_argument("--scale", type=float, help="explicit metres per SfM unit")
    ap.add_argument("--room-height", type=float, default=2.44,
                    help="floor-to-ceiling height (m) used for scale when --camera-height is not given")
    ap.add_argument("--reuse", action="store_true", help="reuse extracted frames and SfM from a previous run")
    ap.add_argument("--max-seconds", type=float, default=None, help="only use the first N seconds of the video")
    ap.add_argument("--calib-only", action="store_true",
                    help="stop after the lens self-calibration SfM (work/sfm + meta.json); no views, no tar")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else src.parent / (src.stem + "_placeframe")
    work = out / "work"
    frames_dir, masks_dir = work / "frames", work / "masks"
    t0 = time.time()

    meta_path = work / "meta.json"
    if args.reuse and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        ts = meta["ts"]
        geoms = [LensGeom(**g) for g in meta["geoms"]]
        print(f"[1-2/5] reusing {len(ts)} frames")
    else:
        shutil.rmtree(work, ignore_errors=True)
        ts = extract_frames(src, args.fps, frames_dir, args.max_seconds)
        print("[2/5] fitting image circles and masks")
        geoms = [fit_circle(frames_dir, lens, ts) for lens in LENSES]
        for lens, g in zip(LENSES, geoms):
            print(f"  {lens}: centre ({g.cx:.1f}, {g.cy:.1f}) radius {g.radius:.1f}px")
        meta_path.write_text(json.dumps({"ts": ts, "geoms": [g.__dict__ for g in geoms]}))
    f_init = float(np.mean([g.radius for g in geoms])) / math.radians(args.lens_half_fov)

    sfm_dir = work / "sfm"
    if args.reuse and (sfm_dir / "images.bin").exists():
        rec = pycolmap.Reconstruction(str(sfm_dir))
        print(f"[3/5] reusing SfM ({rec.num_reg_frames()} frames)")
    else:
        write_sfm_masks(frames_dir, masks_dir, ts, geoms, f_init, args.max_theta, args.bottom_mask)
        rec = run_sfm(args, frames_dir, masks_dir, work, geoms, f_init)

    poses = frame_poses(rec)
    cams = {}
    for image in rec.images.values():
        cams.setdefault(image.name.split("/")[0], rec.cameras[image.camera_id])
    R10, t10 = lens1_from_rig(rec)
    ang = math.degrees(Rotation.from_matrix(R10 @ Ry(math.pi).T).magnitude())
    print(f"  registered {len(poses)}/{len(ts)} frames, {len(rec.points3D)} points")
    for lens in LENSES:
        print(f"  {lens} intrinsics: {np.round(cams[lens].params, 4).tolist()}")
    print(f"  lens1 extrinsic: {ang:.2f}° from ideal back-to-back, offset {np.round(t10, 4).tolist()} (SfM units)")

    if args.calib_only:
        print(f"Calibration done in {time.time() - t0:.1f}s -> {work}")
        return

    print("[4/5] levelling + scaling")
    R_align, origin, s, info = level_and_scale(args, rec, poses)
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("[5/5] rendering pinhole views and writing tar")
    views = build_views(args.tilt, args.include_down)
    lens_from_rig = [(np.eye(3), np.zeros(3)), (R10, t10 * s)]
    remaps = [view_remap(v, cams[LENSES[v.lens]], args.size, args.hfov) for v in views]
    f_px, c_px = remaps[0][2], remaps[0][3]
    masks = []
    for v, (mx, my, _, _) in zip(views, remaps):
        g = geoms[v.lens]
        sample = cv2.imread(str(frames_dir / LENSES[v.lens] / f"{ts[0]}.jpg"), cv2.IMREAD_GRAYSCALE)
        keep = operator_mask(sample.shape, g, args.bottom_mask)
        keep = np.where(np.hypot(*np.meshgrid(np.arange(sample.shape[1]) - g.cx,
                                              np.arange(sample.shape[0]) - g.cy)) < g.radius * 0.98, keep, 0)
        masks.append(cv2.remap(keep, mx, my, cv2.INTER_NEAREST, borderValue=0) > 0)

    step = max(1, round(args.fps / args.out_fps))
    reg_ts = sorted(poses)
    out_ts = []
    for i in range(0, len(ts), step):  # sharpest registered frame per output slot
        cands = [t for t in ts[i:i + step] if t in poses]
        if not cands:
            continue
        sharp = {t: cv2.Laplacian(cv2.imread(str(frames_dir / "lens0" / f"{t}.jpg"), 0)[::4, ::4], cv2.CV_64F).var()
                 for t in cands}
        out_ts.append(max(sharp, key=sharp.get))

    manifest_cams = []
    for i, v in enumerate(views):
        R_lr, t_lr = lens_from_rig[v.lens]
        R_vl = v.lens_from_view().T
        R_cam_rig = R_vl @ R_lr
        t_cam_rig = R_vl @ t_lr
        q = Rotation.from_matrix(R_cam_rig).as_quat() if i else np.array([0.0, 0.0, 0.0, 1.0])
        manifest_cams.append({
            "id": v.name, "ref_sensor": i == 0,
            "rotation": dict(zip("xyzw", map(float, q))),
            "translation": dict(zip("xyz", map(float, t_cam_rig if i else np.zeros(3)))),
            "camera_config": {"width": args.size, "height": args.size, "orientation": "TOP_LEFT",
                              "fx": f_px, "fy": f_px, "cx": c_px + 0.5, "cy": c_px + 0.5},
        })
    manifest = {"axis_convention": "OPENCV", "rigs": [{"id": RIG_ID, "cameras": manifest_cams}]}

    rows = ["timestamp,tx,ty,tz,qx,qy,qz,qw"]
    for t in out_ts:
        R, C = poses[t]
        Rw, Cw = R_align @ R, s * (R_align @ (C - origin))
        q = Rotation.from_matrix(Rw).as_quat()
        rows.append(f"{t},{Cw[0]:.6f},{Cw[1]:.6f},{Cw[2]:.6f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")

    tar_path = out / "capture.tar"
    preview = []
    with tarfile.open(tar_path, "w") as tar:
        def add_bytes(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = int(time.time())
            tar.addfile(ti, io.BytesIO(data))

        add_bytes("manifest.json", json.dumps(manifest, indent=2).encode())
        add_bytes(f"{RIG_ID}/frames.csv", ("\n".join(rows) + "\n").encode())
        for n, t in enumerate(out_ts):
            src_imgs = [cv2.imread(str(frames_dir / lens / f"{t}.jpg")) for lens in LENSES]
            for v, (mx, my, _, _), m in zip(views, remaps, masks):
                img = cv2.remap(src_imgs[v.lens], mx, my, cv2.INTER_LINEAR, borderValue=0)
                img[~m] = 0
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                add_bytes(f"{RIG_ID}/{v.name}/{t}.jpg", buf.tobytes())
                if n == len(out_ts) // 2:
                    preview.append(cv2.resize(img, (256, 256)))
    if preview:
        cols = len(views) // 2
        grid = np.vstack([np.hstack(preview[r * cols:(r + 1) * cols]) for r in range(2)])
        cv2.imwrite(str(out / "preview_views.jpg"), grid)

    path_len = sum(np.linalg.norm(s * R_align @ (poses[b][1] - poses[a][1])) for a, b in zip(reg_ts, reg_ts[1:]))
    summary = {
        "input": str(src), "tar": str(tar_path), "frames_sfm": len(ts), "frames_registered": len(poses),
        "frames_written": len(out_ts), "views": [v.__dict__ for v in views],
        "lens_intrinsics": {lens: cams[lens].params.tolist() for lens in LENSES},
        "lens1_from_lens0_deg_from_ideal": ang, "trajectory_length_m": path_len,
        "metric": info["scale_source"] != "unscaled", **info, "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "conversion.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  trajectory length {path_len:.2f} m, {len(out_ts)} frames x {len(views)} views")
    print(f"Done in {summary['elapsed_s']}s -> {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    print(f"Preview of the virtual views: {out / 'preview_views.jpg'}")


if __name__ == "__main__":
    main()
