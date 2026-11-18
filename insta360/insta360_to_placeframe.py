#!/usr/bin/env python3
"""Convert Insta360 360° footage into a Placeframe capture-session tar.

Input:  an equirectangular 360 video (MP4 exported from Insta360 Studio) or a
        folder of equirectangular images (exported .insp photos / timelapse).
Output: <out>/capture.tar  — ready for `upload_to_placeframe.py`
        <out>/conversion.json — stats, scale info, frame counts

Pipeline:
  1. Sample frames at --fps, drop motion-blurred ones.
  2. Re-project each 360 frame into N virtual pinhole cameras (a camera rig with
     a shared optical centre — exactly what Placeframe's multi-camera rig format
     expects).
  3. Run a rig-aware SfM pass (pycolmap / COLMAP SIFT) to recover a pose per frame,
     since an Insta360 has no AR tracking to supply them.
  4. Level the world to gravity, set metric scale, write manifest.json +
     rig0/frames.csv + rig0/cameraK/<ms>.jpg and tar it up.
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


def _preload_cuda_libs() -> None:
    # pycolmap-cuda12 links libcudart/libcurand but the pip-installed copies (nvidia-* wheels)
    # are not on the loader path; load them globally before importing pycolmap.
    import ctypes
    import glob
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        for lib in ("cuda_runtime/lib/libcudart.so.*", "curand/lib/libcurand.so.*"):
            for path in glob.glob(f"{root}/{lib}"):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _cuda_device_count() -> int:
    import ctypes

    try:
        cudart = ctypes.CDLL("libcudart.so.12")
    except OSError:
        return 0
    n = ctypes.c_int(0)
    return n.value if cudart.cudaGetDeviceCount(ctypes.byref(n)) == 0 else 0


_preload_cuda_libs()
import pycolmap  # noqa: E402
from scipy.spatial.transform import Rotation

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".insp"}
RIG_ID = "rig0"


VOCAB_TREE_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
VOCAB_TREE_SHA256 = "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"


def vocab_tree_path() -> str:
    """Download (once) and cache COLMAP's vocab tree; the pip pycolmap build can't fetch it itself."""
    import hashlib
    import urllib.request

    path = Path.home() / ".cache" / "placeframe-insta360" / Path(VOCAB_TREE_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading vocab tree for loop closure -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(VOCAB_TREE_URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != VOCAB_TREE_SHA256:
            tmp.unlink()
            sys.exit("vocab tree download failed checksum")
        tmp.rename(path)
    return str(path)


# --------------------------------------------------------------------------- views


@dataclass
class View:
    name: str
    yaw_deg: float
    pitch_deg: float

    def rig_from_cam(self) -> np.ndarray:
        # OpenCV camera axes: x right, y down, z forward. Rig frame == camera0 frame.
        # Positive yaw turns right (about +y), positive pitch looks up.
        y, p = math.radians(self.yaw_deg), math.radians(self.pitch_deg)
        ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        return ry @ rx


def parse_views(spec: str | None, count: int, pitch: float) -> list[View]:
    if spec:
        pairs = [tuple(float(v) for v in item.split(":")) for item in spec.split(",")]
    else:
        pairs = [(i * 360.0 / count, pitch) for i in range(count)]
    if abs(pairs[0][0]) > 1e-9 or abs(pairs[0][1]) > 1e-9:
        sys.exit("The first view must be yaw 0, pitch 0 (it is the rig's reference sensor).")
    return [View(f"camera{i}", yaw, pit) for i, (yaw, pit) in enumerate(pairs)]


def build_remap(view: View, size: int, hfov_deg: float, eq_w: int, eq_h: int):
    f = (size / 2) / math.tan(math.radians(hfov_deg) / 2)
    c = (size - 1) / 2
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    d = np.stack([(u - c) / f, (v - c) / f, np.ones_like(u)], axis=-1)
    d = d @ view.rig_from_cam().T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 0], d[..., 2])  # 0 = centre of the equirect image
    lat = -np.arcsin(np.clip(d[..., 1], -1, 1))  # +up
    map_x = ((lon / (2 * math.pi) + 0.5) * eq_w - 0.5).astype(np.float32)
    map_y = ((0.5 - lat / math.pi) * eq_h - 0.5).astype(np.float32)
    return map_x, map_y, f, c


# --------------------------------------------------------------------------- frame sources


def iter_video_frames(path: Path, fps: float):
    """Yield (timestamp_ms, bgr_frame). Prefers ffmpeg (fast fps filter, HEVC/10-bit), else OpenCV."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        w, h = int(stream["width"]), int(stream["height"])
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", str(path), "-vf", f"fps={fps}",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        frame_bytes = w * h * 3
        n = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield round(n * 1000 / fps), np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
        proc.wait()
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Could not open video {path} (install ffmpeg with install.sh)")
    next_t = 0.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t + 1e-6 >= next_t:
            ok, frame = cap.retrieve()
            if ok:
                yield round(t), frame
            next_t += 1000 / fps
    cap.release()


def iter_image_frames(folder: Path, interval_ms: int):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for i, p in enumerate(files):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ! skipping unreadable {p.name}")
            continue
        yield i * interval_ms, img


def sharpness(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    # Middle band only: poles of an equirect are stretched and dominated by sky/floor/stick.
    band = img[h // 4 : 3 * h // 4]
    small = cv2.resize(band, (1024, max(1, int(1024 * band.shape[0] / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# --------------------------------------------------------------------------- main stages


def extract_views(args, views: list[View], images_dir: Path) -> tuple[list[int], dict]:
    src = Path(args.input)
    frames = iter_image_frames(src, args.image_interval_ms) if src.is_dir() else iter_video_frames(src, args.fps)

    remaps = None
    intr: dict = {}
    raw: list[tuple[int, float]] = []
    for ts, eq in frames:
        eh, ew = eq.shape[:2]
        if abs(ew - 2 * eh) > 4:
            sys.exit(
                f"Frame is {ew}x{eh}, not 2:1 equirectangular. Export from Insta360 Studio as a 360 "
                "(equirectangular) video/photo, not a reframed or dual-fisheye file."
            )
        if remaps is None:
            remaps = [build_remap(v, args.size, args.hfov, ew, eh) for v in views]
            f, c = remaps[0][2], remaps[0][3]
            # COLMAP puts pixel centres at +0.5, OpenCV (used for the remap) at 0.
            intr = {"width": args.size, "height": args.size, "fx": f, "fy": f, "cx": c + 0.5, "cy": c + 0.5}
        raw.append((ts, sharpness(eq)))
        for v, (mx, my, _, _) in zip(views, remaps):
            out = cv2.remap(eq, mx, my, cv2.INTER_AREA if args.size < ew / 4 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            d = images_dir / RIG_ID / v.name
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{ts}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if len(raw) % 25 == 0:
            print(f"  extracted {len(raw)} frames (t={ts / 1000:.1f}s)")

    if not raw:
        sys.exit("No frames read from input.")

    # Drop blurry frames relative to their temporal neighbourhood.
    scores = np.array([s for _, s in raw])
    keep: list[int] = []
    for i, (ts, s) in enumerate(raw):
        lo, hi = max(0, i - 4), min(len(raw), i + 5)
        if s >= args.blur_ratio * float(np.median(scores[lo:hi])):
            keep.append(ts)
        else:
            for v in views:
                (images_dir / RIG_ID / v.name / f"{ts}.jpg").unlink(missing_ok=True)
    print(f"  kept {len(keep)}/{len(raw)} frames after blur filtering")
    return keep, intr


def run_sfm(args, views: list[View], intr: dict, images_dir: Path, work: Path) -> pycolmap.Reconstruction:
    db_path = work / "database.db"
    db_path.unlink(missing_ok=True)
    sfm_dir = work / "sfm"
    shutil.rmtree(sfm_dir, ignore_errors=True)
    sfm_dir.mkdir(parents=True)

    use_gpu = pycolmap.has_cuda and not args.cpu and _cuda_device_count() > 0
    if pycolmap.has_cuda and not args.cpu and not use_gpu:
        print("  ! CUDA build of pycolmap but no usable GPU (check nvidia-smi) — falling back to CPU")
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{intr['fx']},{intr['fy']},{intr['cx']},{intr['cy']}"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = args.max_features
    extraction.use_gpu = use_gpu

    print(f"[sfm] extracting SIFT features ({'GPU' if use_gpu else 'CPU'})")
    pycolmap.extract_features(
        str(db_path), str(images_dir), camera_mode=pycolmap.CameraMode.PER_FOLDER, camera_model="PINHOLE",
        reader_options=reader, extraction_options=extraction, device=device,
    )

    rig_cams = []
    for v in views:
        cam = pycolmap.RigConfigCamera()
        cam.image_prefix = f"{RIG_ID}/{v.name}/"
        cam.ref_sensor = v.name == views[0].name
        if not cam.ref_sensor:
            cam.cam_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(v.rig_from_cam().T), np.zeros(3))
        rig_cams.append(cam)
    rig_config = pycolmap.RigConfig()
    rig_config.cameras = rig_cams
    db = pycolmap.Database.open(str(db_path))
    pycolmap.apply_rig_config([rig_config], db)
    db.close()

    print("[sfm] matching (sequential, rig-expanded)")
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = use_gpu
    matching.skip_image_pairs_in_same_frame = True  # zero-baseline views of the same instant
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = args.overlap
    pairing.expand_rig_images = True
    pairing.loop_detection = args.loop_closure
    if args.loop_closure:
        pairing.vocab_tree_path = vocab_tree_path()
    pycolmap.match_sequential(str(db_path), matching_options=matching, pairing_options=pairing, device=device)

    print("[sfm] incremental mapping")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.ba_refine_sensor_from_rig = False  # virtual rig is exact by construction
    if args.min_inliers:
        opts.mapper.abs_pose_min_num_inliers = args.min_inliers
        opts.min_num_matches = min(opts.min_num_matches, args.min_inliers)
    recs = pycolmap.incremental_mapping(str(db_path), str(images_dir), str(sfm_dir), options=opts)
    if not recs:
        sys.exit("SfM failed: no reconstruction. Try a lower --fps blur ratio, more --overlap, or slower walking.")
    best = max(recs.values(), key=lambda r: r.num_reg_frames())
    if len(recs) > 1:
        sizes = sorted((r.num_reg_frames() for r in recs.values()), reverse=True)
        print(f"  ! SfM produced {len(recs)} disconnected models (frames: {sizes}); keeping the largest.")
    return best


def frame_poses(rec: pycolmap.Reconstruction) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """timestamp_ms -> (R_world_from_rig, C_rig_centre_in_world)."""
    prefix = f"{RIG_ID}/camera0/"
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for image in rec.images.values():
        if not image.name.startswith(prefix) or not image.has_pose:
            continue
        frame = rec.frames[image.frame_id]
        rfw = frame.rig_from_world() if callable(frame.rig_from_world) else frame.rig_from_world
        R = rfw.rotation.matrix().T
        C = -R @ np.asarray(rfw.translation).reshape(3)
        out[int(Path(image.name).stem)] = (R, C)
    return out


def gravity_alignment(poses) -> np.ndarray:
    # Insta360 exports are horizon-levelled (FlowState), so each rig's +y (down) is gravity.
    downs = np.array([R @ np.array([0.0, 1.0, 0.0]) for R, _ in poses.values()])
    g = downs.mean(axis=0)
    consistency = float(np.linalg.norm(g))  # 1.0 = every frame's "down" agrees
    spread_deg = math.degrees(math.acos(min(1.0, consistency))) * math.sqrt(2)
    if consistency < 0.97:
        print(f"  ! frames disagree on 'down' by ~{spread_deg:.0f}° rms: the input does not look horizon-locked. "
              "Levelling is approximate; export with FlowState + horizon lock, or use the raw .insv + IMU path.")
    else:
        print(f"  input is horizon-locked (frames agree on 'down' within ~{spread_deg:.1f}° rms)")
    g /= np.linalg.norm(g)
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(g, target)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3) if g @ target > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(axis / s * math.atan2(s, g @ target)).as_matrix()


def estimate_floor_distance(rec, poses, R_align, origin, side: float = 1.0) -> float | None:
    """Distance (unscaled units) from the median camera height to the dominant horizontal plane below it
    (side=+1, the floor) or above it (side=-1, the ceiling)."""
    if not rec.points3D:
        return None
    pts = np.array([p.xyz for p in rec.points3D.values()])
    pts = (R_align @ (pts - origin).T).T
    cam_y = np.median([(R_align @ (C - origin))[1] for _, C in poses.values()])
    h = side * (pts[:, 1] - cam_y)  # +down for the floor, +up for the ceiling
    h = h[h > 0]
    if len(h) < 200:
        return None
    hi = np.percentile(h, 98)
    hist, edges = np.histogram(h, bins=120, range=(0, hi))
    k = int(np.argmax(hist))
    lo_e, hi_e = edges[max(0, k - 1)], edges[min(len(edges) - 1, k + 2)]
    return float(np.median(h[(h >= lo_e) & (h <= hi_e)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="equirectangular 360 video (.mp4) or folder of equirectangular images")
    ap.add_argument("-o", "--out", default=None, help="output directory (default: <input>_placeframe)")
    ap.add_argument("--fps", type=float, default=2.0, help="frames sampled per second of video (default 2)")
    ap.add_argument("--image-interval-ms", type=int, default=500, help="fake timestamp spacing for image folders")
    ap.add_argument("--views", type=int, default=6, help="number of virtual cameras around the horizon (default 6)")
    ap.add_argument("--pitch", type=float, default=0.0, help="pitch (deg) of the horizon ring; ref view stays at 0")
    ap.add_argument("--view-spec", default=None, help='explicit "yaw:pitch,..." list; first must be 0:0')
    ap.add_argument("--hfov", type=float, default=90.0, help="virtual camera field of view in degrees (default 90)")
    ap.add_argument("--size", type=int, default=1024, help="virtual camera image size in px (square, default 1024)")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--blur-ratio", type=float, default=0.6,
                    help="drop frames less sharp than this fraction of the local median (0 disables)")
    ap.add_argument("--max-features", type=int, default=4096, help="SIFT features per view for the pose pass")
    ap.add_argument("--overlap", type=int, default=10, help="sequential matching window, in frames")
    ap.add_argument("--min-inliers", type=int, default=None,
                    help="min 2D-3D inliers to register a frame (COLMAP default 30); lower helps weak hallways")
    ap.add_argument("--tar-step", type=int, default=1,
                    help="write every Nth registered frame to the tar (SfM at high fps, server at lower fps)")
    ap.add_argument("--loop-closure", action="store_true",
                    help="enable vocab-tree loop detection (downloads a vocab tree on first use)")
    scale = ap.add_mutually_exclusive_group()
    scale.add_argument("--camera-height", type=float, default=None,
                       help="height of the camera lens above the floor in metres; used to set metric scale")
    scale.add_argument("--scale", type=float, default=None, help="explicit metres-per-SfM-unit multiplier")
    scale.add_argument("--room-height", type=float, default=None,
                       help="floor-to-ceiling height in metres (uses both planes); alternative to --camera-height")
    ap.add_argument("--cpu", action="store_true", help="force CPU SIFT even if a GPU is available")
    ap.add_argument("--no-gravity-align", action="store_true", help="skip levelling the world to the horizon")
    ap.add_argument("--keep-work", action="store_true", help="keep the intermediate SfM workspace")
    ap.add_argument("--views-only", action="store_true",
                    help="only cut the pinhole views (poses come from elsewhere, e.g. VIO via vio_to_capture.py)")
    ap.add_argument("--reuse-sfm", action="store_true", help="reuse work/sfm from a --keep-work run (e.g. to rescale)")
    ap.add_argument("--reuse-frames", action="store_true", help="skip extraction if work/images already exists")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        sys.exit(f"{src} not found")
    out = Path(args.out).expanduser().resolve() if args.out else src.with_name(src.stem + "_placeframe")
    work = out / "work"
    images_dir = work / "images"
    out.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.view_spec, args.views, args.pitch)
    t0 = time.time()

    meta_path = work / "frames.json"
    if args.reuse_frames and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        kept, intr = meta["kept"], meta["intrinsics"]
        print(f"[1/4] reusing {len(kept)} extracted frames")
    else:
        shutil.rmtree(work, ignore_errors=True)
        images_dir.mkdir(parents=True)
        print(f"[1/4] extracting frames from {src.name} into {len(views)} virtual {args.hfov:.0f}° views")
        kept, intr = extract_views(args, views, images_dir)
        meta_path.write_text(json.dumps({"kept": kept, "intrinsics": intr}))
    if len(kept) < 10:
        sys.exit("Fewer than 10 usable frames — record a longer walk or raise --fps.")
    if args.views_only:
        print(f"views written to {images_dir} ({len(kept)} frames); --views-only: skipping SfM")
        return

    print("[2/4] recovering frame poses with SfM")
    prev = sorted((work / "sfm").glob("*/images.bin")) if args.reuse_sfm else []
    if prev:
        rec = max((pycolmap.Reconstruction(str(p.parent)) for p in prev), key=lambda r: r.num_reg_frames())
        print("  reusing previous SfM model")
    else:
        rec = run_sfm(args, views, intr, images_dir, work)
    poses = frame_poses(rec)
    print(f"  registered {len(poses)}/{len(kept)} frames, {len(rec.points3D)} points")
    if len(poses) < 0.5 * len(kept):
        print("  ! less than half the frames registered — the map will only cover part of the walk.")

    print("[3/4] levelling and scaling")
    first_ts = min(poses)
    origin = poses[first_ts][1].copy()
    R_align = np.eye(3) if args.no_gravity_align else gravity_alignment(poses)
    scale_source = "unscaled"
    s = 1.0
    floor = None
    if args.scale:
        s, scale_source = args.scale, "explicit --scale"
    elif args.camera_height:
        floor = estimate_floor_distance(rec, poses, R_align, origin)
        if floor is None or floor <= 0:
            sys.exit("Could not find a floor plane for --camera-height; pass --scale instead.")
        s, scale_source = args.camera_height / floor, f"camera height {args.camera_height} m"
    elif args.room_height:
        floor = estimate_floor_distance(rec, poses, R_align, origin)
        ceiling = estimate_floor_distance(rec, poses, R_align, origin, side=-1.0)
        if not floor or not ceiling:
            sys.exit("Could not find both floor and ceiling for --room-height; pass --camera-height or --scale.")
        s, scale_source = args.room_height / (floor + ceiling), f"room height {args.room_height} m"
        print(f"  implied camera height {s * floor:.2f} m")
    else:
        print("  ! no --camera-height or --scale given: map units are arbitrary, AR content will be mis-sized.")
    print(f"  scale factor {s:.4f} ({scale_source})")

    # Re-express poses in the levelled, scaled world (OpenCV axes: y = down, origin = first frame).
    final: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ts, (R, C) in poses.items():
        final[ts] = (R_align @ R, s * (R_align @ (C - origin)))
    path_len = sum(np.linalg.norm(final[b][1] - final[a][1]) for a, b in zip(sorted(final), sorted(final)[1:]))
    print(f"  trajectory length {path_len:.2f} {'m' if scale_source != 'unscaled' else 'units'}")

    print("[4/4] writing Placeframe capture tar")
    manifest = {
        "axis_convention": "OPENCV",
        "rigs": [{
            "id": RIG_ID,
            "cameras": [
                {
                    "id": v.name,
                    "ref_sensor": i == 0,
                    "rotation": dict(zip("xyzw", Rotation.from_matrix(v.rig_from_cam().T).as_quat().tolist()
                                         if i else [0.0, 0.0, 0.0, 1.0])),
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "camera_config": {
                        "width": intr["width"], "height": intr["height"], "orientation": "TOP_LEFT",
                        "fx": intr["fx"], "fy": intr["fy"], "cx": intr["cx"], "cy": intr["cy"],
                    },
                }
                for i, v in enumerate(views)
            ],
        }],
    }
    final = {ts: final[ts] for ts in sorted(final)[:: max(1, args.tar_step)]}
    rows = ["timestamp,tx,ty,tz,qx,qy,qz,qw"]
    for ts in sorted(final):
        R, C = final[ts]
        q = Rotation.from_matrix(R).as_quat()
        rows.append(f"{ts},{C[0]:.6f},{C[1]:.6f},{C[2]:.6f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")

    tar_path = out / "capture.tar"
    with tarfile.open(tar_path, "w") as tar:
        def add_bytes(name: str, data: bytes):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))

        add_bytes("manifest.json", json.dumps(manifest, indent=2).encode())
        add_bytes(f"{RIG_ID}/frames.csv", ("\n".join(rows) + "\n").encode())
        for ts in sorted(final):
            for v in views:
                tar.add(images_dir / RIG_ID / v.name / f"{ts}.jpg", arcname=f"{RIG_ID}/{v.name}/{ts}.jpg")

    summary = {
        "input": str(src),
        "tar": str(tar_path),
        "frames_sampled_kept": len(kept),
        "frames_registered": len(poses),
        "views": [{"id": v.name, "yaw": v.yaw_deg, "pitch": v.pitch_deg} for v in views],
        "intrinsics": intr,
        "scale_factor": s,
        "scale_source": scale_source,
        "metric": scale_source != "unscaled",
        "floor_distance_sfm_units": floor,
        "trajectory_length": path_len,
        "sfm_points": len(rec.points3D),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "conversion.json").write_text(json.dumps(summary, indent=2))
    if not args.keep_work:
        shutil.rmtree(work / "sfm", ignore_errors=True)
        (work / "database.db").unlink(missing_ok=True)
    print(f"\nDone in {summary['elapsed_s']}s -> {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    print(f"Next: uv run upload_to_placeframe.py {tar_path}")


if __name__ == "__main__":
    main()
