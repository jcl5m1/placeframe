#!/usr/bin/env python3
"""Add another Insta360 recording to an existing Placeframe map.

Placeframe has no "merge" operation: a map is one reconstruction of one capture session. So merging means
building a new capture session that holds the old frames *and* the new ones, with every pose in the old
map's coordinate frame, and reconstructing that:

  1 calib      lens self-calibration SfM on the first minutes of the new video
  2 frames     extract both fisheye streams at 4 fps
  3 stitch     gravity-levelled equirect frames
  4 views      six pinhole views per frame
  5 vio        open-loop visual-inertial pose for every video frame (the new video's own frame)
  6 capture    capture.tar for the new video alone
  7 align      localize a sample of the new frames against the existing map -> rigid transform (+ drift fix)
  8 combine    old capture's rigs + the new video as one more rig, all in the old map's frame
  9 reconstruct  upload, reconstruct, publish a new localization map (the old one is left untouched)
 10 viewer     point cloud + camera path for the review UI

Every stage leaves a marker in <job-dir>/.done, so a failed or interrupted job resumes where it stopped.

  uv run merge_video.py NEW.insv --job-dir ~/pf_x5/jobs/new --base-map x5-house-2026-11-07
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
from scipy.spatial.transform import Rotation

from upload_to_placeframe import (
    DEFAULT_PLACEFRAME_DIR,
    TERMINAL,
    choose_reconstruction_options,
    get_token,
    public_domain,
    saved_credentials,
)

HERE = Path(__file__).resolve().parent
STAGES = ["calib", "frames", "stitch", "views", "vio", "capture", "align", "combine", "reconstruct", "viewer"]
# Views tried per sampled frame when localizing, most-separated headings first.
LOCALIZE_VIEW_ORDER = [0, 3, 1, 4, 2, 5]


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str]) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=HERE)


def script(name: str, *args) -> list[str]:
    return [sys.executable, "-u", str(HERE / name), *[str(a) for a in args]]


# --------------------------------------------------------------------------- Placeframe API


class Placeframe:
    def __init__(self, args) -> None:
        pf_dir = Path(args.placeframe_dir)
        creds = saved_credentials(pf_dir)
        self.user = creds.get("PLACEFRAME_USER", "user")
        self.password = creds.get("PLACEFRAME_PASSWORD", "password")
        self.realm = args.realm
        server = args.server
        if not server:
            domain = public_domain(pf_dir)
            if not domain:
                sys.exit("No --server given and PUBLIC_DOMAIN not set in placeframe/.env")
            server = f"https://{domain}"
        self.server = server.rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(60, write=None, read=None))
        self.login()

    def login(self) -> None:
        token = get_token(self.client, self.server, self.realm, self.user, self.password)
        self.client.headers["Authorization"] = f"Bearer {token}"

    def request(self, method: str, path: str, **kw) -> httpx.Response:
        """One request, surviving token expiry and brief server/network hiccups."""
        delay = 2.0
        for attempt in range(8):
            try:
                r = self.client.request(method, self.server + path, **kw)
            except httpx.TransportError as e:
                log(f"  ({method} {path}: {e!r}; retrying in {delay:.0f}s)")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 401:
                self.login()
                continue
            if r.status_code in (502, 503, 504) and attempt < 7:
                log(f"  ({method} {path}: server {r.status_code}; retrying in {delay:.0f}s)")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            return r
        sys.exit(f"{method} {path} kept failing")

    def json(self, method: str, path: str, what: str, **kw):
        r = self.request(method, path, **kw)
        if r.status_code >= 300:
            sys.exit(f"{what} failed ({r.status_code}): {r.text[:500]}")
        return r.json()

    def find_map(self, name_or_id: str) -> dict:
        maps = self.json("GET", "/localization-maps", "List localization maps")
        hits = [m for m in maps if m["id"] == name_or_id or m.get("name") == name_or_id]
        if not hits:
            sys.exit(f"No localization map named '{name_or_id}' on {self.server} "
                     f"(have: {sorted({m.get('name') for m in maps})})")
        return max(hits, key=lambda m: m["created_at"])


# --------------------------------------------------------------------------- stages 1-6: the new video alone


def video_duration(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def stage_calib(args, job: Path) -> None:
    cmd = script("insta360_dualfisheye_to_placeframe.py", args.video, "-o", job / "calib", "--loop-closure",
                 "--calib-only")
    duration = video_duration(args.video)
    log(f"video is {duration / 60:.1f} min")
    if duration > args.calib_seconds:
        cmd += ["--max-seconds", str(args.calib_seconds)]
    run(cmd)


def stage_frames(args, job: Path) -> None:
    from insta360_dualfisheye_to_placeframe import extract_frames

    work = job / "work"
    shutil.rmtree(work, ignore_errors=True)
    ts = extract_frames(args.video, args.fps, work / "frames")
    calib_meta = json.loads((job / "calib/work/meta.json").read_text())
    (work / "meta.json").write_text(json.dumps({"ts": ts, "geoms": calib_meta["geoms"]}))
    shutil.copytree(job / "calib/work/sfm", work / "sfm")


def stage_stitch(args, job: Path) -> None:
    shutil.rmtree(job / "equirect", ignore_errors=True)
    run(script("stitch_dualfisheye.py", args.video, "--work", job / "work", "-o", job / "equirect",
               "--bottom-mask", args.stitch_bottom_mask, "--imu-offset-ms", args.imu_offset_ms))


def stage_views(args, job: Path) -> None:
    from insta360_to_placeframe import extract_views, parse_views

    views_dir = job / "views"
    shutil.rmtree(views_dir, ignore_errors=True)
    (views_dir / "images").mkdir(parents=True)
    opts = SimpleNamespace(input=str(job / "equirect"), image_interval_ms=round(1000 / args.fps), fps=args.fps,
                           size=1024, hfov=90.0, jpeg_quality=92, blur_ratio=0.6)
    kept, intr = extract_views(opts, parse_views(None, 6, 0.0), views_dir / "images")
    (views_dir / "frames.json").write_text(json.dumps({"kept": kept, "intrinsics": intr}))


def stage_vio(args, job: Path) -> None:
    run(script("vio_dualfisheye.py", args.video, "--calib", job / "calib/work", "-o", job / "vio",
               "--imu-offset-ms", args.imu_offset_ms))


def stage_capture(args, job: Path) -> None:
    run(script("vio_to_capture.py", "--insv", args.video, "--vio", job / "vio/vio.csv", "--stitch-work", job / "work",
               "--equirect", job / "equirect", "--views", job / "views", "-o", job / "capture",
               "--step", args.tar_step))


# --------------------------------------------------------------------------- stage 7: align to the map


def read_capture(tar_path: Path) -> tuple[dict, dict[str, list[tuple[int, np.ndarray, np.ndarray]]]]:
    """manifest, {rig_id: [(timestamp, position, quaternion xyzw)]} from a capture tar."""
    with tarfile.open(tar_path) as tf:
        manifest = json.load(tf.extractfile("manifest.json"))
        frames = {}
        for rig in manifest["rigs"]:
            rows = tf.extractfile(f"{rig['id']}/frames.csv").read().decode().splitlines()[1:]
            frames[rig["id"]] = [(int(float(c[0])), np.array(c[1:4], float), np.array(c[4:8], float))
                                 for c in (r.split(",") for r in rows if r.strip())]
    return manifest, frames


def fit_yaw_translation(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares dst ≈ Ry(yaw) @ src + t. Both worlds are gravity-levelled (OpenCV, y down)."""
    cs, cd = src.mean(0), dst.mean(0)
    a, b = src - cs, dst - cd
    # Rotation about y: x' = c*x + s*z, z' = -s*x + c*z
    num = np.sum(a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2])
    den = np.sum(a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2])
    yaw = math.atan2(num, den)
    return yaw, cd - yaw_matrix(yaw) @ cs


def yaw_matrix(yaw: float) -> np.ndarray:
    return Rotation.from_rotvec([0.0, yaw, 0.0]).as_matrix()


def ransac_yaw_translation(src: np.ndarray, dst: np.ndarray, threshold: float, rng) -> tuple[float, np.ndarray, np.ndarray]:
    best = np.zeros(len(src), bool)
    n = len(src)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > 2000:
        pairs = [pairs[k] for k in rng.choice(len(pairs), 2000, replace=False)]
    for i, j in pairs:
        if np.linalg.norm(src[i] - src[j]) < 0.5:  # too close together to pin down a heading
            continue
        yaw, t = fit_yaw_translation(src[[i, j]], dst[[i, j]])
        inl = np.linalg.norm((yaw_matrix(yaw) @ src.T).T + t - dst, axis=1) < threshold
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 2:
        return 0.0, np.zeros(3), best
    yaw, t = fit_yaw_translation(src[best], dst[best])
    best = np.linalg.norm((yaw_matrix(yaw) @ src.T).T + t - dst, axis=1) < threshold
    yaw, t = fit_yaw_translation(src[best], dst[best])
    return yaw, t, best


def localize_frames(pf: Placeframe, tar_path: Path, map_id: str, count: int, min_inliers: int) -> list[dict]:
    """Localize an even sample of a capture's frames against a map; returns per-frame map-frame rig poses."""
    manifest, frames = read_capture(tar_path)
    rig = manifest["rigs"][0]
    rows = frames[rig["id"]]
    picks = sorted({int(round(k)) for k in np.linspace(0, len(rows) - 1, min(count, len(rows)))})
    cams = rig["cameras"]
    hits = []
    with tarfile.open(tar_path) as tf:
        for n, idx in enumerate(picks):
            ts, pos, quat = rows[idx]
            found = None
            for ci in [c for c in LOCALIZE_VIEW_ORDER if c < len(cams)][: 3]:
                cam = cams[ci]
                image = tf.extractfile(f"{rig['id']}/{cam['id']}/{ts}.jpg").read()
                r = pf.request("POST", "/localize", data={
                    "map_ids": json.dumps([map_id]), "camera_config": json.dumps(cam["camera_config"]),
                    "axis_convention": manifest["axis_convention"],
                }, files={"image": (f"{ts}.jpg", image, "image/jpeg")})
                if r.status_code >= 300 or not r.json():
                    continue
                loc = r.json()[0]
                if loc["metrics"]["num_inliers"] < min_inliers:
                    continue
                tr = loc["camera_from_map_transform"]
                R_cam_map = Rotation.from_quat([tr["rotation"][k] for k in "xyzw"]).as_matrix()
                t_cam_map = np.array([tr["translation"][k] for k in "xyz"])
                R_cam_rig = Rotation.from_quat([cam["rotation"][k] for k in "xyzw"]).as_matrix()
                t_cam_rig = np.array([cam["translation"][k] for k in "xyz"])
                R_map_rig = R_cam_map.T @ R_cam_rig
                p_map_rig = R_cam_map.T @ (t_cam_rig - t_cam_map)
                found = {"timestamp": ts, "camera": cam["id"], "inliers": loc["metrics"]["num_inliers"],
                         "map_position": p_map_rig.tolist(), "map_rotation": R_map_rig.tolist(),
                         "capture_position": pos.tolist(), "capture_quat": quat.tolist()}
                break
            if found:
                hits.append(found)
            if (n + 1) % 10 == 0 or n + 1 == len(picks):
                log(f"  localized {len(hits)} of {n + 1} sampled frames ({len(picks)} to try)")
    return hits


def heading(R: np.ndarray) -> float:
    return math.atan2(R[0, 2], R[2, 2])


def stage_align(args, job: Path, pf: Placeframe, base: dict) -> None:
    tar_path = job / "capture/capture.tar"
    log(f"localizing new frames against map '{base['name']}' ({base['id']})")
    hits = localize_frames(pf, tar_path, base["id"], args.align_samples, args.align_min_inliers)
    if len(hits) < args.align_min_frames:
        sys.exit(f"Only {len(hits)} of the sampled frames could be localized in the existing map "
                 f"(need {args.align_min_frames}). The new recording must revisit part of the mapped space.")
    src = np.array([h["capture_position"] for h in hits])
    dst = np.array([h["map_position"] for h in hits])
    yaw, t, inl = ransac_yaw_translation(src, dst, args.align_threshold, np.random.default_rng(0))
    if inl.sum() < args.align_min_frames:
        sys.exit(f"Localizations disagree with the VIO path: only {int(inl.sum())} of {len(hits)} fit one rigid "
                 f"transform within {args.align_threshold} m (need {args.align_min_frames}).")
    Ry = yaw_matrix(yaw)
    resid = dst - ((Ry @ src.T).T + t)
    err = np.linalg.norm(resid[inl], axis=1)
    head_err = [abs((heading(np.array(h["map_rotation"])) - heading(Ry @ Rotation.from_quat(h["capture_quat"]).as_matrix())
                     + math.pi) % (2 * math.pi) - math.pi) for h, ok in zip(hits, inl) if ok]
    log(f"  transform: yaw {math.degrees(yaw):.1f}°, translation {np.round(t, 2).tolist()} m")
    log(f"  {int(inl.sum())}/{len(hits)} localizations agree; residual median {np.median(err):.2f} m, "
        f"max {err.max():.2f} m; heading residual median {math.degrees(np.median(head_err)):.1f}°")

    # Pin the path to the map where it was localized: interpolate each anchor's leftover offset over time.
    # Between anchors (new territory) the path keeps its VIO shape.
    _, frames = read_capture(tar_path)
    rows = next(iter(frames.values()))
    anchor_t = np.array([h["timestamp"] for h, ok in zip(hits, inl) if ok], float)
    anchor_r = resid[inl]
    out = ["timestamp,tx,ty,tz,qx,qy,qz,qw"]
    for ts, pos, quat in rows:
        p = Ry @ pos + t
        if not args.no_drift_correction:
            p = p + np.array([np.interp(ts, anchor_t, anchor_r[:, k]) for k in range(3)])
        q = Rotation.from_matrix(Ry @ Rotation.from_quat(quat).as_matrix()).as_quat()
        out.append(f"{ts},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")
    (job / "aligned_frames.csv").write_text("\n".join(out) + "\n")
    (job / "align.json").write_text(json.dumps({
        "base_map": base, "yaw_deg": math.degrees(yaw), "translation": t.tolist(),
        "localized": len(hits), "inliers": int(inl.sum()), "residual_median_m": float(np.median(err)),
        "residual_max_m": float(err.max()), "heading_residual_median_deg": math.degrees(float(np.median(head_err))),
        "drift_corrected": not args.no_drift_correction,
        "anchors": [dict(h, inlier=bool(ok)) for h, ok in zip(hits, inl)],
    }, indent=1))


# --------------------------------------------------------------------------- stage 8: combined capture


def stage_combine(args, job: Path, pf: Placeframe, base: dict) -> None:
    recon = pf.json("GET", f"/reconstructions/{base['reconstruction_id']}", "Get base reconstruction")
    base_tar = job / "base_capture.tar"
    log(f"downloading the existing map's capture session {recon['capture_session_id']}")
    with pf.client.stream("GET", f"{pf.server}/capture_sessions/{recon['capture_session_id']}/tar") as r:
        if r.status_code >= 300:
            sys.exit(f"Download base capture failed ({r.status_code})")
        with base_tar.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)

    new_tar = job / "capture/capture.tar"
    out_dir = job / "merged"
    out_dir.mkdir(exist_ok=True)
    with tarfile.open(base_tar) as bt, tarfile.open(new_tar) as nt:
        base_manifest = json.load(bt.extractfile("manifest.json"))
        new_manifest = json.load(nt.extractfile("manifest.json"))
        if base_manifest["axis_convention"] != new_manifest["axis_convention"]:
            sys.exit("Axis conventions differ between the existing capture and the new one")
        taken = {r["id"] for r in base_manifest["rigs"]}
        new_rig = dict(new_manifest["rigs"][0])
        old_id = new_rig["id"]
        new_rig["id"] = next(f"rig{i}" for i in range(len(taken) + 1) if f"rig{i}" not in taken)
        manifest = {"axis_convention": base_manifest["axis_convention"], "rigs": base_manifest["rigs"] + [new_rig]}

        with tarfile.open(out_dir / "capture.tar", "w") as out:
            def add_bytes(name: str, data: bytes) -> None:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                ti.mtime = int(time.time())
                out.addfile(ti, io.BytesIO(data))

            add_bytes("manifest.json", json.dumps(manifest, indent=2).encode())
            n_base = 0
            for m in bt:
                if not m.isfile() or m.name == "manifest.json":
                    continue
                data = bt.extractfile(m).read()
                if m.name.endswith("/frames.csv"):
                    n_base += len(data.splitlines()) - 1
                add_bytes(m.name, data)
            add_bytes(f"{new_rig['id']}/frames.csv", (job / "aligned_frames.csv").read_bytes())
            for m in nt:
                if m.isfile() and m.name.startswith(old_id + "/") and not m.name.endswith("/frames.csv"):
                    add_bytes(new_rig["id"] + m.name[len(old_id):], nt.extractfile(m).read())
    n_new = len((job / "aligned_frames.csv").read_text().splitlines()) - 1
    (out_dir / "conversion.json").write_text(json.dumps({
        "tar": str(out_dir / "capture.tar"), "pose_source": "insta360 VIO, aligned to the existing map",
        "metric": True, "base_frames": n_base, "new_frames": n_new, "new_rig": new_rig["id"],
    }, indent=1))
    log(f"merged capture: {n_base} existing frames + {n_new} new frames as {new_rig['id']} "
        f"({(out_dir / 'capture.tar').stat().st_size / 1e6:.0f} MB)")
    base_tar.unlink()


# --------------------------------------------------------------------------- stage 9: reconstruct + publish


def stage_reconstruct(args, job: Path, pf: Placeframe, base: dict) -> None:
    state_path = job / "server.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    tar = job / "merged/capture.tar"
    name = args.name

    if "capture_session_id" not in state:
        log(f"uploading merged capture ({tar.stat().st_size / 1e6:.0f} MB) to {pf.server} as '{name}'")
        with tar.open("rb") as fh:
            session = pf.json("POST", "/capture_sessions", "Capture upload",
                              data={"device_type": "ARFoundation", "name": name},
                              files={"data": (tar.name, fh, "application/x-tar")})
        state["capture_session_id"] = session["id"]
        state_path.write_text(json.dumps(state))
    if "reconstruction_id" not in state:
        options, reasons = choose_reconstruction_options(tar, json.loads((job / "merged/conversion.json").read_text()))
        for r in reasons:
            log(f"  - {r}")
        if args.options:
            options |= json.loads(args.options)
        recon = pf.json("POST", "/reconstructions", "Create reconstruction",
                        json={"create": {"capture_session_id": state["capture_session_id"]}, "options": options})
        state["reconstruction_id"] = recon["id"]
        state_path.write_text(json.dumps(state))
        log(f"queued reconstruction {recon['id']} with {options}")

    last = None
    started = time.time()
    while True:
        r = pf.request("GET", f"/reconstructions/{state['reconstruction_id']}")
        if r.status_code >= 300:
            sys.exit(f"Poll reconstruction failed ({r.status_code}): {r.text[:300]}")
        rec = r.json()
        line = rec["status"]
        if rec.get("progress_total"):
            line += f" {rec.get('progress_current') or 0}/{rec['progress_total']}"
        if line != last:
            log(f"  [{int(time.time() - started):>6}s] {line}")
            last = line
        if rec["status"] in TERMINAL:
            break
        time.sleep(15)
    if rec["status"] != "succeeded":
        state.pop("reconstruction_id", None)  # a retry queues a fresh reconstruction of the same upload
        state_path.write_text(json.dumps(state))
        sys.exit(f"Reconstruction {rec['status']}: {rec.get('error') or 'see Grafana logs for reconstructor-cuda'}")

    if "localization_map_id" not in state:
        loc_map = pf.json("POST", "/localization-maps", "Create localization map", json={
            "reconstruction_id": state["reconstruction_id"], "name": name, "active": True, "color": 0,
            "position_x": 0, "position_y": 0, "position_z": 0,
            "rotation_x": 0, "rotation_y": 0, "rotation_z": 0, "rotation_w": 1,
        })
        state["localization_map_id"] = loc_map["id"]
        state_path.write_text(json.dumps(state))
    log(f"localization map {state['localization_map_id']} ('{name}') is live")


# --------------------------------------------------------------------------- stage 10: viewer data


def fetch_array(pf: Placeframe, path: str, widths: list[tuple[int, str]]) -> list[np.ndarray]:
    r = pf.request("GET", path, params={"axis_convention": "OPENCV"})
    if r.status_code >= 300:
        sys.exit(f"GET {path} failed ({r.status_code})")
    buf = r.content
    (n,) = struct.unpack("<I", buf[:4])
    out, off = [], 4
    for width, dtype in widths:
        size = n * width * np.dtype(dtype).itemsize
        out.append(np.frombuffer(buf[off:off + size], dtype).reshape(n, width))
        off += size
    return out


def write_viewer_map(pf: Placeframe, reconstruction_id: str, name: str, frame_interval_ms: int, dest: Path) -> dict:
    """map.json for viewer/index.html. Viewer axes: x, height (up), z — OpenCV world with y negated."""
    positions, colors = fetch_array(pf, f"/reconstructions/{reconstruction_id}/points", [(3, "<f4"), (3, "u1")])
    centres, _ = fetch_array(pf, f"/reconstructions/{reconstruction_id}/frame_poses", [(3, "<f4"), (4, "<f4")])
    pts = [[round(float(p[0]), 3), round(float(-p[1]), 3), round(float(p[2]), 3), int(c[0]), int(c[1]), int(c[2]), 3]
           for p, c in zip(positions, colors)]
    # The server returns registered frames in capture order (rig, then time) but not their timestamps,
    # so the replay slider runs on frame order.
    cams = [[round(float(c[0]), 3), round(float(-c[1]), 3), round(float(c[2]), 3), i * frame_interval_ms]
            for i, c in enumerate(centres)]
    data = {"name": name, "points": pts, "cams": cams, "camsOrig": cams, "gaps": [], "timeIsFrameOrder": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, separators=(",", ":")))
    return {"points": len(pts), "frames": len(cams)}


def stage_viewer(args, job: Path, pf: Placeframe, base: dict) -> None:
    state = json.loads((job / "server.json").read_text())
    root = Path(args.viewer_root).expanduser()
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in args.name)
    info = write_viewer_map(pf, state["reconstruction_id"], args.name, round(1000 / args.fps) * args.tar_step,
                            root / "maps" / safe / "map.json")
    (root / "current_map.json").write_text(json.dumps({
        "name": args.name, "viewer_map": f"maps/{safe}/map.json", "localization_map_id": state["localization_map_id"],
        "reconstruction_id": state["reconstruction_id"], "merged_from": base["name"], "video": args.video.name,
    }, indent=1))
    log(f"viewer map written: {info['points']} points, {info['frames']} frames -> maps/{safe}/map.json")
    if not args.keep_work:
        for d in ("work", "equirect", "views", "calib", "vio"):
            shutil.rmtree(job / d, ignore_errors=True)
        log("removed intermediate frames (the capture tars and logs are kept)")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="raw dual-fisheye .insv")
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--base-map", required=True, help="name or id of the localization map to extend")
    ap.add_argument("--name", default=None, help="name of the merged map (default: <base>+<video stem>)")
    ap.add_argument("--viewer-root", default="~/pf_x5/merge_ui", help="where maps/<name>/map.json is written")
    ap.add_argument("--server", default=None)
    ap.add_argument("--placeframe-dir", default=str(DEFAULT_PLACEFRAME_DIR))
    ap.add_argument("--realm", default="placeframe-dev")
    ap.add_argument("--options", default=None, help="extra reconstruction options as JSON")
    ap.add_argument("--fps", type=float, default=4.0, help="stitched frames per second")
    ap.add_argument("--tar-step", type=int, default=2, help="use every Nth stitched frame in the capture")
    ap.add_argument("--calib-seconds", type=float, default=180.0, help="video length used for lens calibration")
    ap.add_argument("--imu-offset-ms", type=float, default=220.0)
    ap.add_argument("--stitch-bottom-mask", type=float, default=0.55)
    ap.add_argument("--align-samples", type=int, default=60, help="frames localized against the existing map")
    ap.add_argument("--align-min-inliers", type=int, default=30, help="PnP inliers for a localization to count")
    ap.add_argument("--align-min-frames", type=int, default=4)
    ap.add_argument("--align-threshold", type=float, default=2.5, help="metres; VIO drifts, so this is loose")
    ap.add_argument("--no-drift-correction", action="store_true")
    ap.add_argument("--stop-after", choices=STAGES, default=None)
    ap.add_argument("--redo-from", choices=STAGES, default=None, help="forget this stage and every later one")
    ap.add_argument("--keep-work", action="store_true", help="keep extracted frames after success")
    args = ap.parse_args()

    args.video = Path(args.video).expanduser().resolve()
    if not args.video.exists():
        sys.exit(f"{args.video} not found")
    job = Path(args.job_dir).expanduser().resolve()
    done = job / ".done"
    done.mkdir(parents=True, exist_ok=True)
    if args.redo_from:
        for s in STAGES[STAGES.index(args.redo_from):]:
            (done / s).unlink(missing_ok=True)

    pf = None
    base = None
    local = {"calib": stage_calib, "frames": stage_frames, "stitch": stage_stitch, "views": stage_views,
             "vio": stage_vio, "capture": stage_capture}
    remote = {"align": stage_align, "combine": stage_combine, "reconstruct": stage_reconstruct, "viewer": stage_viewer}
    for i, stage in enumerate(STAGES, 1):
        if (done / stage).exists():
            log(f"### STAGE {i}/{len(STAGES)} {stage} (already done)")
        else:
            log(f"### STAGE {i}/{len(STAGES)} {stage}")
            t0 = time.time()
            if stage in local:
                local[stage](args, job)
            else:
                if pf is None:
                    pf = Placeframe(args)
                    base_path = job / "base_map.json"
                    if base_path.exists():  # a resumed job keeps extending the map it started with
                        base = json.loads(base_path.read_text())
                    else:
                        m = pf.find_map(args.base_map)
                        base = {"id": m["id"], "name": m.get("name"), "reconstruction_id": m["reconstruction_id"]}
                        base_path.write_text(json.dumps(base))
                    args.name = args.name or f"{base['name']}+{args.video.stem}"
                remote[stage](args, job, pf, base)
            (done / stage).write_text(f"{time.time() - t0:.0f}s\n")
            log(f"    {stage} took {time.time() - t0:.0f}s")
        if stage == args.stop_after:
            log(f"stopping after {stage} as requested")
            return
    log("### DONE")


if __name__ == "__main__":
    main()
