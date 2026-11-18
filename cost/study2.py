#!/usr/bin/env python3
"""Compute-cost study, v2 (current pipeline: fast prior-pose mapper, GPU bundle adjustment, panoramic pairing):
build size variants of the X5 house capture, reconstruct each on the Placeframe server one at a time (only when
the server and the map updater are idle), and record per-stage wall time + size metrics. Restart-safe: finished
variants are skipped (results_v2/<variant>.json).

Variants (all from the VIO-posed capture, 378 frames x 6 views, 2 fps):
  f25/f50/f100  first 25/50/100% of the walk
  j2/j4/j8      full walk tiled as one map: copies side by side (1x2, 2x2, 4x2 grid, 2 m apart) in one
                reconstruction, so retrieval, pairing and one bundle adjustment run over the whole joined area
                (j8, ~18,000 images, is about the projected venue size; run on its own: study2.py j8)
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "github/placeframe-insta360"))
from add_recording_to_map import PlaceframeApi  # noqa: E402
from upload_to_placeframe import choose_reconstruction_options  # noqa: E402

BASE = Path.home() / "pf_x5/vio_capture/capture.tar"
ROOT = Path.home() / "pf_cost"
VAR, RES = ROOT / "variants", ROOT / "results_v2"
ORDER = ["f25", "f50", "f100", "j2", "j4"]
# Tile offsets (m, floor plane x/z) for joined copies: the walk spans 25.4 x 15.5 m; 2 m gaps.
TILES = {"j2": [(0, 0), (27.4, 0)], "j4": [(0, 0), (27.4, 0), (0, 17.5), (27.4, 17.5)],
         "j8": [(x, z) for z in (0, 17.5) for x in (0, 27.4, 54.8, 82.2)]}
TERMINAL = {"succeeded", "failed", "cancelled"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_variant(name: str) -> Path:
    out = VAR / name
    tar_path = out / "capture.tar"
    if tar_path.exists():
        return tar_path
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BASE) as tf:
        manifest = json.load(tf.extractfile("manifest.json"))
        rows = tf.extractfile("rig0/frames.csv").read().decode().splitlines()
        members = {m.name: m for m in tf if m.isfile()}
        header, frames = rows[0], rows[1:]
        if name.startswith("f"):
            frac = int(name[1:]) / 100
            frames = frames[: max(8, int(round(len(frames) * frac)))]
            copies = 1
        else:
            copies = len(TILES[name])
        keep_ts = {r.split(",")[0] for r in frames}
        rig0 = manifest["rigs"][0]
        manifest["rigs"] = [rig0 | {"id": f"rig{c}"} for c in range(copies)]
        tmp = tar_path.with_suffix(".part")
        with tarfile.open(tmp, "w") as out_tf:
            def add(nm, data):
                ti = tarfile.TarInfo(nm)
                ti.size = len(data)
                ti.mtime = int(time.time())
                out_tf.addfile(ti, io.BytesIO(data))
            add("manifest.json", json.dumps(manifest, indent=2).encode())
            for c in range(copies):
                body = []
                for r in frames:
                    f = r.split(",")
                    dx, dz = TILES[name][c] if name in TILES else (0.0, 0.0)
                    body.append(",".join([f[0], f"{float(f[1]) + dx:.6f}", f[2], f"{float(f[3]) + dz:.6f}"] + f[4:]))
                add(f"rig{c}/frames.csv", ("\n".join([header] + body) + "\n").encode())
                for nm, m in members.items():
                    if not nm.startswith("rig0/") or nm.endswith("frames.csv"):
                        continue
                    if nm.rsplit("/", 1)[1].rsplit(".", 1)[0] not in keep_ts:
                        continue
                    add(nm.replace("rig0/", f"rig{c}/", 1), tf.extractfile(m).read())
        tmp.rename(tar_path)
    span = (int(frames[-1].split(",")[0]) - int(frames[0].split(",")[0])) / 1000
    meta = {"variant": name, "frames_per_copy": len(frames), "copies": copies, "frames": len(frames) * copies,
            "images": len(frames) * copies * len(rig0["cameras"]), "recording_s": span * copies,
            "tar_MB": round(tar_path.stat().st_size / 1e6, 1)}
    (out / "variant.json").write_text(json.dumps(meta, indent=2))
    (out / "conversion.json").write_text(json.dumps({"pose_source": "insta360 VIO", "metric": True}))
    log(f"built {name}: {meta}")
    return tar_path


def server_idle(api: PlaceframeApi) -> bool:
    busy = [r for r in api.get("/reconstructions") if r["status"] not in TERMINAL]
    jobs = []
    for p in (Path.home() / "pf_mapper/jobs").glob("*/job.json"):
        try:
            jobs.append(json.loads(p.read_text()))
        except ValueError:
            return False  # a job file is mid-write; treat as busy and check again later
    running = [j for j in jobs if j.get("state") in ("running", "queued")]
    return not busy and not running


def minio(cmd: str) -> str:
    return subprocess.run(["sg", "docker", "-c", f"docker exec placeframe-minio-1 sh -c '{cmd}'"],
                          capture_output=True, text=True).stdout


def mapped_area(recon_id: str) -> dict:
    raw = subprocess.run(["sg", "docker", "-c", f"docker exec placeframe-minio-1 mc cat l/dev-reconstructions/{recon_id}/sfm_model/points3D.npz"],
                         capture_output=True).stdout
    try:
        P = np.load(io.BytesIO(raw))["positions"].astype(float)
    except Exception:
        return {}
    # Placeframe worlds here are y-down (gravity): floor plan = x, z. Count 0.5 m cells with >= 3 points
    # within the band from 0.3 m below the median camera height to 1.8 m below (walls/furniture/floor).
    xz = P[:, [0, 2]]
    cells = np.floor(xz / 0.5).astype(int)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    occ = int((counts >= 3).sum())
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(xz).volume
    except Exception:
        hull = 0.0
    return {"occupied_area_m2": occ * 0.25, "hull_area_m2": round(hull, 1), "points_in_npz": len(P)}


def run_variant(api: PlaceframeApi, name: str) -> None:
    res_path = RES / f"{name}.json"
    if res_path.exists():
        return
    tar = build_variant(name)
    while not server_idle(api):
        time.sleep(60)
    meta = json.loads((VAR / name / "variant.json").read_text())
    conv = json.loads((VAR / name / "conversion.json").read_text())
    options, _ = choose_reconstruction_options(tar, conv)
    t_up = time.time()
    cs = api.upload_capture(tar, f"cost-study-v2-{name}")
    upload_s = time.time() - t_up
    rid = api.post("/reconstructions", {"create": {"capture_session_id": cs}, "options": options})["id"]
    log(f"{name}: capture {cs}, reconstruction {rid}, upload {upload_s:.0f}s")
    t0 = time.time()
    stages, last = [], None
    while True:
        try:
            st = api.get(f"/reconstructions/{rid}")["status"]
        except Exception as e:
            log(f"poll error {e}")
            time.sleep(10)
            continue
        now = time.time()
        if st != last:
            stages.append({"status": st, "t": round(now - t0, 1), "wall": round(now, 1)})
            log(f"{name}: {st} at {now - t0:.0f}s")
            last = st
        if st in TERMINAL:
            break
        time.sleep(5)
    rec = api.get(f"/reconstructions/{rid}")
    metrics = (rec.get("manifest") or {}).get("metrics") or {}
    pairs = minio(f"mc cat l/dev-reconstructions/{rid}/pairs.txt | wc -l").strip()
    du = minio(f"mc du l/dev-reconstructions/{rid}/").strip()
    out = meta | {"capture_session_id": cs, "reconstruction_id": rid, "status": last, "options": options,
                  "upload_s": round(upload_s, 1), "total_s": stages[-1]["t"], "stages": stages,
                  "pairs": int(pairs) if pairs.isdigit() else None, "recon_storage": du,
                  "metrics": {k: metrics.get(k) for k in ("total_images", "registered_images", "num_3d_points",
                              "average_keypoints_per_image", "all_verified_matches", "map_avg_track_length",
                              "reprojection_pixel_error_50th_percentile", "truth_alignment_rms_residual_m")}}
    out |= mapped_area(rid)
    res_path.write_text(json.dumps(out, indent=2))
    log(f"{name}: done {last} in {out['total_s']:.0f}s")


def main():
    VAR.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    api = PlaceframeApi()
    for name in sys.argv[1:] or ORDER:
        try:
            run_variant(api, name)
        except Exception as e:
            log(f"{name}: ERROR {e!r}")
    log("study complete")


if __name__ == "__main__":
    main()
