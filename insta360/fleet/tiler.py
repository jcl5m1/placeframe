#!/usr/bin/env python3
"""Split a posed capture into overlapping tiles that can be built as independent jobs.

Frames are partitioned by recursive bisection of their horizontal positions (always splitting the longer
side), so tiles are balanced in size even when the walk is uneven. Each tile = its core frames plus every
frame within --overlap metres of the core's footprint, so views near a border are fully contained in some
tile. All tiles keep the capture's world frame (the priors), so their maps line up without a merge step and
the localizer can be given the set of tile maps around a client.

  python fleet/tiler.py capture.tar out_dir --images-per-tile 750 --overlap 3
  -> out_dir/tile_00/capture.tar ... and out_dir/tiles.json
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--images-per-tile", type=int, default=750, help="target core size in images")
    ap.add_argument("--overlap", type=float, default=3.0, help="metres of shared frames around each core")
    a = ap.parse_args()
    plan = make_tiles(a.capture, a.out, a.images_per_tile, a.overlap)
    print(json.dumps({k: v for k, v in plan.items() if k != "tiles"}, indent=1))
    for t in plan["tiles"]:
        print(f"  {t['name']}: core {t['core_frames']} frames, with overlap {t['frames']} frames "
              f"({t['images']} images), {t['extent_m'][0]:.1f} x {t['extent_m'][1]:.1f} m")


def make_tiles(capture: Path, out: Path, images_per_tile: int, overlap: float) -> dict:
    with tarfile.open(capture) as tf:
        manifest = json.load(tf.extractfile("manifest.json"))
        frames = []  # (rig, row, x, z)
        for rig in manifest["rigs"]:
            rows = tf.extractfile(f"{rig['id']}/frames.csv").read().decode().splitlines()
            for row in rows[1:]:
                f = row.split(",")
                frames.append((rig["id"], row, float(f[1]), float(f[3])))
        header = {rig["id"]: tf.extractfile(f"{rig['id']}/frames.csv").read().decode().splitlines()[0]
                  for rig in manifest["rigs"]}
    cams = {rig["id"]: len(rig["cameras"]) for rig in manifest["rigs"]}
    xz = np.array([(f[2], f[3]) for f in frames])
    frames_per_tile = max(8, images_per_tile // max(cams.values()))
    cores = _bisect(np.arange(len(frames)), xz, frames_per_tile)

    tiles = []
    out.mkdir(parents=True, exist_ok=True)
    for i, core in enumerate(cores):
        # Overlap = frames within `overlap` metres of an actual core frame (not of the core's bounding box,
        # which pulls in far more frames on walks that keep revisiting a central hall).
        members = np.unique(np.concatenate([core, *cKDTree(xz).query_ball_point(xz[core], r=overlap)]).astype(int))
        name = f"tile_{i:02d}"
        tar_path = out / name / "capture.tar"
        _write_tile(capture, tar_path, manifest, header, [frames[j] for j in members])
        tiles.append({"name": name, "capture": str(tar_path), "core_frames": len(core), "frames": len(members),
                      "images": int(sum(cams[frames[j][0]] for j in members)),
                      "core_bbox_xz": [*xz[core].min(0).round(2).tolist(), *xz[core].max(0).round(2).tolist()],
                      "extent_m": (xz[core].max(0) - xz[core].min(0)).round(1).tolist()})
    plan = {"source": str(capture), "frames": len(frames), "images": int(sum(cams[f[0]] for f in frames)),
            "images_per_tile": images_per_tile, "overlap_m": overlap, "tile_count": len(tiles),
            "duplication": round(sum(t["frames"] for t in tiles) / len(frames), 2), "tiles": tiles}
    (out / "tiles.json").write_text(json.dumps(plan, indent=1))
    return plan


def _bisect(idx: np.ndarray, xz: np.ndarray, max_frames: int) -> list[np.ndarray]:
    if len(idx) <= max_frames:
        return [idx]
    pts = xz[idx]
    axis = int(np.argmax(pts.max(0) - pts.min(0)))
    order = idx[np.argsort(pts[:, axis], kind="stable")]
    half = len(order) // 2
    return _bisect(order[:half], xz, max_frames) + _bisect(order[half:], xz, max_frames)


def _write_tile(capture: Path, tar_path: Path, manifest: dict, header: dict, members: list) -> None:
    keep = {(rig, row.split(",")[0]) for rig, row, _, _ in members}
    rigs_used = sorted({rig for rig, _ in keep})
    tile_manifest = dict(manifest, rigs=[r for r in manifest["rigs"] if r["id"] in rigs_used])
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_suffix(".part")
    with tarfile.open(capture) as src, tarfile.open(tmp, "w") as out:
        def add(name: str, data: bytes) -> None:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime = len(data), int(time.time())
            out.addfile(ti, io.BytesIO(data))

        add("manifest.json", json.dumps(tile_manifest, indent=1).encode())
        for rig in rigs_used:
            rows = [row for r, row, _, _ in members if r == rig]
            add(f"{rig}/frames.csv", ("\n".join([header[rig]] + rows) + "\n").encode())
        for m in src:
            if not m.isfile() or not m.name.endswith(".jpg"):
                continue
            rig, _, stem = m.name.split("/")
            if (rig, stem.rsplit(".", 1)[0]) in keep:
                out.addfile(m, src.extractfile(m))
    tmp.rename(tar_path)


if __name__ == "__main__":
    main()
