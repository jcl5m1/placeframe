#!/usr/bin/env python3
"""Convert a Microsoft 7-Scenes scene into a Placeframe capture (mapping split) and a query list (test split).

7-Scenes: seq-XX/frame-NNNNNN.color.png + .pose.txt (4x4 camera-to-world, metres, OpenCV-style camera axes),
640x480 RGB, TrainSplit.txt / TestSplit.txt list the sequences. RGB intrinsics follow the common choice in the
localization literature (fx = fy = 525, cx = 320, cy = 240).

  python datasets/sevenscenes_to_placeframe.py <scene_dir> <out_dir> [--step 5] [--query-step 10]
  -> out_dir/capture.tar   (train frames every --step, ground-truth poses as priors)
     out_dir/queries.jsonl (test frames every --query-step: image path + ground-truth camera-to-world)
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CAMERA = {"width": 640, "height": 480, "orientation": "TOP_LEFT", "fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0}
FRAME_MS = 33  # 30 fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--query-step", type=int, default=10)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train, test = _split(a.scene, "TrainSplit.txt"), _split(a.scene, "TestSplit.txt")
    n_map = write_capture(a.scene, train, a.out / "capture.tar", a.step)
    n_q = write_queries(a.scene, test, a.out, a.query_step)
    (a.out / "conversion.json").write_text(json.dumps({"pose_source": "7-Scenes ground truth (KinectFusion)",
                                                       "metric": True, "camera": CAMERA}))
    print(f"mapping frames {n_map} -> {a.out / 'capture.tar'}; queries {n_q} -> {a.out / 'queries.jsonl'}")


def write_capture(scene: Path, seqs: list[str], tar_path: Path, step: int) -> int:
    rows, n = [], 0
    tmp = tar_path.with_suffix(".part")
    with tarfile.open(tmp, "w") as out:
        for s_idx, seq in enumerate(seqs):
            for idx, png, pose in _frames(scene, seq)[::step]:
                ts = (s_idx * 100000 + idx) * FRAME_MS  # unique, increasing across sequences
                t, q = pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()
                rows.append(f"{ts},{t[0]:.6f},{t[1]:.6f},{t[2]:.6f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")
                _add(out, f"rig0/camera0/{ts}.jpg", _jpeg(png))
                n += 1
        manifest = {"axis_convention": "OPENCV", "rigs": [{"id": "rig0", "cameras": [
            {"id": "camera0", "ref_sensor": True, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
             "translation": {"x": 0.0, "y": 0.0, "z": 0.0}, "camera_config": CAMERA}]}]}
        _add(out, "manifest.json", json.dumps(manifest, indent=1).encode())
        _add(out, "rig0/frames.csv", ("\n".join(["timestamp,tx,ty,tz,qx,qy,qz,qw"] + rows) + "\n").encode())
    tmp.rename(tar_path)
    return n


def write_queries(scene: Path, seqs: list[str], out: Path, step: int) -> int:
    qdir = out / "queries"
    qdir.mkdir(exist_ok=True)
    n = 0
    with (out / "queries.jsonl").open("w") as f:
        for seq in seqs:
            for idx, png, pose in _frames(scene, seq)[::step]:
                path = qdir / f"{seq}-{idx:06d}.jpg"
                path.write_bytes(_jpeg(png))
                f.write(json.dumps({"image": str(path), "world_from_camera": pose.tolist()}) + "\n")
                n += 1
    return n


def _split(scene: Path, name: str) -> list[str]:
    lines = [ln.strip() for ln in (scene / name).read_text().splitlines() if ln.strip()]
    return [f"seq-{int(ln.lower().replace('sequence', '')):02d}" for ln in lines]


def _frames(scene: Path, seq: str) -> list[tuple[int, bytes, np.ndarray]]:
    """(index, png bytes, 4x4 camera-to-world) for one sequence, read from seq-XX.zip or an unpacked folder."""
    folder, archive = scene / seq, scene / f"{seq}.zip"
    out = []
    if folder.exists():
        for color in sorted(folder.glob("frame-*.color.png")):
            idx = int(color.name.split("-")[1].split(".")[0])
            pose = np.loadtxt(color.with_name(color.name.replace("color.png", "pose.txt")))
            out.append((idx, color.read_bytes(), pose))
        return out
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for name in sorted(n for n in names if n.endswith(".color.png")):
            idx = int(Path(name).name.split("-")[1].split(".")[0])
            pose = np.loadtxt(io.BytesIO(z.read(name.replace("color.png", "pose.txt"))))
            out.append((idx, z.read(name), pose))
    return out


def _jpeg(png: bytes) -> bytes:
    buf = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _add(out: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime = len(data), int(time.time())
    out.addfile(ti, io.BytesIO(data))


if __name__ == "__main__":
    main()
