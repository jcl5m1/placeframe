#!/usr/bin/env python3
"""Build a Placeframe capture tar whose frames.csv holds Insta360 VIO poses — the slot ARCore/ARKit fill.

Inputs come from the dual-fisheye pipeline:
  * vio.csv                 — vio_dualfisheye.py (one pose per video frame; world z-up, metres)
  * stitched equirect views — insta360_to_placeframe.py work/images (6 pinhole views per frame)
  * stitch work dir         — stitch_dualfisheye.py inputs (frames/ + meta.json), to recover the exact
                              per-frame levelling rotation each equirect was stitched with

  uv run vio_to_capture.py --insv VID.insv --vio ~/cap/vio/vio.csv --stitch-work ~/cap/v2/work \\
      --equirect ~/cap/v2/equirect --views ~/cap/v2/out/work -o ~/cap/vio_capture --step 2
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from insta360_to_placeframe import RIG_ID, parse_views
from stitch_dualfisheye import down_in_lens0, imu_gravity, level_rotation
from vio_dualfisheye import detect_label_shift

# VIO world is z-up; Placeframe OPENCV worlds are y-down (phones' Unity y-up worlds are flipped the same way).
OPENCV_FROM_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insv", required=True)
    ap.add_argument("--vio", required=True, help="vio.csv (vio_dualfisheye.py) or .tum (e.g. OpenVIO IMU poses)")
    ap.add_argument("--imu-from-camera", default=None,
                    help="for .tum IMU poses: OpenVIO config (x5.yaml) whose rig.imu_from_camera maps lens0 -> IMU")
    ap.add_argument("--cam-from-imu", default=None,
                    help="for .tum IMU poses: OpenVINS kalibr_imucam_chain.yaml (cam0 T_cam_imu maps IMU -> lens0)")
    ap.add_argument("--stitch-work", required=True, help="work dir passed to stitch_dualfisheye.py --work")
    ap.add_argument("--equirect", required=True, help="stitch output dir (stitch.json)")
    ap.add_argument("--views", required=True, help="insta360_to_placeframe.py work dir (images/ + frames.json)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--step", type=int, default=2, help="use every Nth stitched frame")
    ap.add_argument("--skip-coast", action="store_true", help="drop frames where VIO was coasting")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.stitch_work).expanduser()
    views_work = Path(args.views).expanduser()
    stitch = json.loads((Path(args.equirect).expanduser() / "stitch.json").read_text())
    meta = json.loads((work / "meta.json").read_text())
    fr = json.loads((views_work / "frames.json").read_text())
    intr, kept = fr["intrinsics"], fr["kept"]

    t_vid, P, Rv, coast, source = load_trajectory(Path(args.vio).expanduser(), args.imu_from_camera, args.cam_from_imu)
    slerp = Slerp(t_vid, Rv)

    fps = 30000 / 1001
    shift = detect_label_shift(args.insv, work / "frames", fps, meta["ts"])
    print(f"stitched-frame labels lag video time by {shift * 1000:.0f} ms")

    # Exactly the levelling the stitcher applied (same IMU fusion, offset, sign).
    t_imu, up = imu_gravity(Path(args.insv), stitch["imu_offset_ms"], 1.5, stitch["gyro_sign"])
    labels = sorted(kept)[:: max(1, args.step)]
    downs = down_in_lens0(t_imu, up, labels)

    views = parse_views(None, 6, 0.0)
    rows = ["timestamp,tx,ty,tz,qx,qy,qz,qw"]
    used = []
    for lab in labels:
        tv = lab / 1000 + shift
        if tv < t_vid[0] or tv > t_vid[-1]:
            continue
        k = int(np.clip(np.searchsorted(t_vid, tv), 1, len(t_vid) - 1))
        if args.skip_coast and coast[k]:
            continue
        pos = np.array([np.interp(tv, t_vid, P[:, j]) for j in range(3)])
        R_world_lens0 = slerp(tv).as_matrix()
        R_world_rig = OPENCV_FROM_ZUP @ R_world_lens0 @ level_rotation(downs[lab])  # levelled equirect camera0
        # The equirect is level by construction, so keep only VIO heading: rotation about world y (down).
        fwd = R_world_rig[:, 2]
        yaw = np.arctan2(fwd[0], fwd[2])
        Rw = Rotation.from_rotvec([0.0, yaw, 0.0]).as_matrix()
        pw = OPENCV_FROM_ZUP @ pos
        q = Rotation.from_matrix(Rw).as_quat()
        rows.append(f"{lab},{pw[0]:.6f},{pw[1]:.6f},{pw[2]:.6f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")
        used.append(lab)

    manifest = {
        "axis_convention": "OPENCV",
        "rigs": [{
            "id": RIG_ID,
            "cameras": [
                {
                    "id": vw.name, "ref_sensor": i == 0,
                    "rotation": dict(zip("xyzw", Rotation.from_matrix(vw.rig_from_cam().T).as_quat().tolist()
                                         if i else [0.0, 0.0, 0.0, 1.0])),
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "camera_config": {"width": intr["width"], "height": intr["height"], "orientation": "TOP_LEFT",
                                      "fx": intr["fx"], "fy": intr["fy"], "cx": intr["cx"], "cy": intr["cy"]},
                }
                for i, vw in enumerate(views)
            ],
        }],
    }

    tar_path = out / "capture.tar"
    with tarfile.open(tar_path, "w") as tar:
        def add_bytes(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = int(time.time())
            tar.addfile(ti, io.BytesIO(data))

        add_bytes("manifest.json", json.dumps(manifest, indent=2).encode())
        add_bytes(f"{RIG_ID}/frames.csv", ("\n".join(rows) + "\n").encode())
        for lab in used:
            for vw in views:
                tar.add(views_work / "images" / RIG_ID / vw.name / f"{lab}.jpg", arcname=f"{RIG_ID}/{vw.name}/{lab}.jpg")

    centers = np.array([[float(x) for x in r.split(",")[1:4]] for r in rows[1:]])
    summary = {"tar": str(tar_path), "frames": len(used), "views_per_frame": len(views), "pose_source": source,
               "label_shift_ms": shift * 1000, "path_length_m": float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))),
               "metric": True}
    (out / "conversion.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def load_trajectory(path: Path, imu_from_camera_yaml: str | None, cam_from_imu_yaml: str | None = None):
    """Returns video times, world positions (z up), world_from_lens0 rotations, coast flags, source label.

    .tum rows are world_from_imu poses (OpenVIO, or OpenVINS converted to Hamilton world_from_imu), so
    world_from_lens0 = R_world_imu @ R_imu_lens0 with R_imu_lens0 = imu_from_camera (OpenVIO) or T_cam_imu^T
    of cam0 (OpenVINS / kalibr)."""
    if path.suffix == ".tum":
        d = np.loadtxt(path)
        R_imu_cam, source = np.eye(3), "insta360 VIO (OpenVIO)"
        if imu_from_camera_yaml:
            R_imu_cam = first_matrix(imu_from_camera_yaml)
        elif cam_from_imu_yaml:
            R_imu_cam, source = first_matrix(cam_from_imu_yaml).T, "insta360 VIO (OpenVINS)"
        R_w_cam = Rotation.from_quat(d[:, 4:8]).as_matrix() @ R_imu_cam
        return d[:, 0], d[:, 1:4], Rotation.from_matrix(R_w_cam), np.zeros(len(d), bool), source
    v = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    P = np.stack([v["x"], v["y"], v["z"]], 1)
    Rv = Rotation.from_quat(np.stack([v["qx"], v["qy"], v["qz"], v["qw"]], 1))
    return v["t_s"], P, Rv, v["status"] == "coast", "insta360 VIO"


def first_matrix(yaml_path: str) -> np.ndarray:
    """Rotation block of the first 4x4 matrix written as '- [a, b, c, d]' rows (cam0 / imu_from_camera)."""
    rows = re.findall(r"^\s+- \[(.*)\]$", Path(yaml_path).expanduser().read_text(), re.M)[:3]
    return np.array([[float(x) for x in r.split(",")][:3] for r in rows])


if __name__ == "__main__":
    main()
