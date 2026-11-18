#!/usr/bin/env python3
"""Export a raw Insta360 dual-fisheye .insv for OpenVINS (ROS1 bag + kalibr-style configs).

OpenVINS (github.com/rpng/open_vins) has no 360 model: the two lenses are exported as two independent
equidistant-fisheye cameras (no stereo overlap). OpenVINS projects through the z>0 plane, so each lens is
masked beyond --max-theta from its axis; the stick/hand region at the bottom of each lens is masked too.

Uses the same calibration as insv_to_openvio.py (lens models + IMU mount/time offset/gyro scale from the
dual-fisheye SfM) so the two estimators get identical inputs.

  uv run insv_to_openvins.py VID.insv --calib ~/cap/fisheye/work -o ~/cap/openvins
  -> x5.bag, config/{estimator_config.yaml,kalibr_imu_chain.yaml,kalibr_imucam_chain.yaml,mask0.png,mask1.png}
  --config-only rewrites config/ (masks, extrinsics, noise) without re-encoding the bag.

Lens-to-lens extrinsic: rotation from the dual-fisheye SfM rig; translation direction from the same rig, scaled
to --lens-baseline metres (the SfM is unscaled; aligned to the metric house map its t10 is ~0.0255 m). The IMU
origin is placed midway between the two optical centres.
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
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from insta360_dualfisheye_to_placeframe import LENSES, LensGeom, lens1_from_rig, operator_mask
from insta360_to_placeframe import pycolmap
from vio_dualfisheye import calibrate_imu, detect_label_shift

IMU_NOISE = {  # same consumer-handheld values that kept OpenVIO stable on the X5 walk
    "accelerometer_noise_density": 2.94e-03, "accelerometer_random_walk": 9.21e-04,
    "gyroscope_noise_density": 4.144e-04, "gyroscope_random_walk": 3.92e-04,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv")
    ap.add_argument("--calib", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--size", type=int, default=640, help="per-lens image size in the bag")
    ap.add_argument("--max-theta", type=float, default=85.0)
    ap.add_argument("--bottom-mask", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--lens-baseline", type=float, default=0.025, help="lens0-lens1 optical centre distance [m]")
    ap.add_argument("--imu-noise-scale", type=float, default=1.0, help="multiplier on IMU_NOISE")
    ap.add_argument("--fixed-intrinsics", action="store_true", help="calib_cam_intrinsics false (SfM-calibrated)")
    ap.add_argument("--config-only", action="store_true", help="write config/ only, keep the existing bag")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    cfg = out / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    work = Path(args.calib).expanduser()
    meta = json.loads((work / "meta.json").read_text())
    geoms = [LensGeom(**g) for g in meta["geoms"]]
    rec = pycolmap.Reconstruction(str(work / "sfm"))
    cams = {}
    for im in rec.images.values():
        cams.setdefault(im.name.split("/")[0], rec.cameras[im.camera_id])
    params = [np.asarray(cams[l].params, float) for l in LENSES]
    R10, t10 = lens1_from_rig(rec)
    t10 = t10 / np.linalg.norm(t10) * args.lens_baseline if args.lens_baseline > 0 else np.zeros(3)

    off_ms, lens0_from_imu, gscale = calibrate_imu(Path(args.insv), rec, 220.0)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=r_frame_rate,nb_frames", "-of", "json", args.insv],
                                      capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = map(int, probe["r_frame_rate"].split("/"))
    fps = num / den
    nframes = int(probe["nb_frames"]) if not args.max_frames else args.max_frames
    shift = detect_label_shift(args.insv, work / "frames", fps, meta["ts"])

    S, F = args.size, 1920
    s = S / F
    intr = []
    for p in params:
        fx, fy, cx, cy = p[:4]
        # COLMAP pixel centres are at +0.5; OpenCV/OpenVINS at 0. Scale about pixel edges.
        intr.append([fx * s, fy * s, (cx - 0.5 + 0.5) * s - 0.5, (cy - 0.5 + 0.5) * s - 0.5])
    masks = []
    for L, (p, g) in enumerate(zip(params, geoms)):
        yy, xx = np.mgrid[0:F, 0:F]
        r = np.hypot((xx + 0.5 - p[2]) / p[0], (yy + 0.5 - p[3]) / p[1])  # theta_d
        th = r.copy()
        k1, k2, k3, k4 = p[4:]
        for _ in range(8):
            t2 = th * th
            th -= (th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - r) / (
                1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4))))
        keep = (th < math.radians(args.max_theta)) & (operator_mask((F, F), g, args.bottom_mask) > 0)
        m = np.where(keep, 0, 255).astype(np.uint8)  # OpenVINS: >127 = ignore
        m = cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(cfg / f"mask{L}.png"), m)
        masks.append(m)
        print(f"  lens{L}: usable {100 * (m < 128).mean():.0f}% of the image")

    # Camera-from-IMU: lens0 from the IMU calibration, lens1 = (lens1_from_lens0) @ lens0_from_imu.
    # IMU origin midway between the optical centres: in lens0 coordinates at c1 / 2, c1 = -R10^T t10.
    p_imu_in_0 = -R10.T @ t10 / 2
    T = []
    for R, t in ((lens0_from_imu, p_imu_in_0), (R10 @ lens0_from_imu, R10 @ p_imu_in_0 + t10)):
        T.append([list(map(float, R[i])) + [float(t[i])] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]])
    noise = {k: v * args.imu_noise_scale for k, v in IMU_NOISE.items()}
    write_configs(cfg, intr, params, T, S, nframes / fps, noise, not args.fixed_intrinsics)
    if args.config_only:
        print(json.dumps({"config": str(cfg), "lens_baseline_m": args.lens_baseline,
                          "imu_noise_scale": args.imu_noise_scale, "max_theta": args.max_theta}, indent=2))
        return

    store = get_typestore(Stores.ROS1_NOETIC)
    Imu, Image = store.types["sensor_msgs/msg/Imu"], store.types["sensor_msgs/msg/Image"]
    Header, Time = store.types["std_msgs/msg/Header"], store.types["builtin_interfaces/msg/Time"]
    Vec3, Quat = store.types["geometry_msgs/msg/Vector3"], store.types["geometry_msgs/msg/Quaternion"]
    T0 = 1_000_000_000  # ROS time must be positive; offset everything by 1000 s

    def stamp(tsec):
        ns = T0 + int(round(tsec * 1e9))
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000), ns

    imu = telemetry_parser.Parser(args.insv).normalized_imu()
    t = np.array([x["timestamp_ms"] for x in imu]) / 1000 - off_ms / 1000 + shift
    acc = np.array([x["accl"] for x in imu], float)
    gyr = np.radians(np.array([x["gyro"] for x in imu], float)) * gscale
    keep = np.concatenate([[True], np.diff(t) > 0]) & (t >= 0) & (t <= nframes / fps + 0.5)
    zero_cov = np.zeros(9)

    bag = out / "x5.bag"
    bag.unlink(missing_ok=True)
    with Writer(bag) as w:
        c_imu = w.add_connection("/imu0", Imu.__msgtype__, typestore=store)
        c_cam = [w.add_connection(f"/cam{L}/image_raw", Image.__msgtype__, typestore=store) for L in range(2)]
        imu_rows = list(zip(t[keep], acc[keep], gyr[keep]))
        ii = 0

        def flush_imu(until):
            nonlocal ii
            while ii < len(imu_rows) and imu_rows[ii][0] <= until:
                ti, a, g = imu_rows[ii]
                st, ns = stamp(ti)
                msg = Imu(header=Header(seq=ii, stamp=st, frame_id="imu0"), orientation=Quat(x=0, y=0, z=0, w=1),
                          orientation_covariance=zero_cov, angular_velocity=Vec3(x=g[0], y=g[1], z=g[2]),
                          angular_velocity_covariance=zero_cov, linear_acceleration=Vec3(x=a[0], y=a[1], z=a[2]),
                          linear_acceleration_covariance=zero_cov)
                w.write(c_imu, ns, store.serialize_ros1(msg, Imu.__msgtype__))
                ii += 1

        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", args.insv, "-filter_complex",
                                 f"[0:v:0]scale={S}:{S},format=gray[a];[0:v:1]scale={S}:{S},format=gray[b];[a][b]hstack",
                                 "-frames:v", str(nframes), "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
        for k in range(nframes):
            buf = proc.stdout.read(S * 2 * S)
            if len(buf) < S * 2 * S:
                break
            img = np.frombuffer(buf, np.uint8).reshape(S, 2 * S)
            tk = k / fps
            flush_imu(tk)
            st, ns = stamp(tk)
            for L in range(2):
                data = np.ascontiguousarray(img[:, L * S:(L + 1) * S]).reshape(-1)
                msg = Image(header=Header(seq=k, stamp=st, frame_id=f"cam{L}"), height=S, width=S, encoding="mono8",
                            is_bigendian=0, step=S, data=data)
                w.write(c_cam[L], ns, store.serialize_ros1(msg, Image.__msgtype__))
            if k % 1000 == 0:
                print(f"  bag: frame {k}/{nframes}")
        flush_imu(1e9)
        proc.kill()
    print(json.dumps({"bag": str(bag), "bag_GB": round(bag.stat().st_size / 1e9, 2), "frames": nframes,
                      "config": str(cfg), "time_zero_offset_s": T0 / 1e9}, indent=2))


def write_configs(cfg: Path, intr, params, T, S: int, duration: float, imu_noise: dict,
                  calib_intrinsics: bool) -> None:
    def mat(m):
        return "\n".join(f"    - [{', '.join(f'{v:.10f}' for v in row)}]" for row in m)

    cams = ""
    for L in range(2):
        cams += f"""
cam{L}:
  T_cam_imu:
{mat(T[L])}
  cam_overlaps: []
  camera_model: pinhole
  distortion_coeffs: [{', '.join(f'{v:.10f}' for v in params[L][4:])}]
  distortion_model: equidistant
  intrinsics: [{', '.join(f'{v:.6f}' for v in intr[L])}]
  resolution: [{S}, {S}]
  rostopic: /cam{L}/image_raw
  timeshift_cam_imu: 0.0
"""
    (cfg / "kalibr_imucam_chain.yaml").write_text("%YAML:1.0\n" + cams)
    noise = "\n".join(f"  {k}: {v:.6g}" for k, v in imu_noise.items())
    (cfg / "kalibr_imu_chain.yaml").write_text(f"""%YAML:1.0
imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
{noise}
  rostopic: /imu0
  time_offset: 0.0
  update_rate: 1000.0
  model: "kalibr"
  Tw: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  R_IMUtoGYRO: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  Ta: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  R_IMUtoACC: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  Tg: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
""")
    (cfg / "estimator_config.yaml").write_text(f"""%YAML:1.0
verbosity: "INFO"
use_fej: true
integration: "rk4"
use_stereo: false
max_cameras: 2
calib_cam_extrinsics: true
calib_cam_intrinsics: {str(calib_intrinsics).lower()}
calib_cam_timeoffset: true
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false
max_clones: 11
max_slam: 50
max_slam_in_update: 25
max_msckf_in_update: 40
dt_slam_delay: 1
gravity_mag: 9.81
feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"
try_zupt: false
zupt_chi2_multipler: 0
zupt_max_velocity: 0.1
zupt_noise_multiplier: 10
zupt_max_disparity: 0.5
zupt_only_at_beginning: false
init_window_time: 2.0
init_imu_thresh: 1.5
init_max_disparity: 10.0
init_max_features: 50
init_dyn_use: true
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_dyn_mle_max_threads: 6
init_dyn_num_pose: 6
init_dyn_min_deg: 10.0
init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-15
init_dyn_bias_g: [0.0, 0.0, 0.0]
init_dyn_bias_a: [0.0, 0.0, 0.0]
record_timing_information: false
record_timing_filepath: "/tmp/traj_timing.txt"
save_total_state: true
filepath_est: "/data/ov_state.txt"
filepath_std: "/data/ov_state_std.txt"
filepath_gt: "/data/ov_gt.txt"
use_klt: true
num_pts: 150
fast_threshold: 20
grid_x: 5
grid_y: 5
min_px_dist: 15
knn_ratio: 0.70
track_frequency: 31.0
downsample_cameras: false
num_opencv_threads: 4
histogram_method: "HISTOGRAM"
use_aruco: false
num_aruco: 1024
downsize_aruco: true
up_msckf_sigma_px: 1
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 1
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1
use_mask: true
mask0: "mask0.png"
mask1: "mask1.png"
relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
""")


if __name__ == "__main__":
    main()
