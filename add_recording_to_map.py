#!/usr/bin/env python3
"""Add a new Insta360 recording to an existing Placeframe map (rebuilds one fused map).

Placeframe builds one map per capture session; a capture may hold several rigs whose poses share one world
frame. This tool processes the new recording (lens calibration, odometry, stitched views), registers it into the
existing map's frame by localizing sample frames against that map with Placeframe's own localizer, then
appends it as an extra rig to the map's original capture and reconstructs a new map.

Every stage writes a marker into the job directory, so re-running the same command resumes after the last
completed stage.

  uv run add_recording_to_map.py VID.insv --map <localization-map-id> --job-dir ~/pf_mapper/jobs/<id> \
      [--name "house v2"] [--replace] [--odometry openvio|openvins]

Odometry backends (both run from local Docker images and write the IMU trajectory to <job>/ov.tum):
  openvio  (default) OpenVIO with its native dual-fisheye 360 model          image openvio:local
  openvins OpenVINS MSCKF, the two lenses as two equidistant fisheye cameras  image openvins:local
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx
import numpy as np
from scipy.spatial.transform import Rotation

import gps_placement
from insta360_to_placeframe import parse_views
from upload_to_placeframe import choose_reconstruction_options, get_token, saved_credentials

HERE = Path(__file__).resolve().parent
PLACEFRAME_DIR = HERE.parent / "placeframe"
OPENVIO_IMAGE = "openvio:local"
OPENVINS_IMAGE = "openvins:local"
OPENVINS_TIME_ZERO = 1.0  # insv_to_openvins.py writes bag time = video time + 1 s (ROS time must be > 0)
OPENVIO_NOISE = ["--imu.gyro_noise=4.144e-04", "--imu.accel_noise=2.94e-03",
                 "--imu.gyro_walk=3.92e-04", "--imu.accel_walk=9.21e-04"]
ODOMETRY = ("openvio", "openvins")


def stages(odometry: str = "openvio") -> list[str]:
    return ["calibrate", f"{odometry}-export", f"{odometry}-run", "stitch", "views", "capture", "register", "merge",
            "reconstruct", "publish", "geolocate"]


STAGES = stages()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv")
    ap.add_argument("--map", required=True, help="target localization map id")
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--name", default=None, help="name for the new map (default: '<old name> +1')")
    ap.add_argument("--replace", action="store_true", help="deactivate the old map once the new one is live")
    ap.add_argument("--register-frames", type=int, default=40, help="frames sampled for registration")
    ap.add_argument("--odometry", choices=ODOMETRY, default="openvio", help="odometry backend (default openvio)")
    args = ap.parse_args()

    job = Path(args.job_dir).expanduser()
    job.mkdir(parents=True, exist_ok=True)
    odo_file = job / "odometry.txt"  # ov.tum and the stages after it belong to one backend
    if odo_file.exists() and odo_file.read_text().strip() != args.odometry:
        raise SystemExit(f"{job} was started with --odometry {odo_file.read_text().strip()}; use a new job dir")
    odo_file.write_text(args.odometry)
    insv = str(Path(args.insv).expanduser().resolve())
    api = PlaceframeApi()

    def run_stage(name, fn):
        marker = job / f".done-{name}"
        if marker.exists():
            log(f"== STAGE {name} (already done)")
            return
        log(f"== STAGE {name}")
        t0 = time.time()
        fn()
        marker.write_text(json.dumps({"seconds": round(time.time() - t0, 1)}))
        log(f"== DONE {name} ({time.time() - t0:.0f}s)")

    py = [sys.executable, "-u"]
    run_stage("calibrate", lambda: sh(py + [str(HERE / "insta360_dualfisheye_to_placeframe.py"), insv,
                                            "-o", str(job / "fisheye"), "--loop-closure"]))
    if args.odometry == "openvins":
        run_stage("openvins-export", lambda: sh(py + [str(HERE / "insv_to_openvins.py"), insv, "--calib",
                                                      str(job / "fisheye/work"), "-o", str(job / "openvins")]))
        run_stage("openvins-run", lambda: run_openvins(job))
        pose_calib = ["--cam-from-imu", str(job / "openvins/config/kalibr_imucam_chain.yaml")]
    else:
        run_stage("openvio-export", lambda: sh(py + [str(HERE / "insv_to_openvio.py"), insv, "--calib",
                                                     str(job / "fisheye/work"), "-o", str(job / "openvio"),
                                                     "--bottom-mask", "1.0"]))
        run_stage("openvio-run", lambda: run_openvio(job))
        pose_calib = ["--imu-from-camera", str(job / "openvio/x5.yaml")]
    run_stage("stitch", lambda: sh(py + [str(HERE / "stitch_dualfisheye.py"), insv, "--work",
                                         str(job / "fisheye/work"), "-o", str(job / "equirect")]))
    run_stage("views", lambda: sh(py + [str(HERE / "insta360_to_placeframe.py"), str(job / "equirect"), "-o",
                                        str(job / "views"), "--image-interval-ms", "500", "--views-only"]))
    run_stage("capture", lambda: sh(py + [str(HERE / "vio_to_capture.py"), "--insv", insv, "--vio",
                                          str(job / "ov.tum"), *pose_calib, "--stitch-work", str(job / "fisheye/work"), "--equirect",
                                          str(job / "equirect"), "--views", str(job / "views/work"),
                                          "-o", str(job / "capture"), "--step", "1"]))
    run_stage("register", lambda: register(api, job, args.map, args.register_frames))
    run_stage("merge", lambda: merge(api, job, args.map))
    run_stage("reconstruct", lambda: reconstruct(api, job, args.map, args.name))
    run_stage("publish", lambda: publish(api, job, args.map, args.replace))
    run_stage("geolocate", lambda: geolocate(api, job, insv))
    result = json.loads((job / "result.json").read_text())
    log(f"== COMPLETE new map {result['map_id']} '{result['name']}'")


# --------------------------------------------------------------------------- stages


def geolocate(api: "PlaceframeApi", job: Path, insv: str) -> None:
    """Put the new map on the globe from the recording's GPS, so clients can find it by location. The X5 has no GPS
    receiver: fixes exist only when the recording was shot from the Insta360 phone app with GPS on, or from a GPS
    remote (see GPS.md). A map that is already placed (merges inherit the base map's placement) is left alone; the
    recording's GPS is only compared against it. Never fails the job."""
    result = json.loads((job / "result.json").read_text())
    try:
        fixes = gps_placement.insv_gps(Path(insv))
    except Exception as e:  # noqa: BLE001 - a GPS problem must not fail a finished merge
        log(f"GPS: could not read fixes ({e!r})")
        fixes = []
    fix = gps_placement.median_fix(fixes)
    (job / "gps.json").write_text(json.dumps({"fixes": fixes, "median": fix}, indent=1))
    if not fix:
        log("GPS: none in this recording (shoot from the Insta360 app with GPS on to record it); map not placed")
        result["gps"] = None
    else:
        new_map = api.get(f"/localization-maps/{result['map_id']}")
        if gps_placement.is_placed(new_map):
            log(f"GPS: {fix['n']} fixes, median {fix['lat']:.6f},{fix['lon']:.6f}; map already placed, left as is")
            result["gps"] = fix | {"placed": False}
        else:
            gps_placement.place_map(api, result["map_id"], fix["lat"], fix["lon"], fix["alt"])
            log(f"GPS: placed map at {fix['lat']:.6f},{fix['lon']:.6f} (median of {fix['n']} fixes; heading unknown)")
            result["gps"] = fix | {"placed": True}
    (job / "result.json").write_text(json.dumps(result, indent=2))


def run_openvio(job: Path) -> None:
    yaml = (job / "openvio/x5.yaml").read_text()
    rows = re.findall(r"^\s+- \[(.*)\]$", yaml, re.M)[:4]
    imu_from_camera = "[" + ",".join("[" + r + "]" for r in rows) + "]"
    sh(["docker", "run", "--rm", "-v", f"{job}:/job", OPENVIO_IMAGE, "build/bin/vio", "odometry",
        "--dataset_path", "/job/openvio", "--config_path", "config/panoair.yaml", "--output_path", "/job/ov.tum",
        "--camera.mask_rectangles=[]", "--camera.params=[0.0]", "--camera.polar_exclusion_ratio=0.12",
        f"--rig.imu_from_camera={imu_from_camera}", "--rig.clock_offset=0.0", *OPENVIO_NOISE])
    n = sum(1 for _ in open(job / "ov.tum"))
    log(f"OpenVIO poses: {n}")


def run_openvins(job: Path) -> None:
    """OpenVINS serial (bag) runner in its ROS Noetic image; state -> ov.tum (video time, world_from_imu)."""
    d = job / "openvins"
    rosrun = ("source /ws/devel/setup.bash && (roscore >/dev/null 2>&1 &) && sleep 4 && "
              "rosrun ov_msckf ros1_serial_msckf _path_bag:=/data/x5.bag _bag_start:=0.0 _bag_durr:=-1 "
              "_verbosity:=WARNING _config_path:=/data/config/estimator_config.yaml _use_stereo:=false "
              "_max_cameras:=2 _save_total_state:=true _filepath_est:=/data/ov_state.txt "
              "_filepath_std:=/data/ov_state_std.txt _filepath_gt:=/data/ov_gt.txt")
    sh(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-v", f"{d}:/data",
        OPENVINS_IMAGE, "bash", "-c", rosrun])
    # ov_state rows: t, q_GtoI (JPL x y z w; equals Hamilton world_from_imu), p_IinG, ... Bag time = video + 1 s.
    rows = []
    for line in open(d / "ov_state.txt"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        rows.append(f"{float(f[0]) - OPENVINS_TIME_ZERO:.6f} {f[5]} {f[6]} {f[7]} {f[1]} {f[2]} {f[3]} {f[4]}")
    if not rows:
        raise SystemExit("OpenVINS produced no poses (did it initialize?)")
    (job / "ov.tum").write_text("\n".join(rows) + "\n")
    log(f"OpenVINS poses: {len(rows)} ({rows[0].split()[0]}-{rows[-1].split()[0]} s)")


def register(api: PlaceframeApi, job: Path, map_id: str, n_frames: int) -> None:
    """Localize sample frames of the new recording against the map; fit map_from_new (rotation + translation)."""
    with tarfile.open(job / "capture/capture.tar") as tf:
        manifest = json.load(tf.extractfile("manifest.json"))
        rows = tf.extractfile("rig0/frames.csv").read().decode().splitlines()[1:]
    frames = [r.split(",") for r in rows]
    cam_cfg = manifest["rigs"][0]["cameras"][0]["camera_config"]
    views = parse_views(None, len(manifest["rigs"][0]["cameras"]), 0.0)
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)]
    pairs = []
    for f in picks:
        ts = f[0]
        R_new = Rotation.from_quat([float(x) for x in f[4:8]]).as_matrix()
        C_new = np.array([float(x) for x in f[1:4]])
        best = None
        for k, vw in enumerate(views):
            img = job / "views/work/images/rig0" / vw.name / f"{ts}.jpg"
            loc = api.localize(map_id, img.read_bytes(), cam_cfg)
            if loc and (best is None or loc["confidence"] > best["confidence"]):
                best = {**loc, "view": k}
            if best and best["confidence"] > 0.5:
                break
        if best is None:
            log(f"  frame {int(ts) / 1000:6.1f}s: not localized")
            continue
        q = best["camera_from_map"]["rotation"]
        t = best["camera_from_map"]["translation"]
        R_cam_map = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        t_cam_map = np.array([t["x"], t["y"], t["z"]])
        R_map_cam = R_cam_map.T
        C_map = -R_map_cam @ t_cam_map
        R_map_rig = R_map_cam @ views[best["view"]].rig_from_cam().T  # cam_from_rig
        pairs.append({"ts": int(ts), "R_new": R_new, "C_new": C_new, "R_map": R_map_rig, "C_map": C_map,
                      "confidence": best["confidence"]})
        log(f"  frame {int(ts) / 1000:6.1f}s: localized via view {best['view']} (confidence {best['confidence']:.2f})")
    if len(pairs) < 4:
        raise SystemExit(f"Registration failed: only {len(pairs)} frames localized. Does the recording overlap the map?")

    best_inl, best_R = None, None
    for p in pairs:  # one correspondence fixes a rigid hypothesis (rotation + position)
        R = p["R_map"] @ p["R_new"].T
        t = p["C_map"] - R @ p["C_new"]
        # Thresholds sized to OpenVIO's open-loop drift on the X5 (~1-1.5 m, a few degrees of yaw).
        inl = [q for q in pairs if np.linalg.norm(R @ q["C_new"] + t - q["C_map"]) < 1.5
               and np.degrees(Rotation.from_matrix((R @ q["R_new"]).T @ q["R_map"]).magnitude()) < 12]
        if best_inl is None or len(inl) > len(best_inl):
            best_inl, best_R = inl, R
    A = np.array([q["C_new"] for q in best_inl])
    B = np.array([q["C_map"] for q in best_inl])
    if len(best_inl) >= 3 and np.ptp(A, axis=0).max() > 1.0:
        ma, mb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    else:
        R, ma, mb = best_R, A.mean(0), B.mean(0)
    t = mb - R @ ma
    res = np.linalg.norm((A @ R.T + t) - B, axis=1)
    if len(best_inl) < 4:
        raise SystemExit(f"Registration failed: only {len(best_inl)} consistent localizations out of {len(pairs)}")
    # Residual drift at each consistent localization becomes a time-interpolated position correction
    # (loop closure against the existing map) applied on top of the rigid transform.
    knots = sorted((q["ts"], (q["C_map"] - (R @ q["C_new"] + t)).tolist()) for q in best_inl)
    out = {"R": R.tolist(), "t": t.tolist(), "inliers": len(best_inl), "localized": len(pairs),
           "sampled": len(picks), "rms_m": float(np.sqrt(np.mean(res ** 2))), "max_m": float(res.max()),
           "drift_knots": knots}
    (job / "registration.json").write_text(json.dumps(out, indent=2))
    log(f"registered: {out['inliers']}/{out['localized']} consistent localizations "
        f"(of {out['sampled']} sampled), residual rms {out['rms_m']:.2f} m, max {out['max_m']:.2f} m")


def merge(api: PlaceframeApi, job: Path, map_id: str) -> None:
    reg = json.loads((job / "registration.json").read_text())
    R, t = np.array(reg["R"]), np.array(reg["t"])
    kt = np.array([k[0] for k in reg["drift_knots"]], float)
    kv = np.array([k[1] for k in reg["drift_knots"]], float)

    def drift_at(ts_ms: int) -> np.ndarray:
        return np.array([np.interp(ts_ms, kt, kv[:, j]) for j in range(3)])

    lm = api.get(f"/localization-maps/{map_id}")
    rec = api.get(f"/reconstructions/{lm['reconstruction_id']}")
    old_tar = job / "base_capture.tar"
    if not old_tar.exists():
        log(f"downloading base capture {rec['capture_session_id']}")
        api.download(f"/capture_sessions/{rec['capture_session_id']}/tar", old_tar)
    with tarfile.open(old_tar) as old:
        manifest = json.load(old.extractfile("manifest.json"))
    with tarfile.open(job / "capture/capture.tar") as new:
        new_manifest = json.load(new.extractfile("manifest.json"))
    taken = {r["id"] for r in manifest["rigs"]}
    new_id = next(f"rig{i}" for i in range(1, 1000) if f"rig{i}" not in taken)
    new_rig = new_manifest["rigs"][0] | {"id": new_id}
    manifest["rigs"].append(new_rig)

    merged = job / "merged.tar"
    with tarfile.open(merged, "w") as out:
        def add_bytes(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = int(time.time())
            out.addfile(ti, io.BytesIO(data))

        add_bytes("manifest.json", json.dumps(manifest, indent=2).encode())
        # Recordings already in the map get the map's solved poses as priors instead of their raw odometry
        # (which drifts by metres/degrees): mapping then only has to solve the new recording.
        map_poses = solved_frame_poses(rec["id"])
        with tarfile.open(old_tar) as old:
            for m in old:
                if m.name == "manifest.json" or not m.isfile():
                    continue
                if m.name.endswith("/frames.csv") and map_poses:
                    rig = m.name.split("/")[0]
                    text, used = refine_priors(old.extractfile(m).read().decode(), rig, map_poses)
                    log(f"{rig}: {used} frame priors taken from the base map's solved poses")
                    add_bytes(m.name, text.encode())
                    continue
                out.addfile(m, old.extractfile(m))
        with tarfile.open(job / "capture/capture.tar") as new:
            for m in new:
                if not m.isfile() or m.name == "manifest.json":
                    continue
                data = new.extractfile(m).read()
                name = m.name.replace("rig0/", f"{new_id}/", 1)
                if m.name == "rig0/frames.csv":
                    lines = data.decode().splitlines()
                    body = []
                    for line in lines[1:]:
                        f = line.split(",")
                        C = R @ np.array([float(x) for x in f[1:4]]) + t + drift_at(int(f[0]))
                        q = Rotation.from_matrix(R @ Rotation.from_quat([float(x) for x in f[4:8]]).as_matrix()).as_quat()
                        body.append(f"{f[0]},{C[0]:.6f},{C[1]:.6f},{C[2]:.6f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")
                    data = ("\n".join([lines[0]] + body) + "\n").encode()
                add_bytes(name, data)
    source = json.loads((job / "capture/conversion.json").read_text()).get("pose_source", "insta360 VIO (OpenVIO)")
    conv = {"pose_source": f"{source} + base map", "metric": True, "rigs": [r["id"] for r in manifest["rigs"]],
            "base_capture_session": rec["capture_session_id"], "base_map": map_id}
    (job / "conversion.json").write_text(json.dumps(conv, indent=2))
    log(f"merged capture: {len(manifest['rigs'])} rigs ({', '.join(conv['rigs'])}), {merged.stat().st_size / 1e6:.0f} MB")


def solved_frame_poses(reconstruction_id: str) -> dict[tuple[str, int], tuple[Rotation, np.ndarray]]:
    """(rig, frame ms) -> world_from_rig of the reference camera, from the stored model (already in map frame)."""
    p = subprocess.run(["sg", "docker", "-c", f"docker exec placeframe-minio-1 mc cat "
                        f"l/dev-reconstructions/{reconstruction_id}/sfm_model/images.txt"], capture_output=True, text=True)
    if p.returncode:
        log(f"base map poses unavailable ({p.stderr.strip()[:200]}); keeping odometry priors")
        return {}
    poses = {}
    for line in [ln for ln in p.stdout.splitlines() if ln and not ln.startswith("#")][0::2]:
        f = line.split()
        rig, cam, stem = f[9].split("/")
        if cam != "camera0":  # the reference sensor: its pose is the rig pose
            continue
        q, tr = [float(x) for x in f[1:5]], np.array([float(x) for x in f[5:8]])
        world_from_cam = Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv()
        poses[(rig, int(stem.split(".")[0]))] = (world_from_cam, -world_from_cam.apply(tr))
    return poses


def refine_priors(frames_csv: str, rig: str, map_poses: dict) -> tuple[str, int]:
    """Replace odometry priors with solved map poses; frames the map never placed keep their odometry,
    re-anchored on the nearest placed frame so they stay consistent with their neighbours."""
    lines = frames_csv.splitlines()
    odo = {int(r.split(",")[0]): r.split(",") for r in lines[1:]}
    placed = sorted(t for (rg, t) in map_poses if rg == rig and t in odo)
    if not placed:
        return frames_csv, 0
    body = []
    for ts, f in sorted(odo.items()):
        Rv, Cv = Rotation.from_quat([float(x) for x in f[4:8]]), np.array([float(x) for x in f[1:4]])
        if (rig, ts) in map_poses:
            Rw, C = map_poses[(rig, ts)]
        else:
            k = min(placed, key=lambda x: abs(x - ts))
            Rk, Ck = map_poses[(rig, k)]
            fk = odo[k]
            Rvk, Cvk = Rotation.from_quat([float(x) for x in fk[4:8]]), np.array([float(x) for x in fk[1:4]])
            correction = Rk * Rvk.inv()
            Rw, C = correction * Rv, Ck + correction.apply(Cv - Cvk)
        q = Rw.as_quat()
        body.append(f"{ts},{C[0]:.6f},{C[1]:.6f},{C[2]:.6f},{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")
    return "\n".join([lines[0]] + body) + "\n", sum(1 for ts in odo if (rig, ts) in map_poses)


def reconstruct(api: PlaceframeApi, job: Path, map_id: str, name: str | None) -> None:
    state = job / "reconstruct.json"
    st = json.loads(state.read_text()) if state.exists() else {}
    old = api.get(f"/localization-maps/{map_id}")
    st.setdefault("name", name or f"{old.get('name') or 'map'} +1")
    if "capture_session_id" not in st:
        log(f"uploading merged capture as '{st['name']}'")
        st["capture_session_id"] = api.upload_capture(job / "merged.tar", st["name"])
        state.write_text(json.dumps(st))
    if "reconstruction_id" not in st:
        conv = json.loads((job / "conversion.json").read_text())
        options, reasons = choose_reconstruction_options(job / "merged.tar", conv)
        for r in reasons:
            log(f"  - {r}")
        st["reconstruction_id"] = api.post("/reconstructions", {"create": {"capture_session_id": st["capture_session_id"]},
                                                                "options": options})["id"]
        state.write_text(json.dumps(st))
    log(f"reconstruction {st['reconstruction_id']}")
    last = None
    while True:
        status = api.get(f"/reconstructions/{st['reconstruction_id']}")["status"]
        if status != last:
            log(f"  server: {status}")
            last = status
        if status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(15)
    if status != "succeeded":
        raise SystemExit(f"Reconstruction {status}")


def publish(api: PlaceframeApi, job: Path, map_id: str, replace: bool) -> None:
    st = json.loads((job / "reconstruct.json").read_text())
    old = api.get(f"/localization-maps/{map_id}")
    try:
        new_map_id = api.get(f"/reconstructions/{st['reconstruction_id']}/localization_map")
    except httpx.HTTPStatusError:
        new_map_id = None
    if not new_map_id:
        body = {k: old[k] for k in ("position_x", "position_y", "position_z", "rotation_x", "rotation_y",
                                    "rotation_z", "rotation_w", "color")}
        new_map_id = api.post("/localization-maps", {**body, "reconstruction_id": st["reconstruction_id"],
                                                     "name": st["name"], "active": True})["id"]
    if replace:
        api.patch(f"/localization-maps/{map_id}", {"active": False})
        log(f"old map {map_id} deactivated (kept; reactivate to roll back)")
    metrics = (api.get(f"/reconstructions/{st['reconstruction_id']}").get("manifest") or {}).get("metrics") or {}
    result = {"map_id": new_map_id, "name": st["name"], "reconstruction_id": st["reconstruction_id"],
              "registered_images": metrics.get("registered_images"), "total_images": metrics.get("total_images"),
              "points": metrics.get("num_3d_points"), "replaced": replace, "old_map": map_id}
    (job / "result.json").write_text(json.dumps(result, indent=2))
    log(f"published map {new_map_id}: {result['registered_images']}/{result['total_images']} images, {result['points']} points")


# --------------------------------------------------------------------------- helpers


class PlaceframeApi:
    def __init__(self) -> None:
        creds = saved_credentials(PLACEFRAME_DIR)
        self.server = creds["PLACEFRAME_URL"].rstrip("/")
        self.user, self.password = creds["PLACEFRAME_USER"], creds["PLACEFRAME_PASSWORD"]
        self.client = httpx.Client(timeout=httpx.Timeout(120, write=None, read=900))
        self._token_at = 0.0

    def _auth(self) -> dict:
        if time.time() - self._token_at > 600:
            self.client.headers["Authorization"] = "Bearer " + get_token(
                self.client, self.server, "placeframe-dev", self.user, self.password)
            self._token_at = time.time()
        return {}

    def _check(self, r: httpx.Response):
        r.raise_for_status()
        return r.json() if r.content else None

    def get(self, path):
        self._auth()
        return self._check(self.client.get(self.server + path))

    def post(self, path, body):
        self._auth()
        return self._check(self.client.post(self.server + path, json=body))

    def patch(self, path, body):
        self._auth()
        return self._check(self.client.patch(self.server + path, json=body))

    def download(self, path, dest: Path):
        self._auth()
        tmp = dest.with_suffix(".part")
        with self.client.stream("GET", self.server + path) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        tmp.rename(dest)

    def upload_capture(self, tar: Path, name: str) -> str:
        self._auth()
        with open(tar, "rb") as fh:
            r = self.client.post(self.server + "/capture_sessions", data={"device_type": "ARFoundation", "name": name},
                                 files={"data": (tar.name, fh, "application/x-tar")})
        return self._check(r)["id"]

    def localize(self, map_id: str, jpeg: bytes, cam_cfg: dict):
        self._auth()
        r = self.client.post(self.server + "/localize", data={
            "map_ids": json.dumps([map_id]), "camera_config": json.dumps(cam_cfg), "axis_convention": "OPENCV"},
            files={"image": ("q.jpg", jpeg, "image/jpeg")})
        if r.status_code >= 400:
            return None
        res = r.json()
        if not res:
            return None
        m = res[0]
        conf = m.get("metrics", {}).get("confidence_loose") or 0.0
        return {"camera_from_map": m["camera_from_map_transform"], "confidence": float(conf)}


def sh(cmd: list[str]) -> None:
    log("$ " + " ".join(cmd[:3]) + (" ..." if len(cmd) > 3 else ""))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
    assert p.stdout is not None
    for line in p.stdout:
        if line.startswith(("I20", "W20")):  # COLMAP glog chatter
            continue
        print("   " + line.rstrip(), flush=True)
    if p.wait() != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd[:3])}")


def log(msg: str) -> None:
    print(msg, flush=True)


if __name__ == "__main__":
    main()
