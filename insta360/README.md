# placeframe-insta360

Turn Insta360 360° footage into a Placeframe visual-positioning map.

## 1. Install (once)
    sudo ./install.sh
Installs ffmpeg, Docker + Compose, the NVIDIA Container Toolkit, uv, and the Python envs for this
tool and `../placeframe`. Then log out and back in, set `PUBLIC_DOMAIN` and `NGROK_AUTHTOKEN` in
`../placeframe/.env`, and run `cd ../placeframe && uv run up --quiet-pull`.

## 2. Capture
- Hold the camera upright (selfie stick above head is ideal) and walk the space slowly; loop back to the start.
- Measure the lens height above the floor; you'll pass it as `--camera-height` for metric scale.
- In Insta360 Studio export **360 (equirectangular) MP4**, FlowState stabilisation + horizon lock ON.

## 3. Convert
    uv run insta360_to_placeframe.py ~/walk.mp4 --camera-height 1.7
Output: `~/walk_placeframe/capture.tar` + `conversion.json`.
Useful flags: `--fps 2` (sampling), `--views 6 --hfov 90` (virtual cameras), `--scale` (instead of
camera height), `--loop-closure`, `--cpu`, `--keep-work`, `--reuse-frames`. A folder of exported
equirect photos also works as input.

## 4. Upload, reconstruct, publish
    uv run upload_to_placeframe.py ~/walk_placeframe/capture.tar
Logs in (user/password by default), uploads the capture, queues reconstruction with rig + pose-prior
options, waits, and creates an active localization map. Then localize with the Placeframe Capture Tool.

## How it works
Each 360 frame → N pinhole views sharing one optical centre (a Placeframe "rig", OPENCV axes, camera0 =
reference). A rig-aware COLMAP pass recovers per-frame poses (the Insta360 has no AR tracking); the world
is levelled using the horizon-locked rig's down axis and scaled from the floor plane under the camera.

## Raw dual-fisheye .insv (no Insta360 Studio export) — e.g. X5
Tested on an X5 handheld walk (3 min, multi-room). Steps:

    # 1. self-calibrate the two lenses + rough poses from the raw fisheye streams
    uv run insta360_dualfisheye_to_placeframe.py VID_xxx.insv -o ~/cap/fisheye --loop-closure
    # 2. stitch gravity-levelled equirect frames (lens calibration from step 1, gravity from the camera IMU)
    uv run stitch_dualfisheye.py VID_xxx.insv --work ~/cap/fisheye/work -o ~/cap/equirect
    # 3. poses on the stitched frames (keep the SfM so it can be rescaled)
    uv run insta360_to_placeframe.py ~/cap/equirect -o ~/cap/out --image-interval-ms 500 --loop-closure --keep-work
    # 4. metric scale from the IMU, then rewrite the tar at that scale (seconds)
    uv run estimate_imu_scale.py VID_xxx.insv ~/cap/out/work/sfm
    uv run insta360_to_placeframe.py ~/cap/equirect -o ~/cap/out --image-interval-ms 500 --keep-work \
        --reuse-frames --reuse-sfm --scale <scale from step 4>
    # 5. upload
    uv run upload_to_placeframe.py ~/cap/out/capture.tar --name my-place

Notes
- The bottom of each lens (selfie stick, hand, operator) is blacked out (`--bottom-mask`).
- X5 IMU axes are hard-coded in `stitch_dualfisheye.py` (`LENS0_FROM_IMU`); other models may differ —
  the stitcher prints the gravity spread, which should be small (< ~0.05).
- `--camera-height` / `--room-height` scaling relies on finding the floor/ceiling plane and is easily fooled
  by tables and masked ceilings; prefer the IMU scale when its fit R^2 is > ~0.6.
- Loop closure uses COLMAP's vocab tree, downloaded once to `~/.cache/placeframe-insta360/`.

## VIO poses in place of ARCore/ARKit (raw .insv, recommended)
Placeframe's reconstructor ingests per-frame device poses (`rig0/frames.csv`) — on phones these come from
ARCore/ARKit VIO. For an Insta360 we compute our own open-loop VIO and put it in that same slot, so the
server builds the map itself (pose-proximity pairs + position priors) instead of relying on our SfM:

    # lens calibration (step 1 above) and stitched views (steps 2-3, any fps; e.g. 4 fps, --bottom-mask 0.55)
    uv run vio_dualfisheye.py VID.insv --calib ~/cap/fisheye/work -o ~/cap/vio      # pose for every video frame
    uv run vio_to_capture.py --insv VID.insv --vio ~/cap/vio/vio.csv --stitch-work ~/cap/v2/work \
        --equirect ~/cap/v2/equirect --views ~/cap/v2/out/work -o ~/cap/vio_capture --step 2
    uv run upload_to_placeframe.py ~/cap/vio_capture/capture.tar --name my-place \
        --options '{"pose_prior_position_sigma_m": 2.0, "neighbors_count": 16}'

VIO = gyro rotation (auto-calibrated against the lens SfM: time offset, mount, gyro scale) + fisheye KLT
translation (operator tracks auto-rejected) + accelerometer scale in 10 s windows. No loop closure. On the
X5 test walk: pose for all 5,777 frames, 0.4-1.8 m median drift over 3 min / ~200 m vs the SfM map.
Note: ffmpeg's fps filter labels extracted frames up to ~0.25 s off true video time; the tools detect this.

## Map updater web UI (add recordings to an existing map)
`https://<PUBLIC_DOMAIN>/mapper/` — log in with the Placeframe account, pick a map, drop raw `.insv` files.

- **Resumable uploads**: 16 MB chunks, each SHA-256 verified; the server keeps exactly what it has. If the connection
  drops the page retries with back-off; if the page is closed, choose the same file again and it resumes.
- Each finished upload queues a job that runs `add_recording_to_map.py`:
  calibrate → OpenVIO → stitch → views → capture → **register** (localize sample frames against the map, fit a
  rigid transform + time-interpolated drift correction) → **merge** (base capture + new rig) → reconstruct → publish.
  Stages leave markers, so a crashed or restarted job continues after its last finished stage.
- The new map is published next to the old one; "make active" deactivates (not deletes) the old map.
- Start the service: `mapper/run.sh` (listens on 127.0.0.1:8150; Caddy funnel-filter routes `/mapper/*` to it).
  Data lives in `~/pf_mapper/{uploads,jobs}`.
- CLI equivalent: `uv run add_recording_to_map.py VID.insv --map <map-id> --job-dir ~/pf_mapper/jobs/manual --replace`

## Recording and replaying phone localizations

The Placeframe API (local patch, `compose.local.yml`) records every `/localize` request the phone sends:
image, camera intrinsics, map ids, server response, latency, client IP and user agent. Recordings go to
`/mnt/hdd12t/placeframe_scratch/localization_recordings/<date>/` (`requests.jsonl` plus one JPEG per request).
To turn this off, remove `LOCALIZATION_RECORD_DIR` from the `api` service.

```bash
python replay_localizations.py summary                      # what clients experienced, per session
python replay_localizations.py replay --out report.json     # regression test: re-send, compare (exit 1 on regression)
python replay_localizations.py replay --map-id <new map>    # same place, different map build: success/confidence
python replay_localizations.py load --clients 8 --duration 120   # simulated concurrent phones: rps, p95, GPU/CPU per request
```

Replayed requests carry `X-Placeframe-Replay`, so they are not recorded again. The phone's own AR tracking pose
is not part of the request, so it is not recorded.
