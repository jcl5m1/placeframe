#!/usr/bin/env python3
"""Upload a converted capture.tar to a Placeframe backend, reconstruct it, and publish a localization map.

  uv run upload_to_placeframe.py out/capture.tar                 # server from ../placeframe/.env PUBLIC_DOMAIN
  uv run upload_to_placeframe.py out/capture.tar --server https://my-domain.ngrok-free.app
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
import time
from pathlib import Path

import httpx

DEFAULT_PLACEFRAME_DIR = Path(__file__).resolve().parent.parent / "placeframe"
TERMINAL = {"succeeded", "failed", "cancelled"}


def public_domain(placeframe_dir: Path) -> str | None:
    env = placeframe_dir / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("PUBLIC_DOMAIN=") and "<" not in line:
            return line.split("=", 1)[1].strip()
    return None


def saved_credentials(placeframe_dir: Path) -> dict[str, str]:
    path = placeframe_dir / ".credentials"
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def get_token(client: httpx.Client, server: str, realm: str, user: str, password: str) -> str:
    r = client.post(
        f"{server}/auth/realms/{realm}/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "placeframe-api", "username": user, "password": password},
    )
    if r.status_code != 200:
        sys.exit(f"Keycloak login failed ({r.status_code}): {r.text[:300]}")
    return r.json()["access_token"]


def check(r: httpx.Response, what: str) -> dict:
    if r.status_code >= 300:
        sys.exit(f"{what} failed ({r.status_code}): {r.text[:500]}")
    return r.json()


def upload_and_queue(client: httpx.Client, server: str, tar: Path, name: str, options: dict) -> dict:
    print(f"Uploading {tar.name} ({tar.stat().st_size / 1e6:.0f} MB) as capture session '{name}'")
    with tar.open("rb") as fh:
        session = check(
            client.post(
                f"{server}/capture_sessions",
                data={"device_type": "ARFoundation", "name": name},
                files={"data": (tar.name, fh, "application/x-tar")},
            ),
            "Capture upload",
        )
    print(f"  capture session {session['id']}")

    print(f"Queuing reconstruction with options {options}")
    recon = check(
        client.post(
            f"{server}/reconstructions",
            json={"create": {"capture_session_id": session["id"]}, "options": options},
        ),
        "Create reconstruction",
    )
    print(f"  reconstruction {recon['id']}")
    return recon


def choose_reconstruction_options(tar: Path, conversion: dict) -> tuple[dict, list[str]]:
    """Pick Placeframe reconstruction options from what the capture actually is."""
    with tarfile.open(tar) as tf:
        manifest = json.load(tf.extractfile("manifest.json"))
    options: dict = {}
    reasons: list[str] = []
    cams = [c for rig in manifest["rigs"] for c in rig["cameras"]]
    if len(cams) > 1:
        options["rig_verification"] = True
        reasons.append(f"{len(cams)}-camera rig: rig-aware geometric verification")

    if is_panoramic_rig(manifest):
        options |= {"rotation_threshold": 180.0, "neighbors_count": 16,
                    "bundle_adjustment_refine_sensor_from_rig": False}
        reasons.append("panoramic rig (views share a centre, span >90°): no heading gate between frames "
                       "(rotation_threshold=180; a patched server instead matches only overlapping views), "
                       "16 pose-nearest frames, rig extrinsics fixed (virtual views are exact)")
    else:
        reasons.append("ordinary rig: Placeframe defaults (12 neighbours, 30° heading gate)")

    source = conversion.get("pose_source", "insta360 SfM" if conversion else "unknown")
    if not conversion.get("metric"):
        reasons.append(f"poses from {source} without metric scale: pose priors off")
    elif "OpenVIO" in source or "OpenVINS" in source:
        options |= {"use_prior_position": True, "pose_prior_position_sigma_m": 1.0}
        backend = "OpenVINS" if "OpenVINS" in source else "OpenVIO"
        reasons.append(f"poses from {backend} (tightly coupled, ~1 m open-loop error on the X5 test): priors sigma 1.0 m")
    elif "VIO" in source:
        options |= {"use_prior_position": True, "pose_prior_position_sigma_m": 2.0}
        reasons.append("poses from open-loop VIO (drifts ~1-3 m): loose position priors, sigma 2.0 m")
    else:
        options |= {"use_prior_position": True, "pose_prior_position_sigma_m": 0.25}
        reasons.append(f"poses from {source} (globally consistent SfM): tight position priors, sigma 0.25 m")
    return options, reasons


def is_panoramic_rig(manifest: dict) -> bool:
    for rig in manifest["rigs"]:
        cams = rig["cameras"]
        if len(cams) < 3:
            continue
        if max(math.dist([0, 0, 0], [c["translation"][k] for k in "xyz"]) for c in cams) > 0.05:
            continue
        axes = [_optical_axis(c["rotation"]) for c in cams]
        spread = max(math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v))))))
                     for u in axes for v in axes)
        if spread > 90:
            return True
    return False


def _optical_axis(q: dict) -> list[float]:
    # Third row of the cam_from_rig rotation matrix = camera z axis expressed in the rig frame.
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    return [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tar", help="capture.tar produced by insta360_to_placeframe.py")
    ap.add_argument("--server", default=None, help="Placeframe base URL (default: https://<PUBLIC_DOMAIN from .env>)")
    ap.add_argument("--placeframe-dir", default=str(DEFAULT_PLACEFRAME_DIR), help="Placeframe repo (to read .env)")
    ap.add_argument("--user", default=None, help="default: PLACEFRAME_USER in placeframe/.credentials, else 'user'")
    ap.add_argument("--password", default=None, help="default: PLACEFRAME_PASSWORD in placeframe/.credentials")
    ap.add_argument("--realm", default="placeframe-dev")
    ap.add_argument("--name", default=None, help="capture session / map name (default: tar's folder name)")
    ap.add_argument("--no-map", action="store_true", help="stop after reconstruction; don't create a localization map")
    ap.add_argument("--no-wait", action="store_true", help="queue reconstruction and exit")
    ap.add_argument("--options", default=None, help="extra reconstruction options as JSON (merged over defaults)")
    ap.add_argument("--watch", default=None, metavar="RECON_ID", help="skip upload; resume waiting on a reconstruction")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (local Caddy certs)")
    args = ap.parse_args()

    creds = saved_credentials(Path(args.placeframe_dir))
    args.user = args.user or creds.get("PLACEFRAME_USER", "user")
    args.password = args.password or creds.get("PLACEFRAME_PASSWORD", "password")
    tar = Path(args.tar).expanduser().resolve()
    if not tar.exists():
        sys.exit(f"{tar} not found")
    name = args.name or tar.parent.name
    conv_path = tar.parent / "conversion.json"
    conversion = json.loads(conv_path.read_text()) if conv_path.exists() else {}

    server = args.server
    if not server:
        domain = public_domain(Path(args.placeframe_dir))
        if not domain:
            sys.exit("No --server given and PUBLIC_DOMAIN not set in placeframe/.env")
        server = f"https://{domain}"
    server = server.rstrip("/")

    options, reasons = choose_reconstruction_options(tar, conversion)
    print("Source-aware reconstruction settings:")
    for r in reasons:
        print(f"  - {r}")
    if args.options:
        options |= json.loads(args.options)

    with httpx.Client(verify=not args.insecure, timeout=httpx.Timeout(60, write=None, read=600)) as client:
        print(f"Logging in to {server} as {args.user}")
        token = get_token(client, server, args.realm, args.user, args.password)
        client.headers["Authorization"] = f"Bearer {token}"

        if args.watch:
            recon = {"id": args.watch}
            print(f"Watching existing reconstruction {args.watch}")
        else:
            recon = upload_and_queue(client, server, tar, name, options)
        if args.no_wait:
            return

        last = None
        started = time.time()
        while True:
            time.sleep(10)
            try:
                r = client.get(f"{server}/reconstructions/{recon['id']}")
            except httpx.TransportError as e:
                print(f"  (poll error, retrying: {e})")
                continue
            if r.status_code >= 500:
                print(f"  (server {r.status_code}, retrying)")
                continue
            if r.status_code == 401:
                client.headers["Authorization"] = f"Bearer {get_token(client, server, args.realm, args.user, args.password)}"
                continue
            status = check(r, "Poll reconstruction")["status"]
            if status != last:
                print(f"  [{int(time.time() - started):>5}s] {status}")
                last = status
            if status in TERMINAL:
                break
        if status != "succeeded":
            sys.exit("Reconstruction did not succeed — check Grafana (/grafana/) logs for reconstructor-cuda.")

        if args.no_map:
            return
        loc_map = check(
            client.post(
                f"{server}/localization-maps",
                json={
                    "reconstruction_id": recon["id"], "name": name, "active": True, "color": 0,
                    "position_x": 0, "position_y": 0, "position_z": 0,
                    "rotation_x": 0, "rotation_y": 0, "rotation_z": 0, "rotation_w": 1,
                },
            ),
            "Create localization map",
        )
        print(f"Localization map {loc_map['id']} is live. Open the Capture Tool and localize against it.")


if __name__ == "__main__":
    main()
