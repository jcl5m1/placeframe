#!/usr/bin/env python3
"""Place a Placeframe map on the globe from a GPS fix, so clients can find it by location.

A map's placement (position_* / rotation_*) is the transform from the map's own coordinates to ECEF (Earth-centred,
Earth-fixed, metres). Placeframe uses it to compute every camera's ECEF position, and GET /localization-maps with
position_x/y/z + radius returns the maps with a camera inside that sphere: that is how clients select nearby maps.

From one GPS fix (latitude, longitude, height above the WGS84 ellipsoid) the map is placed so that its anchor point
(by default the centroid of its camera positions) sits at the fix, with the map's vertical axis along local up.
Heading is unknown from a single fix: --heading sets it (degrees clockwise from north of the map's +x axis, after
levelling), otherwise 0. For map *selection* heading does not matter; for AR content placement refine the result
in the MapRegistrationTool.

  python gps_placement.py place --map-id <id> --lat 37.42 --lon -122.08 [--alt 10] [--heading 0] [--dry-run]
  python gps_placement.py check --lat 37.42 --lon -122.08 [--alt 10] [--radius 100]   # maps a client would load
  python gps_placement.py insv-gps recording.insv                                      # GPS fixes in a recording
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from add_recording_to_map import PlaceframeApi


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
IDENTITY = {"position_x": 0.0, "position_y": 0.0, "position_z": 0.0,
            "rotation_x": 0.0, "rotation_y": 0.0, "rotation_z": 0.0, "rotation_w": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("place")
    p.add_argument("--map-id", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--alt", type=float, default=0.0, help="metres above the WGS84 ellipsoid")
    p.add_argument("--heading", type=float, default=0.0)
    p.add_argument("--up", choices=["auto", "x", "-x", "y", "-y", "z", "-z"], default="auto")
    p.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--lat", type=float, required=True)
    c.add_argument("--lon", type=float, required=True)
    c.add_argument("--alt", type=float, default=0.0)
    c.add_argument("--radius", type=float, default=100.0)
    g = sub.add_parser("insv-gps")
    g.add_argument("insv", type=Path)
    a = ap.parse_args()

    if a.cmd == "insv-gps":
        fixes = insv_gps(a.insv)
        print(json.dumps({"fixes": len(fixes), "median": median_fix(fixes), "first": fixes[:3]}, indent=1))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from add_recording_to_map import PlaceframeApi  # imported here: add_recording_to_map imports this module

    api = PlaceframeApi()
    if a.cmd == "check":
        x, y, z = ecef_from_llh(a.lat, a.lon, a.alt)
        maps = api.get(f"/localization-maps?position_x={x}&position_y={y}&position_z={z}&radius={a.radius}")
        print(f"{len(maps)} map(s) within {a.radius:g} m:")
        for m in maps:
            print(f"  {m['id']}  {m.get('name')}  {'active' if m.get('active') else 'inactive'}")
        return
    body = place_map(api, a.map_id, a.lat, a.lon, a.alt, a.heading, a.up, dry_run=a.dry_run)
    print(json.dumps(body, indent=1))


def place_map(api: "PlaceframeApi", map_id: str, lat: float, lon: float, alt: float = 0.0, heading_deg: float = 0.0,
              up: str = "auto", anchor: np.ndarray | None = None, dry_run: bool = False) -> dict:
    m = api.get(f"/localization-maps/{map_id}")
    positions = map_camera_positions(api, m["reconstruction_id"])
    body = placement(lat, lon, alt, positions, heading_deg, up, anchor)
    if not dry_run:
        api.patch(f"/localization-maps/{map_id}", body)
    return body


def placement(lat: float, lon: float, alt: float, positions: np.ndarray, heading_deg: float = 0.0, up: str = "auto",
              anchor: np.ndarray | None = None) -> dict:
    enu_from_map = _enu_from_map(positions, up)
    enu_from_map = Rotation.from_euler("z", -heading_deg, degrees=True).as_matrix() @ enu_from_map
    ecef_from_map = ecef_from_enu(lat, lon) @ enu_from_map
    p0 = positions.mean(axis=0) if anchor is None else np.asarray(anchor, float)
    t = np.array(ecef_from_llh(lat, lon, alt)) - ecef_from_map @ p0
    q = Rotation.from_matrix(ecef_from_map).as_quat()
    return {"position_x": float(t[0]), "position_y": float(t[1]), "position_z": float(t[2]),
            "rotation_x": float(q[0]), "rotation_y": float(q[1]), "rotation_z": float(q[2]), "rotation_w": float(q[3])}


def is_placed(m: dict) -> bool:
    return any(abs(float(m.get(k) or 0) - v) > 1e-9 for k, v in IDENTITY.items())


def map_camera_positions(api: "PlaceframeApi", reconstruction_id: str) -> np.ndarray:
    api._auth()
    r = api.client.get(f"{api.server}/reconstructions/{reconstruction_id}/frame_poses",
                       params={"axis_convention": "OPENCV"})
    r.raise_for_status()
    n = struct.unpack("<I", r.content[:4])[0]
    return np.frombuffer(r.content[4:4 + n * 12], dtype="<f4").reshape(n, 3).astype(float)


def ecef_from_llh(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    la, lo = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
    return (float((n + alt) * np.cos(la) * np.cos(lo)), float((n + alt) * np.cos(la) * np.sin(lo)),
            float((n * (1 - WGS84_E2) + alt) * np.sin(la)))


def ecef_from_enu(lat: float, lon: float) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    east = [-np.sin(lo), np.cos(lo), 0.0]
    north = [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)]
    up = [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    return np.array([east, north, up]).T


def _enu_from_map(positions: np.ndarray, up: str) -> np.ndarray:
    """Rotation taking the map's axes to East/North/Up. With up=auto the vertical axis is the coordinate axis along
    which the camera positions vary least (a walk-through spreads horizontally, not vertically); its sign cannot be
    told from positions alone and is taken as +."""
    if up == "auto":
        spread = positions.std(axis=0) if len(positions) > 2 else np.array([1.0, 1.0, 0.0])
        k = int(np.argmin(spread))
        up = "xyz"[k]
    sign = -1.0 if up.startswith("-") else 1.0
    k = "xyz".index(up[-1])
    u = np.zeros(3)
    u[k] = sign
    e = np.zeros(3)
    e[(k + 1) % 3] = 1.0
    n = np.cross(u, e)
    return np.array([e, n, u])  # rows: map vectors that become E, N, U


def insv_gps(path: Path) -> list[dict]:
    """GPS fixes recorded in an Insta360 .insv (present only when shot from the phone app with GPS on, or from a GPS
    remote). Each fix: t (s), lat, lon, alt (m)."""
    import telemetry_parser  # local import: heavy, and only needed for this command

    fixes = []
    for block in telemetry_parser.Parser(str(path)).telemetry():
        for group, tags in block.items():
            if "gps" not in group.lower():
                continue
            data = tags.get("Data")
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                fix = _fix(row)
                if fix:
                    fixes.append(fix)
    return fixes


def insv_metadata(path: Path) -> dict:
    """Camera model and recording facts from an .insv's metadata, for showing a recording before upload."""
    import telemetry_parser  # local import: heavy, and only needed here

    p = telemetry_parser.Parser(str(path))
    meta = {}
    for block in p.telemetry():
        if "Default" in block and "Metadata" in block["Default"]:
            meta = block["Default"]["Metadata"] or {}
            break
    gps_field = str(meta.get("gps") or "")
    return {"camera": p.model or p.camera, "gps_metadata": bool(gps_field.strip("0")),
            "gps_sources": meta.get("gps_sources") or [], "first_gps_timestamp": meta.get("first_gps_timestamp")}


def _fix(row) -> dict | None:
    if not isinstance(row, dict):
        return None
    keys = {k.lower(): v for k, v in row.items()}
    lat = keys.get("lat") or keys.get("latitude")
    lon = keys.get("lon") or keys.get("lng") or keys.get("longitude")
    if lat is None or lon is None or (abs(float(lat)) < 1e-9 and abs(float(lon)) < 1e-9):
        return None
    return {"t": float(keys.get("timestamp") or keys.get("t") or 0.0), "lat": float(lat), "lon": float(lon),
            "alt": float(keys.get("altitude") or keys.get("alt") or 0.0)}


def median_fix(fixes: list[dict]) -> dict | None:
    if not fixes:
        return None
    return {k: float(np.median([f[k] for f in fixes])) for k in ("lat", "lon", "alt")} | {"n": len(fixes)}


if __name__ == "__main__":
    main()
