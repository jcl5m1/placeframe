#!/usr/bin/env python3
"""Run a tiled map build on the Placeframe queue: submit every tile as its own reconstruction job, wait,
publish each tile as a localization map, and record timings. Any number of workers can serve the queue
(the local reconstructor, extra local containers, rented instances) — they all pull leases from the API.

  python fleet/build.py submit tiles_dir --name house-750      # upload tiles + queue jobs -> tiles_dir/build.json
  python fleet/build.py wait tiles_dir                         # poll until every tile is terminal; timings
  python fleet/build.py publish tiles_dir [--active]           # one localization map per tile (inactive default)
  python fleet/build.py status tiles_dir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from add_recording_to_map import PlaceframeApi  # noqa: E402
from upload_to_placeframe import choose_reconstruction_options  # noqa: E402

TERMINAL = {"succeeded", "failed", "cancelled"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["submit", "wait", "publish", "status"])
    ap.add_argument("tiles", type=Path)
    ap.add_argument("--name", help="build name (submit)")
    ap.add_argument("--options", type=Path, help="reconstruction options JSON (default: chosen per tile)")
    ap.add_argument("--active", action="store_true", help="publish tile maps as active (visible to clients)")
    a = ap.parse_args()
    api = PlaceframeApi()
    {"submit": submit, "wait": wait, "publish": publish, "status": status}[a.action](api, a)


def submit(api: PlaceframeApi, a) -> None:
    plan = json.loads((a.tiles / "tiles.json").read_text())
    build_path = a.tiles / "build.json"
    build = json.loads(build_path.read_text()) if build_path.exists() else {"name": a.name or a.tiles.name, "tiles": {}}
    fixed = json.loads(a.options.read_text()) if a.options else None
    for t in plan["tiles"]:
        if t["name"] in build["tiles"]:
            continue  # resumable: already submitted
        tar = Path(t["capture"])
        options = fixed or choose_reconstruction_options(tar, {"pose_source": "tile of posed capture", "metric": True})[0]
        cs = api.upload_capture(tar, f"{build['name']} {t['name']}")
        rid = api.post("/reconstructions", {"create": {"capture_session_id": cs}, "options": options})["id"]
        build["tiles"][t["name"]] = {"capture_session_id": cs, "reconstruction_id": rid, "images": t["images"],
                                     "submitted": time.time()}
        build_path.write_text(json.dumps(build, indent=1))
        print(f"{t['name']}: {t['images']} images -> reconstruction {rid}")


def wait(api: PlaceframeApi, a) -> None:
    build = json.loads((a.tiles / "build.json").read_text())
    while True:
        rows = _rows(api, build)
        if all(r["status"] in TERMINAL for r in rows.values()):
            break
        time.sleep(30)
    _report(build, rows, a.tiles)


def status(api: PlaceframeApi, a) -> None:
    build = json.loads((a.tiles / "build.json").read_text())
    _report(build, _rows(api, build), a.tiles)


def publish(api: PlaceframeApi, a) -> None:
    build_path = a.tiles / "build.json"
    build = json.loads(build_path.read_text())
    for name, t in build["tiles"].items():
        if t.get("map_id"):
            continue
        rec = api.get(f"/reconstructions/{t['reconstruction_id']}")
        if rec["status"] != "succeeded":
            print(f"{name}: not published ({rec['status']})")
            continue
        # Tiles keep the capture's world frame, so every tile map is placed with the identity transform.
        body = {"position_x": 0.0, "position_y": 0.0, "position_z": 0.0, "rotation_x": 0.0, "rotation_y": 0.0,
                "rotation_z": 0.0, "rotation_w": 1.0, "color": 0, "reconstruction_id": t["reconstruction_id"],
                "name": f"{build['name']} {name}", "active": a.active}
        t["map_id"] = api.post("/localization-maps", body)["id"]
        build_path.write_text(json.dumps(build, indent=1))
        print(f"{name}: map {t['map_id']}")
    print("map ids:", ",".join(t["map_id"] for t in build["tiles"].values() if t.get("map_id")))


def _rows(api: PlaceframeApi, build: dict) -> dict:
    return {name: api.get(f"/reconstructions/{t['reconstruction_id']}") for name, t in build["tiles"].items()}


def _report(build: dict, rows: dict, tiles_dir: Path) -> None:
    from datetime import datetime  # noqa: PLC0415 — keeps the CLI import light

    def ts(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None

    out = {}
    for name, r in rows.items():
        m = (r.get("manifest") or {}).get("metrics") or {}
        out[name] = {"status": r["status"], "images": build["tiles"][name]["images"],
                     "registered": m.get("registered_images"), "points": m.get("num_3d_points"),
                     "created": ts(r.get("created_at")), "finished": ts(r.get("updated_at")) if r["status"] in TERMINAL else None}
        print(f"{name}: {r['status']:12s} images {out[name]['images']:5d} registered {out[name]['registered']} "
              f"points {out[name]['points']}")
    done = [v for v in out.values() if v["finished"]]
    if done and len(done) == len(out):
        span = max(v["finished"] for v in out.values()) - min(v["created"] for v in out.values())
        print(f"wall clock from first submit to last finish: {span / 60:.1f} min")
    (tiles_dir / "build_status.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
