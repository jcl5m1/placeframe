#!/usr/bin/env python3
"""Localize query images against Placeframe map(s) and score them against ground-truth poses.

queries.jsonl lines: {"image": path, "world_from_camera": 4x4} in the same world frame the map was built in
(the ground-truth poses were the mapping priors, and maps are anchored to the priors). Map placement transforms
must be identity (fleet/build.py and the dataset tools publish them that way).

  python datasets/eval_queries.py out/queries.jsonl --map-id <id> [--map-id <id> ...] --camera '<json>'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from add_recording_to_map import PlaceframeApi  # noqa: E402

THRESHOLDS = [(0.05, 5.0), (0.25, 2.0), (0.5, 5.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", type=Path)
    ap.add_argument("--map-id", action="append", required=True)
    ap.add_argument("--camera", required=True, help="camera_config JSON (or path to conversion.json with 'camera')")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cam = json.loads(Path(a.camera).read_text())["camera"] if Path(a.camera).exists() else json.loads(a.camera)
    rows = [json.loads(line) for line in a.queries.read_text().splitlines()]
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    api = PlaceframeApi()
    results = [localize(api, r, a.map_id, cam) for r in rows]
    report = summarize(results)
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"report": report, "results": results}, indent=1))


def localize(api: PlaceframeApi, q: dict, map_ids: list[str], cam: dict) -> dict:
    api._auth()
    t0 = time.monotonic()
    r = api.client.post(api.server + "/localize", headers={"X-Placeframe-Replay": "1"},
                        data={"map_ids": json.dumps(map_ids), "camera_config": json.dumps(cam), "axis_convention": "OPENCV"},
                        files={"image": ("q.jpg", Path(q["image"]).read_bytes(), "image/jpeg")})
    ms = (time.monotonic() - t0) * 1000
    out = {"image": q["image"], "ms": round(ms, 1), "status": r.status_code}
    if r.status_code >= 400 or not r.json():
        return out
    best = max(r.json(), key=lambda m: (m.get("metrics") or {}).get("confidence_loose") or 0)
    tr = best["camera_from_map_transform"]
    R_cw = Rotation.from_quat([tr["rotation"][k] for k in "xyzw"]).as_matrix()
    t_cw = np.array([tr["translation"][k] for k in "xyz"])
    gt = np.array(q["world_from_camera"])
    est_center, est_R_wc = -R_cw.T @ t_cw, R_cw.T
    out |= {"map_id": best["id"], "dt_m": float(np.linalg.norm(est_center - gt[:3, 3])),
            "dr_deg": float(np.degrees(Rotation.from_matrix(est_R_wc.T @ gt[:3, :3]).magnitude())),
            "confidence": (best.get("metrics") or {}).get("confidence_loose")}
    return out


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "dt_m" in r]
    dt = np.array([r["dt_m"] for r in ok]) if ok else np.array([np.inf])
    dr = np.array([r["dr_deg"] for r in ok]) if ok else np.array([np.inf])
    n = len(results)
    rep = {"queries": n, "localized": len(ok), "median_dt_m": round(float(np.median(dt)), 3),
           "median_dr_deg": round(float(np.median(dr)), 2),
           "latency_ms_p50": round(float(np.median([r["ms"] for r in results])), 0)}
    for t, d in THRESHOLDS:
        rep[f"recall@{int(t * 100)}cm,{d:g}deg"] = round(sum(1 for r in ok if r["dt_m"] <= t and r["dr_deg"] <= d) / n, 3)
    return rep


if __name__ == "__main__":
    main()
