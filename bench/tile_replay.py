#!/usr/bin/env python3
"""Replay a recorded phone walk against one map, or against overlapping tiles of it with tile routing.

  single  the whole map; also records the candidates without a prior, for track_analyze.py's reference
  tiled   one tracker for the session; each tracked query goes to the tile the router chooses (core + hysteresis,
          docker/localizer/src/tiles.py); with no prediction (first frame, lost track) every tile is queried

Runs in the fleet worker environment (as bench/localizer_profile.py):

  python bench/tile_replay.py BUNDLE OUT_DIR

BUNDLE: code/pf_localizer, code/core, calibration/global.json, replay/ (session.json, img/), maps/<name>/<map id>/
(map files) with maps/<name>/metrics.json; maps/single for the whole map, maps/tile_* for tiles; tiles.json
(fleet/tiler.py's plan, its tile names = the maps/ names). Writes OUT_DIR/{single,tiled}.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SHA = os.environ.get("LOCALIZER_SHA", "tree-37154ed4b6d1b4193e4b1f69bdb225010b373f50")


def load(bundle: Path, name: str):
    from uuid import UUID

    from core.reconstruction_metrics import ReconstructionMetrics
    from pf_localizer import map as mapmod

    d = bundle / "maps" / name
    map_id = next(p.name for p in d.iterdir() if p.is_dir())

    class NoDownloads:
        def get_paginator(self, _name):
            return type("P", (), {"paginate": lambda self, **_kw: []})()

    metrics = ReconstructionMetrics.model_validate(json.loads((d / "metrics.json").read_text()))
    return map_id, mapmod.load_map(UUID(map_id), NoDownloads(), "unused", d, metrics)


def main() -> None:
    bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bundle / "code"))
    from core.calibration import load_global_calibration
    from core.camera_config import PinholeCameraConfig
    from pf_localizer import localize
    from pf_localizer.tiles import TileRouter
    from pf_localizer.tracked import localize_candidates
    from pf_localizer.tracker import Tracker, pose_from_quaternion

    localize.load_models()
    cal = load_global_calibration(bundle / "calibration/global.json", SHA)
    session = json.loads((bundle / "replay/session.json").read_text())
    plan = json.loads((bundle / "tiles.json").read_text())
    tile_names = [t["name"] for t in plan["tiles"]]
    maps = {name: load(bundle, name) for name in ["single", *tile_names]}
    print({k: len(v[1].ordered_image_ids) for k, v in maps.items()}, flush=True)
    router = TileRouter.from_plan(plan, {n: n for n in tile_names})

    def dump(cs):
        return [{"m": c.map_from_camera.tolist(), "conf": c.confidence, "inl": c.num_inliers, "src": c.source}
                for c in cs]

    frames = []
    for r in session:
        q = r["request"]
        o = q["odometry_from_camera"]
        frames.append((r, pose_from_quaternion([o["translation"][k] for k in "xyz"], [o["rotation"][k] for k in "xyzw"]),
                       PinholeCameraConfig.model_validate(q["camera_config"]), (bundle / "replay/img" / r["image"]).read_bytes()))

    for mode in ("single", "tiled"):
        tracker, current = Tracker(), None
        f_out = (out / f"{mode}.jsonl").open("w")
        t0 = time.time()
        for i, (r, odom, cam, img) in enumerate(frames):
            prior = tracker.predicted_map_from_camera(odom)
            confident = tracker.confident
            t1 = time.time()
            rec: dict = {"i": i, "t": r["received_at"], "odom": odom.tolist(),
                         "prior": prior.tolist() if prior is not None else None}
            if mode == "single":
                m = maps["single"][1]
                plain = localize_candidates(m, cam, img, None, False, SHA, cal)
                cands = localize_candidates(m, cam, img, prior, confident, SHA, cal) if prior is not None else plain
                rec |= {"plain": dump(plain), "with_prior": dump([c for c in cands if c.source == "prior"]),
                        "tiles_queried": 1}
            else:
                tile = router.choose(current, prior)
                queried = [tile] if tile else tile_names
                cands = []
                for name in queried:
                    cands += localize_candidates(maps[name][1], cam, img, prior, confident if tile else False, SHA, cal)
                if tile is None and cands:  # cold start: continue in the tile of the best candidate
                    best = max(cands, key=lambda c: c.confidence)
                    tile = router.choose(None, best.map_from_camera)
                rec |= {"tile": tile, "previous_tile": current, "tiles_queried": len(queried),
                        "candidates": dump(cands)}
                current = tile
            res = tracker.update(odom, cands)
            b = tracker.best
            rec |= {"best_p": b.probability if b else 0, "best_A": b.odom_from_map.tolist() if b else None,
                    "agreed": res.agreed, "ms": round((time.time() - t1) * 1000, 1)}
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            if i % 50 == 0:
                print(mode, i, f"{time.time() - t0:.0f}s", rec.get("tile"), f"p={rec['best_p']:.2f}", flush=True)
        f_out.close()
        print(mode, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
