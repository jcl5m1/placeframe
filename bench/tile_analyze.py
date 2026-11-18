#!/usr/bin/env python3
"""Score a tiled replay (bench/tile_replay.py) against the single map, with track_analyze.py's reference.

The reference alignment per frame comes from the single-map replay's candidates (track_analyze.references).
Both modes are scored against it; for the tiled mode we also count tile switches, cold starts (queries sent to
every tile) and the jump of the served pose at each switch.

  python bench/tile_analyze.py OUT_DIR       (holds single.jsonl and tiled.jsonl)
"""
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_analyze as TA  # noqa: E402


def main() -> None:
    d = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(d / "single.jsonl", Path(tmp) / "candidates.jsonl")
        single = TA.load(Path(tmp))
    refs = TA.references(single)
    tiled = [json.loads(line) for line in (d / "tiled.jsonl").open()]
    for f, s in zip(tiled, single):
        f["odom"] = np.array(f["odom"])
        assert np.allclose(f["odom"], s["odom"])
    served = lambda frames: [(f, np.array(f["best_A"]) if f["best_A"] else None, f["best_p"]) for f in frames]  # noqa: E731
    res = {"referenced_frames": sum(r is not None for r, _ in refs)}
    for name, frames in (("single map", single), ("tiles, routed", tiled)):
        shown, correct, wrong = TA.evaluate(name, served(frames), refs)
        res[name] = {"shown": shown, "within_0.5m": correct, "wrong_over_1m": wrong,
                     "ms_mean": round(float(np.mean([f["ms"] for f in frames])), 1)}

    queried = Counter(f["tiles_queried"] for f in tiled)
    switches = [i for i, f in enumerate(tiled) if f["previous_tile"] and f["tile"] != f["previous_tile"]]
    jumps = []
    for i in switches:
        a, b = tiled[i - 1]["best_A"], tiled[i]["best_A"]
        if a and b:
            jumps.append(TA.agree(np.array(a), np.array(b), tiled[i]["odom"])[0])
    # Error of the served pose near switches (±5 frames), against the reference
    near = sorted({j for i in switches for j in range(max(0, i - 5), min(len(tiled), i + 6))})
    near_err = [TA.agree(np.array(tiled[j]["best_A"]), refs[j][0], tiled[j]["odom"])[0]
                for j in near if tiled[j]["best_A"] and refs[j][0] is not None and tiled[j]["best_p"] >= 0.8]
    res["tiled"] = {"frames_per_tile": dict(Counter(f["tile"] for f in tiled)),
                    "tiles_queried": {str(k): v for k, v in sorted(queried.items())},
                    "switches": len(switches), "switch_frames": switches,
                    "served_pose_jump_at_switch_m": [round(x, 3) for x in jumps],
                    "error_near_switches_m": {"n": len(near_err),
                                              "median": round(float(np.median(near_err)), 3) if near_err else None,
                                              "max": round(float(np.max(near_err)), 3) if near_err else None}}
    print(json.dumps(res, indent=1))
    (d / "analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
