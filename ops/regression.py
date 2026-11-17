#!/usr/bin/env python3
"""Localization regression check: fixed query sets against fixed maps, compared with a saved baseline.

Suites (the same queries and maps every run, so results are comparable):
  phone_walk      recorded phone requests (C# client) replayed against the house map
  heads, stairs   7-Scenes test queries against their maps, scored against ground truth

  python ops/regression.py                 # run, compare with the baseline, write the latest results
  python ops/regression.py --set-baseline  # run and save as the new baseline (after an intended change)

Exit status 1 and a watchdog incident (dashboard banner) when a suite got worse beyond its tolerance. Run nightly at
04:00 and after any change to the reconstructor or localizer.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "datasets"))
import eval_queries  # noqa: E402
import replay_localizations as rl  # noqa: E402
from add_recording_to_map import PlaceframeApi  # noqa: E402

OUT = Path.home() / "pf_watchdog" / "regression"
DATASETS = Path("/mnt/hdd12t/placeframe_scratch/datasets/7scenes")
HOUSE_MAP = "913aee92-baec-4cf5-9ce0-2f2a075a858c"  # re-registered house map (2026-11-17)
SUITES = {
    "heads": {"map": "22699ebb-a9e8-45d0-972b-373d4f739df0", "data": DATASETS / "heads_pf"},
    "stairs": {"map": "8fb152c1-b5f4-4c39-b90f-246c91fa0302", "data": DATASETS / "stairs_pf"},
}
# A suite regresses when a metric moves the wrong way by more than this.
TOLERANCE = {"localized_frac": 0.05, "recall@25cm,2deg": 0.05, "recall@50cm,5deg": 0.05, "median_dt_m": 0.03,
             "latency_ms_p50": 1.5}  # latency: ratio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-baseline", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    api = PlaceframeApi()
    results = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "suites": {"phone_walk": phone_walk(api)}}
    for name, s in SUITES.items():
        rows = [json.loads(line) for line in (s["data"] / "queries.jsonl").read_text().splitlines()]
        cam = json.loads((s["data"] / "conversion.json").read_text())["camera"]
        rep = eval_queries.summarize([eval_queries.localize(api, r, [s["map"]], cam) for r in rows])
        results["suites"][name] = rep | {"localized_frac": round(rep["localized"] / rep["queries"], 3)}
    (OUT / "latest.json").write_text(json.dumps(results, indent=1))
    base_path = OUT / "baseline.json"
    if a.set_baseline or not base_path.exists():
        base_path.write_text(json.dumps(results, indent=1))
        print(json.dumps(results, indent=1))
        print("baseline saved")
        return
    worse = compare(json.loads(base_path.read_text()), results)
    print(json.dumps(results, indent=1))
    if worse:
        print("REGRESSIONS:\n" + "\n".join(worse))
        report(worse)
        raise SystemExit(1)
    print("no regressions")


def phone_walk(api: PlaceframeApi) -> dict:
    reqs = rl.load_requests(rl.ROOT, None, "csharp")
    ok, ms, conf = 0, [], []
    for r in reqs:
        q = r["request"]
        api._auth()
        t0 = time.monotonic()
        resp = api.client.post(api.server + "/localize", headers={"X-Placeframe-Replay": "1"},
                               data={"map_ids": json.dumps([HOUSE_MAP]), "camera_config": json.dumps(q["camera_config"]),
                                     "axis_convention": q["axis_convention"]},
                               files={"image": ("q.jpg", r["_path"].read_bytes(), "image/jpeg")})
        ms.append((time.monotonic() - t0) * 1000)
        if resp.status_code < 400 and resp.json():
            ok += 1
            conf.append(max((m.get("metrics") or {}).get("confidence_loose") or 0 for m in resp.json()))
    return {"queries": len(reqs), "localized": ok, "localized_frac": round(ok / max(1, len(reqs)), 3),
            "median_confidence": round(statistics.median(conf), 3) if conf else None,
            "latency_ms_p50": round(statistics.median(ms)) if ms else None}


def report(worse: list[str]) -> None:
    """A regression appears in the mapper dashboard's banner (it reads the watchdog's unresolved incidents)."""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "localization-regression",
           "summary": f"localization got worse: {worse[0]}" + (f" (+{len(worse) - 1} more)" if len(worse) > 1 else ""),
           "status": "needs_attention", "evidence": "\n".join(worse) + f"\nbaseline/latest: {OUT}"}
    with (Path.home() / "pf_watchdog" / "incidents.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def compare(base: dict, cur: dict) -> list[str]:
    worse = []
    for suite, b in base["suites"].items():
        c = cur["suites"].get(suite, {})
        for metric, tol in TOLERANCE.items():
            if b.get(metric) is None or c.get(metric) is None:
                continue
            if metric == "median_dt_m" and c[metric] > b[metric] + tol:
                worse.append(f"{suite}: median error {b[metric]} -> {c[metric]} m")
            elif metric == "latency_ms_p50" and c[metric] > b[metric] * tol:
                worse.append(f"{suite}: latency p50 {b[metric]} -> {c[metric]} ms")
            elif metric not in ("median_dt_m", "latency_ms_p50") and c[metric] < b[metric] - tol:
                worse.append(f"{suite}: {metric} {b[metric]} -> {c[metric]}")
    return worse


if __name__ == "__main__":
    main()
