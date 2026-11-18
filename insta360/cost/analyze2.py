#!/usr/bin/env python3
"""Cost study v2 report: map building with the current pipeline (v1 for comparison), extrapolation to a 10,000 m^2
venue, a rented right-sized GPU, and map serving (serving.py). Writes report_v2/index.html, charts and a concise
markdown summary (report_v2/COST_STUDY.md).

  python analyze2.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
V1, V2, OUT = ROOT / "results", ROOT / "results_v2", ROOT / "report_v2"
VAST = ROOT / "vast_test.json"
FAIRHILLS = {"variant": "Fairhills (3 recordings, merged)", "images": 8370, "total_s": 2075.0, "mapper": "prior poses"}
VENUE_M2, OPEN_M2_PER_S = 10_000, 6.0  # open halls: ~6 m walking lanes at ~1 m/s
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"]


def load(d: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def clean_stages(r: dict) -> list[dict]:
    """Stages of the last attempt: a build interrupted by an outage was re-leased and started over (its record spans
    both); only the final, uninterrupted attempt counts."""
    st = r["stages"]
    starts = [i for i, s in enumerate(st) if i and s["status"] in ("queued", "downloading") and st[i - 1]["status"] not in ("queued", "downloading")]
    return st[starts[-1]:] if starts else st


def total_seconds(r: dict) -> float:
    st = clean_stages(r)
    return st[-1]["t"] - st[0]["t"]


def stage_seconds(r: dict) -> dict[str, float]:
    st = clean_stages(r)
    out: dict[str, float] = {}
    for a, b in zip(st, st[1:]):
        out[a["status"]] = out.get(a["status"], 0) + b["t"] - a["t"]
    return out


def mapping_s(r: dict) -> float:
    return stage_seconds(r).get("reconstructing", 0.0)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    v1, v2 = load(V1), load(V2)
    rows = []
    for name, r in v2.items():
        old = v1.get(name) or v1.get({"j2": "x2", "j4": "x4"}.get(name, ""))
        rows.append({"variant": name, "images": r["images"], "v2_total_s": total_seconds(r), "v2_mapping_s": mapping_s(r),
                     "v1_total_s": old["total_s"] if old else None, "v1_mapping_s": mapping_s(old) if old else None,
                     "v1_variant": old["variant"] if old else None, "registered": (r.get("metrics") or {}).get("registered_images"),
                     "points": (r.get("metrics") or {}).get("num_3d_points"), "storage": r.get("recon_storage", "").split("\t")[0],
                     "stages": stage_seconds(r), "occupied_m2": r.get("occupied_area_m2")})
    # Seconds per image for the joined (prior-pose) builds: what a venue built from merged recordings costs.
    joined = [x for x in rows if x["variant"].startswith("j")]
    per_img = [x["v2_total_s"] / x["images"] for x in joined] + [FAIRHILLS["total_s"] / FAIRHILLS["images"]]
    s_per_image = float(np.median(per_img)) if per_img else None
    house = v2.get("f100")
    venue = None
    if house and s_per_image:
        img_per_s = house["images"] / house["recording_s"]
        rec_s = VENUE_M2 / OPEN_M2_PER_S
        images = rec_s * img_per_s
        venue = {"recording_min": rec_s / 60, "images": images, "build_h_3090": images * s_per_image / 3600,
                 "s_per_image": s_per_image}
    vast = json.loads(VAST.read_text()) if VAST.exists() else None
    serving = json.loads((ROOT / "report" / "serving.json").read_text()) if (ROOT / "report" / "serving.json").exists() else None
    report = {"builds": rows, "fairhills": FAIRHILLS, "venue_build": venue, "vast": vast, "serving": serving}
    (OUT / "cost_study_v2.json").write_text(json.dumps(report, indent=1, default=float))
    charts(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "serving"}, indent=1, default=float)[:3000])


def charts(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, label, c in (("v1_total_s", "v1: incremental mapper, CPU BA (2x/4x were separate copies 300 m apart)", C[1]),
                          ("v2_total_s", "v2: GPU BA; 2x/4x joined into one map (prior-pose mapper)", C[0])):
        pts = sorted((x["images"], x[key] / 60) for x in rows if x.get(key))
        if pts:
            ax.plot(*zip(*pts), "o-", color=c, label=label)
            for i, m in pts:
                ax.annotate(f"{m:.0f} min", (i, m), textcoords="offset points", xytext=(4, 4), fontsize=7, color=c)
    ax.plot([FAIRHILLS["images"]], [FAIRHILLS["total_s"] / 60], "s", color=C[2], label="Fairhills: 3 real recordings merged (v2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("map images")
    ax.set_ylabel("server build time on one RTX 3090 (min)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.set_title("Build time vs map size: before and after the pipeline changes", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "1_build_time.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
