#!/usr/bin/env python3
"""Cost stack-ups for the cost study: building a map (one time, by stage) and serving it (per event day).

  python cost/stackup.py   -> cost/4_build_cost_by_stage.png, cost/5_serving_cost_by_use.png, cost/stackup.json

Build: stage times measured on the home RTX 3090 (cost study v2: the full house walk, and 4 copies joined into one
map, timed from its clean restart); the venue is projected at the joined map's seconds per image. Preprocessing
(calibration, odometry, stitching; CPU) scales with recording length: 25 min per 14 min of recording. All time is
priced as rented RTX 3090 time ($0.15/h); storage on Cloudflare R2 adds cents per month.

Serving: serving.py's venue load model (8 h day, 45 min sessions, 15 min breaks), RTX 3090s at 8 tracked q/s,
autoscaled every 5 min to the window's peak with 30% headroom. GPU-hours split into tracking queries, place-search
queries when the app starts, and headroom + idle (provisioned but unused). With R2, map loads and point-cloud
downloads have no egress cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import serving as S  # noqa: E402

HERE = Path(__file__).parent
PRICE = 0.15  # $/h, rented RTX 3090 (interruptible)
GPU_QPS = 8.0
# Reference palette (light), validated with the dataviz validator: categorical slots 1..6, fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# ---------------------------------------------------------------- build
PRE_MIN_PER_WALK_MIN = 25 / 14
STAGES = ["Preprocessing (CPU)", "Feature extraction", "Compression training", "Matching",
          "Geometric verification", "Mapping"]
# seconds per stage on the home RTX 3090 (cost study v2 results), and minutes of walking in the capture
HOUSE = {"images": 2268, "walk_min": 192.25 / 60, "s": [None, 166.0, 10.0, 185.4, 35.2, 616.6]}
JOINED = {"images": 9072, "walk_min": 769.0 / 60, "s": [None, 734.5, 40.2, 782.1, 155.8, 476.7]}
VENUE_IMAGES = 19_700
VENUE_WALK_MIN = 28.0


def build_rows() -> list[dict]:
    rows = []
    venue_scale = VENUE_IMAGES / JOINED["images"]
    specs = [("House, full walk (measured)", HOUSE["images"], HOUSE["walk_min"], HOUSE["s"][1:]),
             ("4 houses joined (measured)", JOINED["images"], JOINED["walk_min"], JOINED["s"][1:]),
             ("10,000 m² venue (projected)", VENUE_IMAGES, VENUE_WALK_MIN, [x * venue_scale for x in JOINED["s"][1:]])]
    for label, images, walk, gpu_s in specs:
        minutes = [walk * PRE_MIN_PER_WALK_MIN] + [x / 60 for x in gpu_s]
        rows.append({"label": label, "images": images, "minutes": minutes, "total_min": sum(minutes),
                     "cost": sum(minutes) / 60 * PRICE})
    return rows


def plot_build(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        left = 0.0
        for k, m in enumerate(r["minutes"]):
            ax.barh(yi, m, left=left, height=0.5, color=SLOTS[k], edgecolor=SURFACE, linewidth=2,
                    label=STAGES[k] if yi == y[0] else None)
            if m >= 0.11 * rows[-1]["total_min"]:  # direct label on segments wide enough to hold it
                ax.text(left + m / 2, yi, f"{m:.0f} min", ha="center", va="center", fontsize=8, color=INK)
            left += m
        ax.text(left + rows[-1]["total_min"] * 0.01, yi, f"{r['total_min']:.0f} min  ·  ${r['cost']:.2f}",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [f"{r['label']}\n{r['images']:,} images" for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, rows[-1]["total_min"] * 1.22)
    ax.set_xlabel("Minutes of rented RTX 3090 time (one host, $0.15/h)", fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_title("Building a map: time and cost by stage (one time per venue map)", loc="left", fontsize=11,
                 color=INK, pad=28)
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8, labelcolor=INK,
              handlelength=1.2, columnspacing=1.2)
    fig.text(0.01, 0.01, "Storage on Cloudflare R2: ~3.2 GB for the venue (map + capture), ~\\$0.05/month. "
             "Capture walking (~28 min for the venue) is staff time, not shown.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- serving
RATES = [("Adaptive\n(0.19 q/s)", 0.19), ("0.5 Hz", 0.5), ("1 Hz", 1.0), ("2 Hz", 2.0)]
ATTENDEES = [1000, 3000, 10000]
USES = ["Tracking queries", "Place search at app start (<5%)", "Autoscaling headroom + idle GPUs"]


def serving_rows() -> dict:
    out = {}
    for att in ATTENDEES:
        S.ATTENDEES = att
        for label, rate in RATES:
            t, q = S.load(rate)
            track = att * S.active_fraction(t) * rate
            g = S.gpus_needed(q, GPU_QPS)
            prov_h = float(g.sum() * S.SCALE_WINDOW_S / 3600)
            track_h = float(track.sum() / GPU_QPS / 3600)
            search_h = float((q - track).sum() / GPU_QPS / 3600)
            out[(att, label)] = {"attendees": att, "rate": label.replace("\n", " "), "peak_gpus": int(g.max()),
                                 "gpu_hours": round(prov_h, 1),
                                 "cost": [track_h * PRICE, search_h * PRICE, (prov_h - track_h - search_h) * PRICE],
                                 "total": prov_h * PRICE}
    S.ATTENDEES = 3000
    return out


def plot_serving(rows: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), dpi=150, facecolor=SURFACE)
    x = np.arange(len(RATES))
    for ax, att in zip(axes, ATTENDEES):
        ax.set_facecolor(SURFACE)
        top = max(rows[(att, lab)]["total"] for lab, _ in RATES)
        for xi, (lab, _) in zip(x, RATES):
            r, bottom = rows[(att, lab)], 0.0
            for k, c in enumerate(r["cost"]):
                ax.bar(xi, c, bottom=bottom, width=0.55, color=SLOTS[k], edgecolor=SURFACE, linewidth=2,
                       label=USES[k] if (att, lab) == (ATTENDEES[0], RATES[0][0]) else None)
                bottom += c
            ax.text(xi, bottom + top * 0.02, f"${r['total']:,.0f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK)
        ax.set_xticks(x, [lab for lab, _ in RATES], fontsize=8, color=INK2)
        ax.set_ylim(0, top * 1.15)
        ax.set_title(f"{att:,} attendees", fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    axes[0].set_ylabel("GPU cost per event day (RTX 3090 at $0.15/h)", fontsize=8.5, color=INK2)
    fig.suptitle("Serving a 10,000 m² venue for one 8 h day: GPU cost by use, by query rate and attendees",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.945),
               frameon=False, fontsize=8, labelcolor=INK, handlelength=1.2)
    fig.text(0.01, 0.01, "Each panel has its own scale. Query rate = tracked queries per walking user. "
             "With R2, map loads and point-cloud downloads cost \\$0; with S3/GCS/Azure and marketplace GPUs, map "
             "egress adds ~\\$70-93/day at 3,000 attendees. GCP L4s: ~1.45x the GPUs at ~4.7x the price.",
             fontsize=7.2, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    b = build_rows()
    s = serving_rows()
    plot_build(b, HERE / "4_build_cost_by_stage.png")
    plot_serving(s, HERE / "5_serving_cost_by_use.png")
    data = {"build": [dict(r, stages=STAGES) for r in b], "serving": list(s.values()), "uses": USES,
            "price_per_gpu_hour": PRICE, "gpu_qps": GPU_QPS}
    (HERE / "stackup.json").write_text(json.dumps(data, indent=1))
    for r in b:
        print(r["label"], f"{r['total_min']:.0f} min ${r['cost']:.2f}", [round(m, 1) for m in r["minutes"]])


if __name__ == "__main__":
    main()
