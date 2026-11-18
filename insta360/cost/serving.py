#!/usr/bin/env python3
"""Map serving cost model: a 10,000 m^2 hotel conference venue, 3,000 attendees localizing while they walk.

Load: an 8 h programme of 45 min sessions and 15 min breaks. During sessions few people walk (5% active); in
breaks most do (75%), ramping up over the first 3 min of the break and down over its last 4 min. A user who becomes
active first searches (about SEARCH_QUERIES queries at 2 Hz until the place is confirmed), then tracks at the
policy's rate. Query cost: measured GPU time per tracked query on an RTX 3090 (see GPU_QPS). GPUs autoscale every
5 min to that window's peak with 30% headroom.

Storage and data movement: the venue map (size from the rebuilt Fairhills map per image), loaded onto each GPU
server as it starts, plus the point cloud the app downloads per session. Compared across object stores, with GPUs
rented from a marketplace (cross-provider egress) or in the same cloud as the store (no egress).

  python serving.py [--gpu-qps 6.0] -> report/serving.json, report/5_serving_load.png, report/6_serving_cost.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "report"
ATTENDEES = 3000
AREA_M2 = 10_000
DAY_H = 8
SESSION_MIN, BREAK_MIN = 45, 15
ACTIVE_SESSION, ACTIVE_BREAK = 0.05, 0.75
RAMP_UP_MIN, RAMP_DOWN_MIN = 3, 4
SEARCH_QUERIES = 10  # queries to confirm the place after the app starts (2 Hz until the tracker is confident)
# Per active user while tracking (queries/s): fixed rates, and the adaptive policy measured on a 5-minute recorded
# walk (every 5 s while queries agree; every frame after a disagreement or an odometry jump).
POLICIES = {"fixed 2 Hz": 2.0, "fixed 1 Hz": 1.0, "adaptive (measured)": 0.19}
HEADROOM = 1.3
SCALE_WINDOW_S = 300

# $/GPU-hour, Nov 2026. Marketplace prices from research/rentable_compute.md; cloud on-demand list prices.
GPU_PRICES = {"Vast.ai RTX 3090 interruptible": 0.15, "Vast.ai RTX 4090 interruptible": 0.30,
              "GCP L4 on-demand (g2-standard-4)": 0.71, "AWS A10G on-demand (g5.xlarge)": 1.01}
# Object storage: $/GB-month, internet egress $/GB, and whether egress to a marketplace GPU is billed.
STORES = {"Cloudflare R2": (0.015, 0.0), "AWS S3 Standard": (0.023, 0.09), "GCS Standard": (0.020, 0.12),
          "Azure Blob Hot": (0.018, 0.087)}
MAP_MB_PER_IMAGE = 501 / 8370  # rebuilt Fairhills map: 501 MiB for 8,370 images
CAPTURE_MB_PER_IMAGE = 0.106  # capture tar kept for rebuilds (house: 240 MB / 2,268 images)
POINTCLOUD_MB = 1.0  # what the app downloads per session to draw the map
QUERY_KB = 45  # 640x480 JPEG


def active_fraction(t_s: np.ndarray) -> np.ndarray:
    cycle = (SESSION_MIN + BREAK_MIN) * 60
    t = t_s % cycle
    a = np.full_like(t, ACTIVE_SESSION, dtype=float)
    b = t - SESSION_MIN * 60  # seconds into the break
    in_break = b >= 0
    up = np.clip(b / (RAMP_UP_MIN * 60), 0, 1)
    down = np.clip((BREAK_MIN * 60 - b) / (RAMP_DOWN_MIN * 60), 0, 1)
    a[in_break] = ACTIVE_SESSION + (ACTIVE_BREAK - ACTIVE_SESSION) * np.minimum(up, down)[in_break]
    return a


def load(rate: float, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, DAY_H * 3600, dt)
    active = ATTENDEES * active_fraction(t)
    starts = np.maximum(np.diff(active, prepend=active[0]), 0) / dt + ATTENDEES * ACTIVE_SESSION / 600
    qps = active * rate + starts * SEARCH_QUERIES  # session-time churn: 1/10 of the active restart every minute
    return t, qps


def gpus_needed(qps: np.ndarray, gpu_qps: float) -> np.ndarray:
    n = int(SCALE_WINDOW_S)
    windows = qps[: len(qps) // n * n].reshape(-1, n).max(axis=1)
    return np.ceil(windows * HEADROOM / gpu_qps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-qps", type=float, default=5.0, help="tracked queries/s one RTX 3090 serves")
    ap.add_argument("--venue-images", type=float, default=19_662, help="map images for the venue (build study)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    result = {"inputs": {"attendees": ATTENDEES, "area_m2": AREA_M2, "day_h": DAY_H, "gpu_qps": a.gpu_qps,
                         "venue_images": a.venue_images, "policies_qps_per_user": POLICIES}, "policies": {}}
    curves = {}
    for name, rate in POLICIES.items():
        t, qps = load(rate)
        g = gpus_needed(qps, a.gpu_qps)
        gpu_h = float(g.sum() * SCALE_WINDOW_S / 3600)
        starts = int(np.sum(np.maximum(np.diff(np.concatenate([[0], g])), 0)))  # instances started over the day
        result["policies"][name] = {"queries_per_day": float(qps.sum()), "peak_qps": float(qps.max()),
                                    "mean_qps": float(qps.mean()), "peak_gpus": int(g.max()), "gpu_hours": gpu_h,
                                    "instance_starts": starts,
                                    "gpu_cost": {p: gpu_h * price for p, price in GPU_PRICES.items()}}
        curves[name] = (t, qps, g)
    map_gb = a.venue_images * MAP_MB_PER_IMAGE / 1024
    stored_gb = a.venue_images * (MAP_MB_PER_IMAGE + CAPTURE_MB_PER_IMAGE) / 1024
    sessions = ATTENDEES * 3  # app sessions per attendee per day (opens it in about three breaks)
    adaptive = result["policies"]["adaptive (measured)"]
    data = {"map_GB": map_gb, "stored_GB": stored_gb,
            "map_loads_GB_per_day": map_gb * adaptive["instance_starts"],
            "pointcloud_GB_per_day": sessions * POINTCLOUD_MB / 1024,
            "query_upload_GB_per_day": adaptive["queries_per_day"] * QUERY_KB / 1024 / 1024}
    result["data"] = data
    result["storage"] = {}
    for store, (gb_month, egress) in STORES.items():
        monthly_storage = stored_gb * gb_month
        # GPUs on a marketplace pull the map across providers; phones download point clouds from the store.
        day_egress_marketplace = (data["map_loads_GB_per_day"] + data["pointcloud_GB_per_day"]) * egress
        day_egress_same_cloud = data["pointcloud_GB_per_day"] * egress
        result["storage"][store] = {"storage_per_month": monthly_storage,
                                    "egress_per_day_gpus_on_marketplace": day_egress_marketplace,
                                    "egress_per_day_gpus_in_same_cloud": day_egress_same_cloud}
    (OUT / "serving.json").write_text(json.dumps(result, indent=1))
    charts(curves, result, a.gpu_qps)
    for name, r in result["policies"].items():
        print(f"{name:22s} queries/day {r['queries_per_day']:>12,.0f}  peak {r['peak_qps']:7.0f} q/s  "
              f"peak GPUs {r['peak_gpus']:4d}  GPU-h {r['gpu_hours']:7.1f}  "
              + "  ".join(f"{k.split(' ')[0]} {k.split(' ')[1]} ${v:,.0f}" for k, v in r["gpu_cost"].items()))
    print(json.dumps(data, indent=1))
    for s, v in result["storage"].items():
        print(f"{s:18s} storage ${v['storage_per_month']:.2f}/mo  egress/day: marketplace GPUs "
              f"${v['egress_per_day_gpus_on_marketplace']:.2f}, same-cloud GPUs ${v['egress_per_day_gpus_in_same_cloud']:.2f}")


def charts(curves, result, gpu_qps) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#eb6834", "#2a78d6", "#1baf7a"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for (name, (t, qps, g)), c in zip(curves.items(), colors):
        h = t / 3600
        ax[0].plot(h, qps, color=c, lw=1, label=name)
        ax[1].step(np.arange(len(g)) * SCALE_WINDOW_S / 3600, g, where="post", color=c, lw=1.4, label=name)
    ax[0].set_ylabel("localization queries / s")
    ax[0].set_yscale("log")
    ax[1].set_ylabel(f"RTX 3090s (autoscaled, {gpu_qps:g} q/s each)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("hours into the programme (45 min sessions, 15 min breaks)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"{ATTENDEES:,} attendees, {AREA_M2:,} m² venue: localization load over a day", fontsize=11)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "5_serving_load.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = list(result["policies"])
    prices = list(GPU_PRICES)
    w = 0.8 / len(prices)
    for i, p in enumerate(prices):
        vals = [result["policies"][n]["gpu_cost"][p] for n in names]
        bars = ax.bar(np.arange(len(names)) + i * w, vals, w, label=p, color=plt.cm.Blues(0.35 + 0.17 * i))
        ax.bar_label(bars, labels=[f"${v:,.0f}" for v in vals], fontsize=7, padding=2)
    ax.set_xticks(np.arange(len(names)) + w * (len(prices) - 1) / 2, names)
    ax.set_ylabel("GPU cost per event day (US$)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Serving cost per day by query policy and GPU source (storage and egress: see table)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "6_serving_cost.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
