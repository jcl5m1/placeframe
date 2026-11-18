#!/usr/bin/env python3
"""Autoscaling policies for serving, with GPU start-up delay and attendance that the scaler cannot know.

Assumptions (stated in the cost study):
  * Conference behaviour is assumed, not measured: serving.py's programme (45 min sessions, 15 min breaks), 5% of
    attendees localizing in sessions and 75% in breaks, ramps of 3 and 4 min.
  * The schedule is known in advance (break times), but attendance in each break is not: each break's active share
    is multiplied by a random factor (mean 1, s.d. ATTENDANCE_SD).
  * A new GPU serves START_S after it is requested (measured on Vast: ~2.3 min to a ready worker, plus map
    loading); it is billed from the request, and runs at least MIN_LIFE_S.
  * Each GPU serves GPU_QPS tracked queries/s. Queries above capacity wait in a queue (first in, first out).

Policies (target GPUs are re-computed every `step` seconds):
  study      the cost study's model: each 5-min window gets its own peak x 1.3, at once (no delay, clairvoyant)
  reactive   from the observed peak of the last step x headroom; no schedule knowledge
  schedule   from the expected load (schedule) over the next step x headroom, requested START_S ahead, plus
             reactive correction if the observed load is higher

  python cost/autoscale.py  -> cost/autoscale.json, printed table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import serving as S

GPU_QPS = 8.0
PRICE = 0.15
START_S = 180
MIN_LIFE_S = 600
ATTENDANCE_SD = 0.15
SEEDS = 20
WAIT_LIMIT_S = 0.5


def demand(rate: float, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
    """Queries per second over the day: (actual with random break attendance, expected from the schedule)."""
    t = np.arange(0, S.DAY_H * 3600, 1.0)
    cycle = (S.SESSION_MIN + S.BREAK_MIN) * 60
    base = S.active_fraction(t)
    factor = np.ones_like(t)
    if rng is not None:
        per_break = np.clip(rng.normal(1.0, ATTENDANCE_SD, int(np.ceil(len(t) / cycle)) + 1), 0.5, 1.6)
        factor = per_break[(t // cycle).astype(int)]
    def qps(frac: np.ndarray) -> np.ndarray:
        active = S.ATTENDEES * frac
        starts = np.maximum(np.diff(active, prepend=active[0]), 0) + S.ATTENDEES * S.ACTIVE_SESSION / 600
        return active * rate + starts * S.SEARCH_QUERIES
    actual_frac = S.ACTIVE_SESSION + (base - S.ACTIVE_SESSION) * factor
    actual = qps(actual_frac)
    if rng is not None:
        actual = rng.poisson(actual).astype(float)
    return actual, qps(base)


def simulate(actual: np.ndarray, expected: np.ndarray, policy: str, step: int, headroom: float) -> dict:
    n = len(actual)
    if policy == "study":
        w = 300
        win = actual[: n // w * w].reshape(-1, w).max(axis=1)
        cap = np.repeat(np.ceil(win * 1.3 / GPU_QPS), w)
        cap = np.concatenate([cap, np.full(n - len(cap), cap[-1])])
        billed_s = float(cap.sum())
    else:
        cap = np.zeros(n)
        billed_s = 0.0
        target = int(np.ceil(expected[0] * headroom / GPU_QPS))
        gpus = [(-START_S, -START_S)] * target  # (requested, ready): warm at the start of the day
        for s0 in range(0, n, step):
            seen = actual[max(0, s0 - step):s0].max() if s0 else actual[0]
            need = seen * headroom / GPU_QPS
            if policy == "schedule":
                ahead = expected[s0 + START_S: s0 + START_S + step]
                if len(ahead):
                    need = max(need, ahead.max() * headroom / GPU_QPS)
            target = int(np.ceil(need))
            live = len(gpus)
            if target > live:
                gpus += [(s0, s0 + START_S)] * (target - live)
            elif target < live:  # stop the newest GPUs that have run their minimum life
                keep = sorted(gpus, key=lambda g: g[0])
                removable = [g for g in keep if s0 - g[0] >= MIN_LIFE_S]
                for g in removable[::-1][: live - target]:
                    keep.remove(g)
                    billed_s += s0 - max(0, g[0])
                gpus = keep
            end = min(n, s0 + step)
            for req, rdy in gpus:
                lo = int(max(s0, rdy))
                if lo < end:
                    cap[lo:end] += 1
        billed_s += sum(n - max(0, g[0]) for g in gpus)
    # queue: queries above capacity wait; delay of each second's arrivals ~ backlog / capacity
    backlog, late = 0.0, 0.0
    for i in range(n):
        c = cap[i] * GPU_QPS
        backlog = max(0.0, backlog + actual[i] - c)
        if backlog > 0 and (c == 0 or backlog / c > WAIT_LIMIT_S):
            late += actual[i]
    # load shedding instead: queries above capacity are skipped (the phone keeps tracking on its odometry)
    shed = float(np.maximum(actual - cap * GPU_QPS, 0).sum())
    return {"gpu_hours": billed_s / 3600, "cost": billed_s / 3600 * PRICE, "late_share": late / actual.sum(),
            "shed_share": shed / actual.sum(),
            "peak_gpus": int(cap.max())}


POLICIES = [("study: 5-min windows, peak x1.3, no delay (clairvoyant)", "study", 300, 1.3),
            ("reactive, 5-min steps, x1.3", "reactive", 300, 1.3),
            ("reactive, 1-min steps, x1.3", "reactive", 60, 1.3),
            ("schedule-aware, 1-min steps, x1.3", "schedule", 60, 1.3),
            ("schedule-aware, 1-min steps, x1.2", "schedule", 60, 1.2),
            ("schedule-aware, 1-min steps, x1.15", "schedule", 60, 1.15),
            ("schedule-aware, 1-min steps, x1.05", "schedule", 60, 1.05)]


def main() -> None:
    out = {"assumptions": {"start_s": START_S, "min_life_s": MIN_LIFE_S, "attendance_sd": ATTENDANCE_SD,
                           "gpu_qps": GPU_QPS, "price": PRICE, "wait_limit_s": WAIT_LIMIT_S, "seeds": SEEDS},
           "results": []}
    for attendees in (3000,):
        S.ATTENDEES = attendees
        for rate_name, rate in (("adaptive", 0.19), ("1 Hz", 1.0)):
            runs = {name: [] for name, *_ in POLICIES}
            for seed in range(SEEDS):
                actual, expected = demand(rate, np.random.default_rng(seed))
                for name, pol, step, h in POLICIES:
                    runs[name].append(simulate(actual, expected, pol, step, h))
            for name, rs in runs.items():
                row = {"attendees": attendees, "rate": rate_name, "policy": name,
                       "cost": round(float(np.mean([r["cost"] for r in rs])), 2),
                       "gpu_hours": round(float(np.mean([r["gpu_hours"] for r in rs])), 1),
                       "late_share_pct": round(100 * float(np.mean([r["late_share"] for r in rs])), 2),
                       "late_share_worst_pct": round(100 * float(np.max([r["late_share"] for r in rs])), 2),
                       "shed_share_pct": round(100 * float(np.mean([r["shed_share"] for r in rs])), 2),
                       "shed_share_worst_pct": round(100 * float(np.max([r["shed_share"] for r in rs])), 2)}
                out["results"].append(row)
                print(f"{rate_name:9s} {name:58s} ${row['cost']:7.2f}  late {row['late_share_pct']:5.2f}% "
                      f"(worst {row['late_share_worst_pct']:.2f}%)  shed {row['shed_share_pct']:5.2f}% "
                      f"(worst {row['shed_share_worst_pct']:.2f}%)")
    (Path(__file__).parent / "autoscale.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
