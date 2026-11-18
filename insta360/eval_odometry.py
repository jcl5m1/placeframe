#!/usr/bin/env python3
"""Score an open-loop trajectory against the SfM map: anchor on the first N seconds only, then report drift.

  uv run eval_odometry.py traj.tum [more.tum ...] --map ~/pf_x5/viewer/map.json --label-shift 0.234
  (TUM: time x y z qx qy qz qw; time = video seconds. map.json cams carry converter labels in ms.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectories", nargs="+")
    ap.add_argument("--map", required=True)
    ap.add_argument("--label-shift", type=float, default=0.234, help="video time = map label + shift [s]")
    ap.add_argument("--anchor-s", type=float, default=20.0)
    ap.add_argument("--scale", action="store_true", help="also fit scale on the anchor window (off: trust metric)")
    args = ap.parse_args()

    m = json.loads(Path(args.map).expanduser().read_text())
    C = np.array(m["cams"], float)
    gt_t = C[:, 3] / 1000 + args.label_shift
    gt = np.stack([C[:, 0], -C[:, 1], C[:, 2]], 1)  # map OpenCV world (y down)

    for path in args.trajectories:
        d = np.loadtxt(Path(path).expanduser())
        t, P = d[:, 0], d[:, 1:4]
        ok = (gt_t >= t[0]) & (gt_t <= t[-1])
        est = np.stack([np.interp(gt_t[ok], t, P[:, j]) for j in range(3)], 1)
        g, tt = gt[ok], gt_t[ok]
        a = tt <= tt[0] + args.anchor_s
        mx, my = est[a].mean(0), g[a].mean(0)
        U, S, Vt = np.linalg.svd((g[a] - my).T @ (est[a] - mx))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        s = (S * np.diag(D)).sum() / ((est[a] - mx) ** 2).sum() if args.scale else 1.0
        aligned = s * (est - mx) @ R.T + my
        err = np.linalg.norm(aligned - g, axis=1)
        # distance travelled along the (smoothed, 1 s) ground-truth path since the anchor end
        dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1) * (np.diff(tt) < 1.6))])
        d0 = np.interp(tt[0] + args.anchor_s, tt, dist)
        rows = []
        for cp in (37, 80, 110, 141, 170, 192):
            k = int(np.argmin(np.abs(tt - cp)))
            dd = dist[k] - d0
            if dd > 5:
                rows.append(f"{tt[k]:5.0f}s {err[k]:5.2f}m/{dd:4.0f}m={100 * err[k] / dd:4.1f}%")
        after = tt > tt[0] + args.anchor_s
        span = f"{t[0]:.1f}-{t[-1]:.1f}s"
        print(f"{Path(path).name:28s} [{span}] scale x{s:.3f} | median err {np.median(err[after]):.2f} m, "
              f"max {err[after].max():.2f} m | " + " | ".join(rows))


if __name__ == "__main__":
    main()
