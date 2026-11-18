"""Analyse a replayed tracked-localization session (bench/track_replay.py output) without ground truth.

Reference: the phone's odometry is locally accurate, so within a short window every correct localization implies
the same odometry-from-map alignment. For each frame, the alignment supported by the most frames in a ±W-frame
window (RANSAC over all candidates) is the reference; candidates and the tracker are scored against it.

  python bench/track_analyze.py ~/pf_data/track_replay/<session>
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path.home() / "github/placeframe/docker/localizer"))
from src import tracker as T  # noqa: E402

W = 20  # frames each side (~1 s per frame)
POS_TOL, ANG_TOL = 0.5, np.radians(10)
MIN_SUPPORT = 4


def load(d: Path):
    frames = [json.loads(line) for line in (d / "candidates.jsonl").open()]
    for f in frames:
        f["odom"] = np.array(f["odom"])
        f["time"] = datetime.fromisoformat(f["t"]).timestamp()
        for k in ("plain", "with_prior"):
            for c in f[k]:
                c["m"] = np.array(c["m"])
                c["A"] = f["odom"] @ np.linalg.inv(c["m"])
    return frames


def agree(A, B, odom) -> tuple[float, float]:
    """Distance (m) and angle between where two alignments put the phone at this odometry pose."""
    pa = np.linalg.inv(A) @ odom
    pb = np.linalg.inv(B) @ odom
    return float(np.linalg.norm(pa[:3, 3] - pb[:3, 3])), T._angle(pa, pb)


def segments(frames) -> list[int]:
    """Odometry segment of each frame: a new one starts where the client's odometry jumped (re-anchored itself), since
    the true alignment changes there."""
    seg, out = 0, []
    for i, f in enumerate(frames):
        if i:
            moved = np.linalg.norm(f["odom"][:3, 3] - frames[i - 1]["odom"][:3, 3])
            dt = max(0.05, f["time"] - frames[i - 1]["time"])
            seg += moved > T.JUMP_MIN_M and moved / dt > T.JUMP_SPEED_M_S
        out.append(seg)
    return out


def references(frames):
    refs = []
    seg = segments(frames)
    for i, f in enumerate(frames):
        lo, hi = max(0, i - W), min(len(frames), i + W + 1)
        while seg[lo] != seg[i]:
            lo += 1
        while seg[hi - 1] != seg[i]:
            hi -= 1
        pool = [(j, c) for j in range(lo, hi) for c in frames[j]["plain"] + frames[j]["with_prior"]]
        best, best_support = None, 0
        for _, c in pool:
            support = set()
            for j, d in pool:
                dist, ang = agree(c["A"], d["A"], frames[j]["odom"])
                if dist < POS_TOL and ang < ANG_TOL:
                    support.add(j)
            if len(support) > best_support:
                best, best_support = c["A"], len(support)
        refs.append((best, best_support) if best_support >= MIN_SUPPORT else (None, best_support))
    return refs


def main():
    d = Path(sys.argv[1])
    frames = load(d)
    refs = references(frames)
    have_ref = [r for r, _ in refs if r is not None]
    print(f"{len(frames)} frames; reference alignment for {len(have_ref)} (support ≥ {MIN_SUPPORT} frames in ±{W})")

    # Candidates vs reference
    rows = []
    for f, (ref, _) in zip(frames, refs):
        if ref is None:
            continue
        for kind in ("plain", "with_prior"):
            for rank, c in enumerate(sorted(f[kind], key=lambda c: -c["conf"])):
                dist, ang = agree(c["A"], ref, f["odom"])
                rows.append((kind, rank, c["conf"], c["inl"], dist, ang, dist < POS_TOL and ang < ANG_TOL))
    rows_np = rows
    for kind in ("plain", "with_prior"):
        rk = [r for r in rows_np if r[0] == kind]
        if not rk:
            continue
        good = [r for r in rk if r[6]]
        print(f"\n{kind}: {len(rk)} candidates, {len(good)} correct ({100 * len(good) / len(rk):.0f}%)")
        for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            b = [r for r in rk if lo <= r[2] < hi]
            if b:
                print(f"  conf {lo:.2f}-{hi:.2f}: {len(b):4d}  correct {100 * sum(r[6] for r in b) / len(b):5.1f}%"
                      f"  median inliers {int(np.median([r[3] for r in b]))}")
    frames_with_ref = [(f, r) for f, (r, _) in zip(frames, refs) if r is not None]
    any_correct = sum(any(agree(c["A"], r, f["odom"])[0] < POS_TOL for c in f["plain"] + f["with_prior"])
                      for f, r in frames_with_ref)
    top_correct = sum(bool(f["plain"]) and agree(max(f["plain"], key=lambda c: c["conf"])["A"], r, f["odom"])[0] < POS_TOL
                      for f, r in frames_with_ref)
    print(f"\nframes with a correct candidate: {any_correct}/{len(frames_with_ref)}; "
          f"single-frame top candidate correct: {top_correct}/{len(frames_with_ref)}")

    # Odometry drift: how much the reference alignment changes along the walk (at the phone)
    drift = []
    prev = None
    for f, (r, _) in zip(frames, refs):
        if r is not None and prev is not None:
            drift.append(agree(r, prev[0], f["odom"])[0])
        if r is not None:
            prev = (r, f)
    path = sum(np.linalg.norm(frames[i + 1]["odom"][:3, 3] - frames[i]["odom"][:3, 3]) for i in range(len(frames) - 1))
    print(f"path {path:.0f} m; reference jumps frame to frame: median {np.median(drift):.2f} m, "
          f"p90 {np.percentile(drift, 90):.2f} m")
    first = next(r for r, _ in refs if r is not None)
    last = next(r for r, _ in reversed(refs) if r is not None)
    print(f"alignment change first→last (at the last frame): {agree(first, last, frames[-1]['odom'])[0]:.2f} m")

    # Tracker as replayed (live algorithm at replay time), then the current tracker.py simulated on the candidates
    evaluate("replayed tracker", [(f, np.array(f["best_A"]) if f["best_A"] else None, f["best_p"]) for f in frames], refs)
    evaluate("current tracker.py (simulated)", simulate(frames), refs)
    return frames, refs


def simulate(frames):
    """Run tracker.py over the saved candidates; candidates found near the predicted pose are used when the
    simulated tracker has a prediction (an approximation: they were found around the replayed tracker's)."""
    tracker = T.Tracker()
    served = []
    for f in frames:
        use_prior = tracker.predicted_map_from_camera(f["odom"]) is not None
        cands = [T.Candidate(c["m"], c["conf"], c["inl"], c["src"]) for c in f["plain"] + (f["with_prior"] if use_prior else [])]
        tracker.update(f["odom"], cands, now=f["time"])
        b = tracker.best
        served.append((f, b.odom_from_map if b else None, b.probability if b else 0))
    return served


def evaluate(name, served, refs, threshold=0.8):
    shown = wrong = correct = 0
    errs = []
    for (f, A, p), (ref, _) in zip(served, refs):
        if A is None or ref is None or p < threshold:
            continue
        shown += 1
        dist, ang = agree(A, ref, f["odom"])
        errs.append(dist)
        if dist > 1.0 or ang > np.radians(20):
            wrong += 1
        elif dist < POS_TOL:
            correct += 1
    n = sum(r is not None for r, _ in refs)
    print(f"\n{name}: shown (p≥{threshold}) on {shown}/{n} referenced frames; within {POS_TOL} m: {correct}; "
          f"wrong (>1 m): {wrong}; median error {np.median(errs) if errs else float('nan'):.2f} m")
    return shown, correct, wrong


if __name__ == "__main__":
    main()
