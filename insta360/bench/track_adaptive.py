"""Simulate adaptive query rates on a replayed tracked session: frames are sent only as often as the policy allows
(the rest are skipped: the tracker only carries the pose by odometry), and the served pose is scored on every frame.

  python bench/track_adaptive.py REPLAY_DIR CANDIDATES.jsonl
"""
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import track_analyze as A  # noqa: E402

T = A.T


def run(frames, policy, react_to_jumps=True):
    tracker = T.Tracker()
    served, sent, last_sent, last_agreed = [], 0, -1e9, -1e9
    prev, jump_pending, last_ok = None, False, False
    for f in frames:
        best = tracker.best
        p = best.probability if best else 0.0
        interval = policy(p, f["time"] - last_agreed) if last_ok else 0.0
        # The client sees its own odometry jump (re-anchoring): query at once, the alignment just changed.
        jumped = prev is not None and np.linalg.norm(f["odom"][:3, 3] - prev["odom"][:3, 3]) > T.JUMP_MIN_M and \
            np.linalg.norm(f["odom"][:3, 3] - prev["odom"][:3, 3]) / max(0.05, f["time"] - prev["time"]) > T.JUMP_SPEED_M_S
        prev = f
        if jumped and react_to_jumps:
            jump_pending = True
            last_ok = False
        if f["time"] - last_sent >= interval - 1e-6 or jump_pending:
            jump_pending = False
            use_prior = tracker.predicted_map_from_camera(f["odom"]) is not None
            cands = [T.Candidate(c["m"], c["conf"], c["inl"], c["src"])
                     for c in f["plain"] + (f["with_prior"] if use_prior else [])]
            res = tracker.update(f["odom"], cands, now=f["time"])
            sent += 1
            last_sent = f["time"]
            last_ok = res.agreed  # slow down only while queries keep agreeing
            if res.agreed:
                last_agreed = f["time"]
        b = tracker.best
        served.append((f, b.odom_from_map if b else None, b.probability if b else 0))
    return served, sent


def main():
    d = Path(sys.argv[1])
    frames = [json.loads(line) for line in (d / sys.argv[2]).open()]
    for f in frames:
        f["odom"] = np.array(f["odom"])
        f["time"] = A.datetime.fromisoformat(f["t"]).timestamp()
        for k in ("plain", "with_prior"):
            for c in f[k]:
                c["m"] = np.array(c["m"])
    refs = pickle.loads((d / "reference.pkl").read_bytes())
    duration = frames[-1]["time"] - frames[0]["time"]
    policies = {
        "every frame (recorded ~1 Hz)": lambda p, since: 0,
        "confident: every 3 s": lambda p, since: 3.0 if p >= 0.95 and since < 10 else 0,
        "confident: every 5 s": lambda p, since: 5.0 if p >= 0.95 and since < 15 else 0,
        "confident: every 10 s": lambda p, since: 10.0 if p >= 0.95 and since < 30 else 0,
    }
    print(f"{'policy':32s} {'queries':>7s} {'per min':>7s} {'shown':>6s} {'<0.5 m':>6s} {'>1 m':>5s} {'median m':>8s}")
    for (name, pol), react in [(x, r) for x in policies.items() for r in (False, True)]:
        if not react and name.startswith("every frame"):
            continue
        name = name + (" + jump" if react else "")
        served, sent = run(frames, pol, react)
        with contextlib.redirect_stdout(io.StringIO()):
            shown, good, wrong = A.evaluate(name, served, refs)
        errs = [A.agree(Am, r, f["odom"])[0] for (f, Am, p), (r, _) in zip(served, refs) if Am is not None and r is not None and p >= 0.8]
        print(f"{name:32s} {sent:7d} {60 * sent / duration:7.1f} {shown:6d} {good:6d} {wrong:5d} {np.median(errs):8.2f}")


if __name__ == "__main__":
    main()
