"""Score candidate-generation variants of one replayed session against the same reference (built once from the
baseline run): frames with a correct candidate, single-frame top candidate correct, simulated tracker, latency.

  python bench/track_compare.py REPLAY_DIR baseline.jsonl variant1.jsonl ...
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import track_analyze as A  # noqa: E402

d = Path(sys.argv[1])
cache = d / "reference.pkl"


def load(name):
    frames = [json.loads(line) for line in (d / name).open()]
    for f in frames:
        f["odom"] = np.array(f["odom"])
        f["time"] = A.datetime.fromisoformat(f["t"]).timestamp()
        for k in ("plain", "with_prior"):
            for c in f[k]:
                c["m"] = np.array(c["m"])
                c["A"] = f["odom"] @ np.linalg.inv(c["m"])
    return frames


base = load(sys.argv[2])
if cache.exists():
    refs = pickle.loads(cache.read_bytes())
else:
    refs = A.references(base)
    cache.write_bytes(pickle.dumps(refs))
print(f"{'variant':34s} {'any correct':>11s} {'top correct':>11s} {'tracker ok':>10s} {'wrong>1m':>8s} {'ms p50':>7s}")
for name in sys.argv[2:]:
    frames = load(name)
    anyc = topc = 0
    for f, (r, _) in zip(frames, refs):
        if r is None:
            continue
        ok = [A.agree(c["A"], r, f["odom"])[0] < A.POS_TOL for c in f["plain"] + f["with_prior"]]
        anyc += any(ok)
        if f["plain"]:
            top = max(f["plain"], key=lambda c: c["conf"])
            topc += A.agree(top["A"], r, f["odom"])[0] < A.POS_TOL
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        shown, good, wrong = A.evaluate(name, A.simulate(frames), refs)
    ms = np.median([f.get("plain_ms", np.nan) for f in frames])
    print(f"{name:34s} {anyc:11d} {topc:11d} {good:10d} {wrong:8d} {ms:7.0f}")
