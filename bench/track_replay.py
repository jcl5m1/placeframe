"""Replay a recorded tracked-localization session inside the localizer image: for every frame, the candidate poses
with and without the tracker's prior, and the tracker's output, saved for offline analysis (bench/track_analyze.py).

  docker cp bench/track_replay.py placeframe-localizer-cuda-1:/tmp/
  docker exec -e REPLAY_DIR=/tmp/replay placeframe-localizer-cuda-1 /opt/uv/venv/bin/python /tmp/track_replay.py RID
REPLAY_DIR holds session.json (the recorded requests), img/ and metrics.json; writes candidates.jsonl.
"""
import json
import os
import sys
import time
from pathlib import Path
from uuid import UUID

import numpy as np

sys.path.insert(0, "/app/docker/localizer")
import src.localize as L  # noqa: E402
from common.boto_clients import create_s3_client  # noqa: E402
from core.calibration import load_global_calibration  # noqa: E402
from core.camera_config import PinholeCameraConfig  # noqa: E402
from src.map import load_map  # noqa: E402
from src.tracked import localize_candidates  # noqa: E402
from src.tracker import Tracker, pose_from_quaternion  # noqa: E402

# Settings under test, from the environment: TOPK, PLACE_RADIUS, PRIOR_TOPK, PRIOR_RADIUS, MIN_CORR, RESIZE (shorter
# side the query is resized to; 0 = keep the camera's size), OUT (output file name).
import core.image_preprocess as IP  # noqa: E402
import src.tracked as TR  # noqa: E402

for env, mod, name in (("TOPK", TR, "RETRIEVAL_TOP_K"), ("PLACE_RADIUS", TR, "PLACE_RADIUS_M"),
                       ("PRIOR_TOPK", TR, "PRIOR_TOP_K"), ("PRIOR_RADIUS", TR, "PRIOR_RADIUS_M"),
                       ("MIN_CORR", TR, "MIN_CORRESPONDENCES"), ("RESIZE", IP, "LOCAL_FEATURE_RESIZE_SHORTER_SIDE")):
    if os.environ.get(env) and os.environ[env] != "0":
        v = float(os.environ[env])
        setattr(mod, name, int(v) if v.is_integer() else v)
        print(f"{name} = {getattr(mod, name)}", flush=True)
if os.environ.get("RESIZE") == "0":
    IP._resized_dimensions = lambda w, h: (w, h)
RID = sys.argv[1]
D = Path(os.environ.get("REPLAY_DIR", "/tmp/replay"))
s3 = create_s3_client(minio_endpoint_url=os.environ["MINIO_ENDPOINT_URL"], minio_access_key=os.environ["MINIO_ACCESS_KEY"],
                      minio_secret_key=os.environ["MINIO_SECRET_KEY"])
L.load_models()
cal = load_global_calibration(Path("/etc/placeframe/calibration/global.json"), os.environ["LOCALIZER_SHA"])


class _Metrics:
    def model_dump(self):
        return json.loads((D / "metrics.json").read_text())


m = load_map(UUID(RID), s3, os.environ.get("RECONSTRUCTIONS_BUCKET", "dev-reconstructions"), Path("/tmp/reconstructions"), _Metrics())
print("map:", len(m.ordered_image_ids), "images", flush=True)
session = json.loads((D / "session.json").read_text())
tracker = Tracker()
out = (D / os.environ.get("OUT", "candidates.jsonl")).open("w")


def dump(cs):
    return [{"m": c.map_from_camera.tolist(), "conf": c.confidence, "inl": c.num_inliers, "src": c.source} for c in cs]


t0 = time.time()
for i, r in enumerate(session):
    q = r["request"]
    o = q["odometry_from_camera"]
    odom = pose_from_quaternion([o["translation"][k] for k in "xyz"], [o["rotation"][k] for k in "xyzw"])
    cam = PinholeCameraConfig.model_validate(q["camera_config"])
    img = (D / "img" / r["image"]).read_bytes()
    t1 = time.time()
    plain = localize_candidates(m, cam, img, None, False, os.environ["LOCALIZER_SHA"], cal)
    plain_ms = (time.time() - t1) * 1000
    prior = tracker.predicted_map_from_camera(odom)
    confident = tracker.confident and os.environ.get("CONFIDENT_MODE", "1") == "1"
    t2 = time.time()
    cands = localize_candidates(m, cam, img, prior, confident, os.environ["LOCALIZER_SHA"], cal) if prior is not None else plain
    live_ms = (time.time() - t2) * 1000 if prior is not None else plain_ms
    res = tracker.update(odom, cands)
    b = tracker.best
    out.write(json.dumps({"i": i, "t": r["received_at"], "odom": odom.tolist(), "plain": dump(plain),
                          "with_prior": dump([c for c in cands if c.source == "prior"]),
                          "prior": prior.tolist() if prior is not None else None,
                          "best_p": b.probability if b else 0, "best_A": b.odom_from_map.tolist() if b else None,
                          "agreed": res.agreed, "plain_ms": plain_ms, "live_ms": live_ms,
                          "confident": confident}) + "\n")
    out.flush()
    if i % 25 == 0:
        print(i, f"{time.time() - t0:.0f}s", len(plain), f"p={b.probability if b else 0:.2f}", flush=True)
print("done", time.time() - t0)
