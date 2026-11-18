#!/usr/bin/env python3
"""Run the prior-pose mapper offline on a copy of a reconstructor database (inside the reconstructor image).

  docker run --rm --cpus 6 -v <bench dir>:/bench -v <prior_mapper.py>:<pkg>/prior_mapper.py:ro \
      --entrypoint /opt/uv/venv/bin/python <reconstructor image> /bench/bench_prior_mapper.py \
      /bench/database.db /bench/capture.tar /bench/options.json /bench/out
"""
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from placeframe_api_client.models.reconstruction_options import ReconstructionOptions
from reconstructor.metrics_builder import MetricsBuilder
from reconstructor.options_builder import OptionsBuilder
from reconstructor.prior_mapper import prior_pose_mapping
from scipy.spatial.transform import Rotation

db, tar_path, opts_path, out = (Path(a) for a in sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
(out / "images").mkdir(exist_ok=True)
options = OptionsBuilder(ReconstructionOptions(**json.loads(opts_path.read_text())))
priors = {}
with tarfile.open(tar_path) as tf:
    manifest = json.load(tf.extractfile("manifest.json"))
    assert manifest.get("axis_convention", "OPENCV") == "OPENCV"
    for rig in manifest["rigs"]:
        rows = tf.extractfile(f"{rig['id']}/frames.csv").read().decode().splitlines()[1:]
        for r in rows:
            f = r.split(",")
            priors[f"{rig['id']}/{f[0]}"] = (Rotation.from_quat([float(x) for x in f[4:8]]).as_matrix(),
                                             np.array([float(x) for x in f[1:4]]))
print(f"{len(priors)} frame priors", flush=True)
t0 = time.time()
pipeline = options.incremental_pipeline_options()
pipeline.ba_use_gpu = os.environ.get("BENCH_BA_GPU") == "1"
print("ba_use_gpu", pipeline.ba_use_gpu, flush=True)
recs = prior_pose_mapping(db, out / "images", out, priors, pipeline,
                          options.pose_prior_position_sigma_m(), log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
rec = recs[0]
elapsed = time.time() - t0
mb = MetricsBuilder()
mb.build_reconstruction_metrics(rec)
(out / "sfm").mkdir(exist_ok=True)
rec.write_text(str(out / "sfm"))
m = {k: v for k, v in mb.metrics.model_dump().items() if v is not None}
m["mapping_seconds"] = round(elapsed, 1)
(out / "result.json").write_text(json.dumps(m, indent=1))
print(json.dumps(m, indent=1))
