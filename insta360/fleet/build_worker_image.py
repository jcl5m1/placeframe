#!/usr/bin/env python3
"""Build the fleet worker image: stage the local reconstructor patches from the placeframe repo, then
docker build gpu/Dockerfile.worker. Needs gpu/dist/pycolmap-*.whl (gpu/Dockerfile.pycolmap-cuda).

  python fleet/build_worker_image.py --tag placeframe-worker:local
"""
import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path.home() / "github/placeframe/docker/reconstructor/src/reconstructor"
PATCHES = ["colmap.py", "main.py", "pairs.py", "prior_mapper.py", "run_reconstruction.py", "checkpoints.py",
           "progress_publisher.py"]
CORE_SRC = Path.home() / "github/placeframe/packages/python/core/src/core"
CORE_PATCHES = ["opq.py", "lightglue.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="placeframe-worker:local")
a = ap.parse_args()
if not list((ROOT / "gpu/dist").glob("pycolmap-*.whl")):
    raise SystemExit("gpu/dist has no pycolmap wheel: build gpu/Dockerfile.pycolmap-cuda first")
stage = ROOT / "placeframe-patches"
shutil.rmtree(stage, ignore_errors=True)
stage.mkdir()
for f in PATCHES:
    shutil.copy2(SRC / f, stage / f)
core_stage = ROOT / "placeframe-patches-core"
shutil.rmtree(core_stage, ignore_errors=True)
core_stage.mkdir()
for f in CORE_PATCHES:
    shutil.copy2(CORE_SRC / f, core_stage / f)
subprocess.run(["sg", "docker", "-c", f"docker build -f gpu/Dockerfile.worker -t {a.tag} ."], cwd=ROOT, check=True)
print("built", a.tag)
