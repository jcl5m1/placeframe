#!/bin/bash
# Right-sized rented-GPU build for the cost study: the j4 joined map (9,072 images) on one RTX 3090-class Vast
# instance like the home machine (>=12 cores, >=32 GB RAM). The home worker is paused meanwhile and restored after.
set -u
FLEET=~/pf_fleet; REPO=~/github/placeframe-insta360; PF=~/github/placeframe
HOST=${PF_PUBLIC_HOST:?set PF_PUBLIC_HOST to the server's public host name}
# The home worker keeps running (it is busy with another build); submit after it has leased that one.
sg docker -c "docker rm -f pf-lease-agent" >/dev/null 2>&1
sg docker -c "docker run -d --name pf-lease-agent --network placeframe_default --add-host $HOST:172.17.0.1 \
  -v placeframe_auth_data:/run/secrets:ro -v $FLEET:/root/pf_fleet -v $REPO/fleet:/fleet:ro \
  -v $PF/local/reconstructor-mapper.json:/etc/placeframe/mapper.json:ro \
  --env-file $FLEET/local-worker.env --entrypoint /opt/uv/venv/bin/python placeframe-worker:local -u /fleet/lease_agent.py" >/dev/null
cd $REPO && .venv/bin/python - <<'PY'
import json, sys, time
sys.path.insert(0, '.')
from pathlib import Path
from add_recording_to_map import PlaceframeApi
from upload_to_placeframe import choose_reconstruction_options
api = PlaceframeApi()
tar = Path.home() / "pf_cost/variants/j4/capture.tar"
opts, _ = choose_reconstruction_options(tar, {"pose_source": "insta360 VIO", "metric": True})
t0 = time.time()
cs = api.upload_capture(tar, "cost-study-v2-j4-vast")
rid = api.post("/reconstructions", {"create": {"capture_session_id": cs}, "options": opts})["id"]
json.dump({"capture_session_id": cs, "reconstruction_id": rid, "submitted": time.time(), "upload_s": time.time() - t0},
          open(Path.home() / "pf_cost/vast_job.json", "w"))
print("submitted", rid)
PY
cd $REPO && timeout 3h .venv/bin/python fleet/broker.py --provider vast --max-instances 1 --budget 2 --max-bid 0.30 \
  --offer-filter 'gpu_name in [RTX_3090,RTX_4090] cpu_cores_effective>=12 cpu_ram>=32' --until-idle
sg docker -c "docker rm -f pf-lease-agent" >/dev/null 2>&1
echo VAST_TEST_DONE
