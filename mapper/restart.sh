#!/bin/sh
# Safely restart the map-updater service: stop the running merge job (it resumes after its last finished stage),
# stop the service, mark running jobs queued (atomically), start the service again, verify it answers.
set -u
# service first (so its worker can't mark the interrupted job failed), then the job's process
for pid in $(ps -eo pid,args | awk '/uvicorn mapper.server:app/ && !/awk/ {print $1}'); do kill "$pid"; done
sleep 2
# Stop each job with all its children: jobs run in their own process group; earlier jobs (started before that) and
# their converters are also caught by path. Job containers (OpenVIO/OpenVINS) are labelled pf-mapper-job.
for pid in $(ps -eo pid,args | awk '/python -u .*add_recording_to_map.py/ && !/awk/ {print $1}'); do
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' '); [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null; kill "$pid" 2>/dev/null
done
for pid in $(ps -eo pid,args | awk '/pf_mapper\/(jobs|uploads)\// && /(placeframe-insta360\/.*\.py|ffmpeg)/ && !/awk/ {print $1}'); do kill "$pid" 2>/dev/null; done
sg docker -c "docker ps -q --filter label=pf-mapper-job=1" | xargs -r -I{} sg docker -c "docker kill {}" >/dev/null 2>&1
sleep 2
python3 - <<'PY'
import json
from pathlib import Path
for p in (Path.home() / "pf_mapper/jobs").glob("*/job.json"):
    try:
        j = json.loads(p.read_text())
    except ValueError:
        continue
    if j.get("state") == "running":
        j["state"] = "queued"
        tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(j, indent=2)); tmp.replace(p)
PY
setsid nohup "$(dirname "$0")/run.sh" >> "$HOME/pf_mapper/service.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do sleep 2; curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8150/ | grep -q 200 && { echo "service up"; exit 0; }; done
echo "service did not come up — see ~/pf_mapper/service.log"; exit 1
