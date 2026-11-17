#!/usr/bin/env python3
"""Run a benchmark script on one rented GPU (Vast.ai) in the fleet worker environment, without credentials on the
host: the bundle, the environment and the result upload are pre-signed R2 URLs (logged like the broker's).

  python fleet/remote_bench.py BUNDLE_DIR SCRIPT OUT.json [--args "..."] [--budget 1.5] [--max-bid 0.40]
      [--offer-filter '...'] [--max-minutes 90]

The script runs as `python SCRIPT /tmp/bundle /tmp/out.json ARGS`; /tmp/out.json and the log are uploaded every
60 s and at the end. The instance is destroyed when the result is final, on a timeout, or when the budget is reached.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broker  # noqa: E402
import r2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("script", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--args", default="")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--max-bid", type=float, default=0.40)
    ap.add_argument("--bid-factor", type=float, default=1.5)
    ap.add_argument("--offer-filter", default="gpu_name in [RTX_3090] cpu_cores_effective>=8 cpu_ram>=32 "
                                              "inet_down>=2000 reliability>0.99 geolocation in [US,CA]")
    ap.add_argument("--max-minutes", type=float, default=90)
    a = ap.parse_args()
    broker.PROVIDER = "vast"
    s3, bucket = r2.client()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(a.bundle, arcname="bundle")
        tar.add(a.script, arcname="script.py")
    data = buf.getvalue()
    run = f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
    bundle_key = f"bench/bundle-{hashlib.sha256(data).hexdigest()[:12]}.tar"
    if "Contents" not in s3.list_objects_v2(Bucket=bucket, Prefix=bundle_key):
        s3.upload_fileobj(io.BytesIO(data), bucket, bundle_key)
    broker.log("bench_bundle", run=run, key=bundle_key, bytes=len(data))
    exp = int(6 * 3600)
    url = lambda m, k: broker._presign(s3, bucket, m, k, exp, run)  # noqa: E731, SLF001
    q = {k: shlex.quote(v) for k, v in {
        "env": url("get_object", broker.PAYLOAD_KEY), "bundle": url("get_object", bundle_key),
        "out": url("put_object", f"bench/{run}/out.json"), "log": url("put_object", f"bench/{run}/log.txt")}.items()}
    script = f"""T0=$(date +%s); log(){{ echo "[pf +$(( $(date +%s) - T0 ))s] $*"; }}
exec > /tmp/bench.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq curl zstd ca-certificates >/dev/null && log apt done
push(){{ [ -f /tmp/out.json ] && curl -sS -X PUT --upload-file /tmp/out.json {q['out']}; curl -sS -X PUT --upload-file /tmp/bench.log {q['log']}; }}
( while sleep 60; do push; done ) &
curl -sS --fail {q['env']} | zstd -dq -T0 | tar -x -C / && log environment ready
mkdir -p /tmp/b && curl -sS --fail {q['bundle']} | tar -x -C /tmp/b && log bundle ready
export LD_LIBRARY_PATH=/opt/pf_syslibs TORCH_HOME=/opt/torch_cache PYTHONUNBUFFERED=1
/opt/uv/venv/bin/python /tmp/b/script.py /tmp/b/bundle /tmp/out.json {a.args}; log script exited $?
/opt/uv/venv/bin/python -c "import json,pathlib;p=pathlib.Path('/tmp/out.json');d=json.loads(p.read_text()) if p.exists() else {{}};d['done']=True;p.write_text(json.dumps(d))"
push; log pushed; sleep infinity"""
    inst = None
    t_start, deadline = time.time(), time.time() + a.max_minutes * 60
    result: dict = {}
    try:
        while inst is None and time.time() < deadline:
            inst = broker._launch(a, run, script)  # noqa: SLF001
            if inst is None:
                broker.log("no_offer", dispatch=run)
                time.sleep(60)
        if inst is None:
            raise SystemExit("no offer")
        broker.log("instance_launched", dispatch=run, **inst)
        lost = 0
        while time.time() < deadline:
            time.sleep(30)
            spent = inst["bid"] * (time.time() - inst["launched"]) / 3600 + inst.get("download_cost", 0)
            try:
                result = json.loads(s3.get_object(Bucket=bucket, Key=f"bench/{run}/out.json")["Body"].read())
                a.out.write_text(json.dumps(result | {"instance": inst, "spent_est": round(spent, 3)}, indent=2))
            except Exception:  # noqa: BLE001 - nothing uploaded yet
                pass
            if result.get("done"):
                break
            info = broker._instance(inst["id"])  # noqa: SLF001
            status = (info or {}).get("actual_status")
            lost = lost + 1 if (info is None or status in broker.LOST) and time.time() - inst["launched"] > 180 else 0
            if lost >= 2:
                broker.log("instance_lost", dispatch=run, instance=inst["id"], status=status)
                break
            if spent >= a.budget:
                broker.log("budget_reached", dispatch=run, spent=round(spent, 3))
                break
    finally:
        if inst is not None:
            inst["ended"] = time.time()
            broker._destroy(inst["id"], run, "bench_done")  # noqa: SLF001
            spent = inst["bid"] * (inst["ended"] - inst["launched"]) / 3600 + inst.get("download_cost", 0)
            try:
                log_txt = s3.get_object(Bucket=bucket, Key=f"bench/{run}/log.txt")["Body"].read()
                a.out.with_suffix(".log").write_bytes(log_txt)
            except Exception:  # noqa: BLE001
                pass
            a.out.write_text(json.dumps(result | {"instance": inst, "spent_est": round(spent, 3),
                                                  "wall_min": round((time.time() - t_start) / 60, 1)}, indent=2))
            broker.log("bench_done", dispatch=run, spent=round(spent, 3), done=bool(result.get("done")))


if __name__ == "__main__":
    main()
