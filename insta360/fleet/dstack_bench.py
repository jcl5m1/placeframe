#!/usr/bin/env python3
"""Run the same benchmark as fleet/remote_bench.py, but let dstack find, start and stop the Vast instance.

Compares dstack with our own launcher: time to a running task, what offers it can filter, how it bids, and
cleanup. The task gets pre-signed R2 URLs only (no credentials), like remote_bench.py.

  python fleet/dstack_bench.py BUNDLE_DIR SCRIPT OUT.json [--args "..."] [--gpu RTX3090:1] [--max-price 0.45]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broker  # noqa: E402
import r2  # noqa: E402


def sh(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return p.stdout + p.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("script", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--args", default="")
    ap.add_argument("--gpu", default="RTX3090:1")
    ap.add_argument("--cpu", default="8..")
    ap.add_argument("--memory", default="32GB..")
    ap.add_argument("--max-price", type=float, default=0.45)
    ap.add_argument("--max-minutes", type=float, default=75)
    a = ap.parse_args()
    s3, bucket = r2.client()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(a.bundle, arcname="bundle")
        tar.add(a.script, arcname="script.py")
    data = buf.getvalue()
    run = f"dstack-bench-{time.strftime('%Y%m%d-%H%M%S')}"
    bundle_key = f"bench/bundle-{hashlib.sha256(data).hexdigest()[:12]}.tar"
    if "Contents" not in s3.list_objects_v2(Bucket=bucket, Prefix=bundle_key):
        s3.upload_fileobj(io.BytesIO(data), bucket, bundle_key)
    url = lambda m, k: broker._presign(s3, bucket, m, k, 6 * 3600, run)  # noqa: E731, SLF001
    env = {"PF_ENV_URL": url("get_object", broker.PAYLOAD_KEY), "PF_BUNDLE_URL": url("get_object", bundle_key),
           "PF_OUT_URL": url("put_object", f"bench/{run}/out.json")}
    commands = [
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update -qq >/dev/null && apt-get install -y -qq curl zstd ca-certificates >/dev/null",
        'curl -sS --fail "$PF_ENV_URL" | zstd -dq -T0 | tar -x -C /',
        'mkdir -p /tmp/b && curl -sS --fail "$PF_BUNDLE_URL" | tar -x -C /tmp/b',
        "export LD_LIBRARY_PATH=/opt/pf_syslibs TORCH_HOME=/opt/torch_cache PYTHONUNBUFFERED=1",
        f"/opt/uv/venv/bin/python /tmp/b/script.py /tmp/b/bundle /tmp/out.json {a.args}",
        'curl -sS -X PUT --upload-file /tmp/out.json "$PF_OUT_URL"',
    ]
    task = {"type": "task", "name": run.replace("_", "-")[:40], "image": "ubuntu:24.04", "commands": commands,
            "env": [f"{k}={v}" for k, v in env.items()], "backends": ["vastai"], "spot_policy": "spot",
            "max_price": a.max_price, "max_duration": f"{int(a.max_minutes)}m",
            "retry": {"on_events": ["no-capacity", "interruption", "error"], "duration": "30m"},
            "resources": {"gpu": a.gpu, "cpu": a.cpu, "memory": a.memory, "disk": "60GB.."}}
    cfg = Path(f"/tmp/{run}.dstack.yml")
    cfg.write_text(json.dumps(task))  # JSON is valid YAML
    cfg.chmod(0o600)  # holds pre-signed URLs
    t0 = time.time()
    broker.log("dstack_apply", dispatch=run, gpu=a.gpu, max_price=a.max_price)
    print(sh(["dstack", "apply", "-f", str(cfg), "-y", "-d"])[-1500:], flush=True)
    events, last = [], None
    result: dict = {}
    try:
        while time.time() - t0 < a.max_minutes * 60:
            time.sleep(20)
            ps = sh(["dstack", "ps", "-a", "--json"])
            try:
                runs = json.loads(ps)
                runs = runs.get("runs", runs) if isinstance(runs, dict) else runs
                me = next(r for r in runs if (r.get("run_spec", {}).get("run_name") or r.get("run_name")) == task["name"])
            except (ValueError, StopIteration):
                continue
            status = me.get("status")
            if status != last:
                events.append({"status": status, "t": round(time.time() - t0, 1)})
                print(time.strftime("%H:%M:%S"), status, flush=True)
                last = status
            if status in ("done", "failed", "terminated", "aborted"):
                result["run"] = {k: v for k, v in me.items() if k != "run_spec"}  # run_spec holds the URLs
                break
        try:
            result |= json.loads(s3.get_object(Bucket=bucket, Key=f"bench/{run}/out.json")["Body"].read())
        except Exception:  # noqa: BLE001 - no result uploaded
            pass
    finally:
        print(sh(["dstack", "stop", task["name"], "-y"])[-300:], flush=True)
        cfg.unlink(missing_ok=True)
        result["dstack_events"] = events
        result["wall_min"] = round((time.time() - t0) / 60, 1)
        a.out.write_text(json.dumps(result, indent=2, default=str))
        broker.log("dstack_done", dispatch=run, wall_min=result["wall_min"], last_status=last)


if __name__ == "__main__":
    main()
