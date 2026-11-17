#!/usr/bin/env python3
"""Run one capture on the fleet without the Placeframe API: stands in for lease_agent.py for measurements whose
capture is too big for the API's 1 GiB upload limit (or when the API should not get the map).

Stages the capture to R2, announces it to the broker (start fleet/broker.py separately), records every phase
change of the remote worker with its time, and writes the final status (metrics, memory peaks) as JSON. The map
itself is not imported into Placeframe; the results tar stays in R2 under jobs/<id>/.

  python fleet/direct_job.py capture.tar out.json [--conversion conversion.json] [--name j8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import r2  # noqa: E402
from upload_to_placeframe import choose_reconstruction_options  # noqa: E402

BROKER = Path.home() / "pf_fleet/broker"
MAPPER = Path.home() / "github/placeframe/local/reconstructor-mapper.json"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--conversion", type=Path)
    ap.add_argument("--name", default="")
    ap.add_argument("--staged", help="reconstruction id whose capture is already in R2 (jobs/<id>/capture.tar)")
    a = ap.parse_args()
    conv_path = a.conversion or a.capture.with_name("conversion.json")
    options, _ = choose_reconstruction_options(a.capture, json.loads(conv_path.read_text()))
    rid, cid = a.staged or str(uuid.uuid4()), str(uuid.uuid4())
    prefix = f"jobs/{rid}"
    s3, bucket = r2.client()
    t_up = time.time()
    if a.staged:
        size = s3.head_object(Bucket=bucket, Key=f"{prefix}/capture.tar")["ContentLength"]
        if size != a.capture.stat().st_size:
            raise SystemExit(f"staged capture is {size} bytes, {a.capture} is {a.capture.stat().st_size}")
        for old in s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/").get("Contents", []):
            if old["Key"] != f"{prefix}/capture.tar":  # a fresh run: no old status or checkpoints
                s3.delete_object(Bucket=bucket, Key=old["Key"])
    else:
        s3.upload_file(str(a.capture), bucket, f"{prefix}/capture.tar")
    upload_s = time.time() - t_up
    log(f"staged {a.capture} ({a.capture.stat().st_size / 1e9:.2f} GB) to R2 in {upload_s:.0f}s as {prefix}")
    job = {"reconstruction_id": rid, "capture_id": cid, "options": options,
           "mapper_config": json.loads(MAPPER.read_text()) if MAPPER.exists() else None,
           "r2_prefix": prefix, "leased": time.time(), "direct": a.name}
    (BROKER / "jobs").mkdir(parents=True, exist_ok=True)
    job_file = BROKER / "jobs" / f"{rid}.json"
    job_file.write_text(json.dumps(job, indent=1))
    t0, stages, last, st = time.time(), [], None, {}
    try:
        while True:
            time.sleep(15)
            try:
                st = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/status.json")["Body"].read())
            except Exception:  # noqa: BLE001 - no status yet (instance still starting)
                st = st or {}
            phase = (st.get("state"), st.get("phase"), st.get("dispatch"))
            if st and phase != last:
                stages.append({"state": st.get("state"), "phase": st.get("phase"), "dispatch": st.get("dispatch"),
                               "t": round(time.time() - t0, 1)})
                log(f"{phase} at {time.time() - t0:.0f}s memory={st.get('memory')}")
                last = phase
            if st.get("state") in ("succeeded", "failed"):
                break
            if (BROKER / "outcome" / rid).exists():
                log("broker gave up on the job")
                break
    finally:
        job_file.unlink(missing_ok=True)
        (BROKER / "done").mkdir(exist_ok=True)
        (BROKER / "done" / rid).write_text(str(time.time()))
        out = {"name": a.name, "reconstruction_id": rid, "r2_prefix": prefix, "options": options,
               "upload_s": round(upload_s, 1), "total_s": round(time.time() - t0, 1), "stages": stages,
               "final_status": st}
        a.out.write_text(json.dumps(out, indent=2))
        log(f"wrote {a.out}: {st.get('state')}")


if __name__ == "__main__":
    main()
