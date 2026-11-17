#!/usr/bin/env python3
"""Watch a rented run whose controller (direct_job.py or remote_bench.py) stopped, and finish it the same way:
record its status, save the result, destroy the instance when the result is final.

  python fleet/reattach.py job   RECONSTRUCTION_ID INSTANCE_ID OUT.json
  python fleet/reattach.py bench RUN_NAME          INSTANCE_ID OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import broker  # noqa: E402
import r2  # noqa: E402


def main() -> None:
    kind, name, iid, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    broker.PROVIDER = "vast"
    s3, bucket = r2.client()
    key = f"jobs/{name}/status.json" if kind == "job" else f"bench/{name}/out.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    stages, last, st, lost = prev.get("stages", []), None, {}, 0
    t0 = time.time()
    try:
        while True:
            time.sleep(30)
            try:
                st = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            except Exception:  # noqa: BLE001
                pass
            if kind == "job":
                phase = (st.get("state"), st.get("phase"), st.get("dispatch"))
                if st and phase != last:
                    stages.append({"state": st.get("state"), "phase": st.get("phase"), "wall": round(time.time(), 1),
                                   "memory": st.get("memory")})
                    print(time.strftime("%H:%M:%S"), phase, st.get("memory"), flush=True)
                    last = phase
                done = st.get("state") in ("succeeded", "failed")
            else:
                done = bool(st.get("done"))
            out.write_text(json.dumps(prev | {"reattached": True, "stages": stages, "final_status": st}
                                      if kind == "job" else st, indent=2))
            if done:
                break
            info = broker._instance(iid)  # noqa: SLF001
            status = (info or {}).get("actual_status")
            lost = lost + 1 if info is None or status in broker.LOST else 0
            if lost >= 2:
                print(time.strftime("%H:%M:%S"), "instance lost:", status, flush=True)
                break
    finally:
        broker._destroy(iid, name, "reattach_done")  # noqa: SLF001
        if kind == "job":
            (broker.BROKER / "jobs" / f"{name}.json").unlink(missing_ok=True)
            (broker.BROKER / "done").mkdir(exist_ok=True)
            (broker.BROKER / "done" / name).write_text(str(time.time()))
        print(time.strftime("%H:%M:%S"), f"finished after {(time.time() - t0) / 60:.0f} min; instance destroyed",
              flush=True)


if __name__ == "__main__":
    main()
