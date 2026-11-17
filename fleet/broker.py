#!/usr/bin/env python3
"""Fleet broker: runs leased Placeframe jobs on rented interruptible GPUs (Vast.ai) without giving them credentials.

Works with fleet/lease_agent.py (which holds the Placeframe credentials and stages data). For every job the agent
announces (~/pf_fleet/broker/jobs/<reconstruction id>.json) the broker:
  * writes a ticket of pre-signed R2 URLs scoped to that job (capture and checkpoints to read; status, checkpoints
    and results to write), each expiring after --url-hours, and logs every URL it issues;
  * rents an instance (walking down the offer list: offers go stale fast) whose start script fetches the worker
    environment and fleet/remote_worker.py by pre-signed URL and runs the ticket;
  * watches it: an instance that disappears (outbid, preempted, host failure) is replaced, and the new ticket includes
    the checkpoints the old one saved, up to --attempts times; then the job is abandoned (the agent fails it);
  * destroys the instance when the agent reports the job done, and stops renting at the --budget cap.
Every access grant and instance event is appended to ~/pf_fleet/access_log.jsonl.

  python fleet/broker.py --max-instances 2 --budget 3 [--chaos-kill-after 300] [--until-idle]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2  # noqa: E402

FLEET = Path.home() / "pf_fleet"
BROKER = FLEET / "broker"
LOG = FLEET / "access_log.jsonl"
STATE = BROKER / "dispatches.json"
PAYLOAD_KEY = "payload/worker_env-7c55.tar.zst"
OFFER_QUERY = "reliability>0.98 num_gpus=1 gpu_ram>=20 inet_down>=1000 disk_space>=40 cuda_vers>=12.8 rentable=true"
LOST = {"exited", "offline", "stopped", "unknown", "dead"}
PROVIDER = "vast"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-instances", type=int, default=1)
    ap.add_argument("--budget", type=float, default=2.0, help="stop renting once estimated spend reaches this ($)")
    ap.add_argument("--max-bid", type=float, default=0.30, help="highest $/h bid for one GPU")
    ap.add_argument("--offer-filter", default="",
                    help="extra Vast offer conditions to rent what the job needs, not the cheapest big GPU, e.g. "
                         "'gpu_name in [RTX_3090,RTX_4090] cpu_cores_effective>=12 cpu_ram>=32'")
    ap.add_argument("--attempts", type=int, default=5, help="instances lost (not counting chaos kills) before giving up")
    ap.add_argument("--bid-factor", type=float, default=1.5,
                    help="bid = minimum bid x this: low bids get outbid (preempted) often")
    ap.add_argument("--url-hours", type=float, default=6)
    ap.add_argument("--chaos-kill-after", type=float, default=0, help="destroy each job's first instance after N s")
    ap.add_argument("--chaos-after-checkpoint", action="store_true",
                    help="destroy each job's first instance once it has saved its features checkpoint (tests resume)")
    ap.add_argument("--until-idle", action="store_true", help="exit when no job is active for 10 minutes")
    ap.add_argument("--interval", type=float, default=20)
    ap.add_argument("--provider", choices=["vast", "local"], default="vast",
                    help="local: run each 'instance' as a GPU container on this machine (free end-to-end test)")
    a = ap.parse_args()
    global PROVIDER
    PROVIDER = a.provider

    BROKER.mkdir(parents=True, exist_ok=True)
    (BROKER / "outcome").mkdir(exist_ok=True)
    s3, bucket = r2.client()
    code_key = _publish_code(s3, bucket)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    (BROKER / "capacity").write_text(str(a.max_instances))
    log("broker_start", max_instances=a.max_instances, budget=a.budget, max_bid=a.max_bid, offer_filter=a.offer_filter)
    idle_since = time.time()
    try:
        while True:
            spent = sum(_cost(d) for d in state.values())
            jobs = {p.stem: json.loads(p.read_text()) for p in (BROKER / "jobs").glob("*.json")}
            for rid, d in state.items():
                if d["status"] in ("running", "launching"):
                    _watch(a, s3, bucket, code_key, state, rid, d, jobs.get(rid), spent)
            for rid, job in jobs.items():
                live = sum(1 for d in state.values() if d["status"] in ("running", "launching"))
                waiting = rid in state and state[rid]["status"] == "waiting_for_offer"
                if (rid in state and not waiting) or live >= a.max_instances:
                    continue
                if spent >= a.budget:
                    log("budget_reached", spent=round(spent, 3), budget=a.budget)
                    (BROKER / "capacity").write_text("0")
                    break
                state.setdefault(rid, {"attempt": 0, "status": "new", "instances": []})
                _dispatch(a, s3, bucket, code_key, state, rid, job)
            STATE.write_text(json.dumps(state, indent=1))
            if any(d["status"] in ("running", "launching") for d in state.values()) or jobs:
                idle_since = time.time()
            elif a.until_idle and time.time() - idle_since > 600:
                break
            time.sleep(a.interval)
    finally:
        (BROKER / "capacity").write_text("0")
        for rid, d in state.items():
            if d["status"] in ("running", "launching"):
                _destroy(d["instances"][-1]["id"], rid, "broker_exit")
                d["instances"][-1]["ended"] = time.time()
                d["status"] = "stopped"
        STATE.write_text(json.dumps(state, indent=1))
        log("broker_stop", spent=round(sum(_cost(d) for d in state.values()), 3))


def _watch(a, s3, bucket, code_key, state, rid, d, job, spent) -> None:
    inst = d["instances"][-1]
    if (BROKER / "done" / rid).exists():
        _destroy(inst["id"], rid, "job_done")
        inst["ended"] = time.time()
        d["status"] = "done"
        return
    info = _instance(inst["id"])
    if info is None and time.time() - inst["launched"] < 180:
        return  # a new instance can take a moment to be listed
    status = (info or {}).get("actual_status") or "unknown"
    if status == "running" and not inst.get("running_since"):
        inst["running_since"] = time.time()
        log("instance_running", reconstruction=rid, instance=inst["id"], dispatch=inst["dispatch"])
    checkpointed = a.chaos_after_checkpoint and job and "Contents" in s3.list_objects_v2(
        Bucket=bucket, Prefix=f"{job['r2_prefix']}/checkpoints/features.npz")
    timed_out = a.chaos_kill_after and inst.get("running_since") and \
        time.time() - inst["running_since"] > a.chaos_kill_after
    if d["attempt"] == 1 and not inst.get("chaos") and (checkpointed or timed_out):
        inst["chaos"] = True
        log("chaos_kill", reconstruction=rid, instance=inst["id"])
        _destroy(inst["id"], rid, "chaos")
        status = "unknown"
    gone = info is None or status in LOST or info.get("intended_status") == "stopped"
    # Vast can report a starting container as exited for a moment: only a state seen on two checks counts.
    inst["lost_checks"] = inst.get("lost_checks", 0) + 1 if gone and not inst.get("chaos") else (99 if gone else 0)
    if inst["lost_checks"] >= 2:
        inst["ended"] = time.time()
        log("instance_lost", reconstruction=rid, instance=inst["id"], status=status,
            intended=(info or {}).get("intended_status"), message=((info or {}).get("status_msg") or "")[:200],
            chaos=bool(inst.get("chaos")))
        _destroy(inst["id"], rid, "lost")
        if job is None:
            d["status"] = "done"
        elif sum(1 for i in d["instances"] if not i.get("chaos")) >= a.attempts or spent >= a.budget:
            d["status"] = "abandoned"
            (BROKER / "outcome" / rid).write_text("abandoned")
            log("job_abandoned", reconstruction=rid, attempts=d["attempt"])
        else:
            _dispatch(a, s3, bucket, code_key, state, rid, job)


def _dispatch(a, s3, bucket, code_key, state, rid, job) -> None:
    d = state[rid]
    d["attempt"] += 1
    dispatch = f"{rid[:8]}-{d['attempt']}"
    prefix, exp = job["r2_prefix"], int(a.url_hours * 3600)
    existing = {o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/checkpoints/").get("Contents", [])}
    gets = {"capture.tar": f"{prefix}/capture.tar"} | {k.removeprefix(f"{prefix}/"): k for k in sorted(existing)}
    puts = {"status": f"{prefix}/status.json", "results": f"{prefix}/results.tar",
            "checkpoints/features.npz": f"{prefix}/checkpoints/features.npz",
            "checkpoints/matches.npz": f"{prefix}/checkpoints/matches.npz"}
    ticket = {"dispatch": dispatch, "reconstruction_id": rid, "capture_id": job["capture_id"],
              "options": job["options"], "mapper_config": job.get("mapper_config"),
              "get": {n: _presign(s3, bucket, "get_object", k, exp, dispatch) for n, k in gets.items()},
              "put": {n: _presign(s3, bucket, "put_object", k, exp, dispatch) for n, k in puts.items()}}
    ticket_key = f"{prefix}/tickets/{dispatch}.json"
    s3.put_object(Bucket=bucket, Key=ticket_key, Body=json.dumps(ticket).encode())
    urls = {name: _presign(s3, bucket, "get_object", key, exp, dispatch)
            for name, key in (("ticket", ticket_key), ("env", PAYLOAD_KEY), ("code", code_key))}
    inst = _launch(a, dispatch, _start_script(urls))
    if inst is None:
        d["status"] = "waiting_for_offer"
        d["attempt"] -= 1
        log("no_offer", reconstruction=rid, dispatch=dispatch)
        return
    d["instances"].append(inst | {"dispatch": dispatch})
    d["status"] = "launching"
    log("instance_launched", reconstruction=rid, dispatch=dispatch, **inst)


def _start_script(urls: dict) -> str:
    q = {k: shlex.quote(v) for k, v in urls.items()}
    return f"""T0=$(date +%s); log(){{ echo "[pf +$(( $(date +%s) - T0 ))s] $*"; }}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq curl zstd ca-certificates >/dev/null && log apt done
curl -sS --fail {q['env']} | zstd -dq -T0 | tar -x -C / && log environment ready
curl -sS --fail -o /tmp/remote_worker.py {q['code']} && log worker code ready
export LD_LIBRARY_PATH=/opt/pf_syslibs TORCH_HOME=/opt/torch_cache PYTHONUNBUFFERED=1
/opt/uv/venv/bin/python /tmp/remote_worker.py {q['ticket']}; log worker exited $?
sleep infinity"""


def _launch(a, dispatch: str, script: str) -> dict | None:
    if PROVIDER == "local":
        name = f"pf-fleet-{dispatch}"
        p = subprocess.run(["sg", "docker", "-c", " ".join(shlex.quote(x) for x in [
            "docker", "run", "-d", "--gpus", "all", "--name", name, "--label", "pf-fleet=1", "ubuntu:24.04",
            "/bin/bash", "-c", script])], capture_output=True, text=True)
        if p.returncode:
            log("offer_unavailable", dispatch=dispatch, offer="local", detail=p.stderr.strip()[-200:])
            return None
        return {"id": name, "offer": "local", "gpu": "local", "bid": 0.0, "geo": "home", "launched": time.time()}
    query = f"{OFFER_QUERY} {a.offer_filter}".strip()
    offers = json.loads(_vast(["search", "offers", query, "-i", "-o", "dph_total", "--raw"]) or "[]")
    for o in offers[:8]:
        bid = round(max(o.get("min_bid") or 0, 0.0) * a.bid_factor + 0.005, 3)
        if bid > a.max_bid:
            continue
        out = _vast(["create", "instance", str(o["id"]), "--image", "ubuntu:24.04", "--disk", "40",
                     "--bid_price", str(bid), "--label", f"pf-fleet-{dispatch}", "--cancel-unavail",
                     "--entrypoint", "/bin/bash", "--args", "-c", script])
        m = re.search(r"new_contract'?\"?:\s*(\d+)", out)
        if m:
            iid = int(m.group(1))
            # Hosts may charge for downloads: every new instance pulls the ~4.3 GB worker environment.
            down_cost = float(o.get("inet_down_cost") or 0) * 5.0
            return {"id": iid, "offer": o["id"], "gpu": o.get("gpu_name"), "bid": bid, "geo": o.get("geolocation"),
                    "download_cost": round(down_cost, 4), "launched": time.time()}
        log("offer_unavailable", dispatch=dispatch, offer=o["id"], detail=out.strip()[-120:])
    return None


def _instance(iid) -> dict | None:
    if PROVIDER == "local":
        p = subprocess.run(["sg", "docker", "-c", f"docker inspect -f '{{{{.State.Status}}}}' {iid}"],
                           capture_output=True, text=True)
        return {"actual_status": p.stdout.strip()} if p.returncode == 0 else None
    for x in json.loads(_vast(["show", "instances", "--raw"]) or "[]"):
        if x.get("id") == iid:
            return x
    return None


def _destroy(iid, rid: str, reason: str) -> None:
    if PROVIDER == "local":
        subprocess.run(["sg", "docker", "-c", f"docker logs {iid} > {BROKER}/{iid}.log 2>&1; docker rm -f {iid}"],
                       capture_output=True, text=True)
    else:
        subprocess.run(["vastai", "destroy", "instance", str(iid)], input="y\n", capture_output=True, text=True)
    log("instance_destroyed", reconstruction=rid, instance=iid, reason=reason)


def _cost(d: dict) -> float:
    return sum(i["bid"] * ((i.get("ended") or time.time()) - i["launched"]) / 3600 + i.get("download_cost", 0)
               for i in d["instances"])


def _presign(s3, bucket: str, method: str, key: str, expires: int, dispatch: str) -> str:
    url = s3.generate_presigned_url(method, Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
    log("url_issued", dispatch=dispatch, access="read" if method == "get_object" else "write", key=key,
        expires=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() + expires)))
    return url


def _publish_code(s3, bucket: str) -> str:
    code = (Path(__file__).resolve().parent / "remote_worker.py").read_bytes()
    key = f"code/remote_worker-{hashlib.sha256(code).hexdigest()[:12]}.py"
    s3.put_object(Bucket=bucket, Key=key, Body=code)
    return key


def _vast(args: list[str]) -> str:
    p = subprocess.run(["vastai", *args], capture_output=True, text=True, timeout=120)
    return p.stdout + p.stderr


def log(event: str, **kw) -> None:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "by": "broker", "event": event, **kw}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
