#!/usr/bin/env python3
"""Worker fleet for the Placeframe build queue: scales reconstructor workers with the queue depth.

Workers pull leases from the API (worker-pull queue, no orchestrator), so a worker can disappear at any time:
the API re-queues its job once the lease stops heart-beating (LEASE_TIMEOUT_MINUTES), and with
RECONSTRUCTOR_CHECKPOINTS=1 the next worker resumes after the last finished stage (features, matches).

Providers:
  local  extra reconstructor containers on this machine (same image, config and secrets as the main worker).
         --chaos N kills a random busy worker every N minutes and replaces it, to test preemption recovery.
  vast   interruptible Vast.ai instances (bid price). DRY RUN unless --live is given; needs VAST_API_KEY and the
         remote-worker settings in ~/pf_fleet/vast.env (API/auth URLs reachable from the internet, storage
         endpoint + keys, worker image). Guards: --max-workers, --max-price, --budget, idle shutdown.

  python fleet/controller.py --provider local --max-workers 2 --chaos 6 --until-empty
  python fleet/controller.py --provider vast --max-workers 8 --max-price 0.25 --budget 5     # dry run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from add_recording_to_map import PlaceframeApi  # noqa: E402

STATE_DIR = Path.home() / "pf_fleet"
TERMINAL = {"succeeded", "failed", "cancelled"}
BASE_WORKER = "placeframe-reconstructor-cuda-1"
LOCAL_PREFIX = "pf-worker-"
# Extra modules mounted into every worker on top of the image (the local patches).
PATCHED = ["colmap.py", "main.py", "pairs.py", "prior_mapper.py", "run_reconstruction.py", "checkpoints.py",
           "progress_publisher.py"]
PATCH_SRC = Path.home() / "github/placeframe/docker/reconstructor/src/reconstructor"
PATCH_DST = "/opt/uv/venv/lib/python3.13/site-packages/reconstructor"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["local", "vast"], required=True)
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--image", help="worker image (default: the main worker's image)")
    ap.add_argument("--chaos", type=float, default=0, help="local: kill a busy worker every N minutes")
    ap.add_argument("--max-price", type=float, default=0.25, help="vast: max $/h bid per instance")
    ap.add_argument("--budget", type=float, default=5.0, help="vast: stop creating instances past this $ spend")
    ap.add_argument("--live", action="store_true", help="vast: actually rent instances (otherwise dry run)")
    ap.add_argument("--until-empty", action="store_true", help="exit (and remove workers) when the queue drains")
    ap.add_argument("--interval", type=float, default=20)
    a = ap.parse_args()
    STATE_DIR.mkdir(exist_ok=True)
    provider = LocalProvider(a) if a.provider == "local" else VastProvider(a)
    Controller(provider, a).run()


class Controller:
    def __init__(self, provider, a) -> None:
        self.p, self.a = provider, a
        self.api = PlaceframeApi()
        self.log_path = STATE_DIR / "events.jsonl"
        self.last_chaos = time.time()

    def run(self) -> None:
        self.event("start", provider=self.a.provider, max_workers=self.a.max_workers)
        while True:
            recs = self.api.get("/reconstructions")
            queued = sum(r["status"] == "queued" for r in recs)
            active = sum(r["status"] not in TERMINAL and r["status"] != "queued" for r in recs)
            workers = self.p.list()
            # The main local worker also serves the queue; extra workers cover the rest.
            want = min(self.a.max_workers, max(0, queued + active - self.p.base_capacity()))
            if len(workers) < want:
                for _ in range(want - len(workers)):
                    self.event("scale_up", **self.p.create())
            elif len(workers) > want:
                for w in self.p.idle(workers)[: len(workers) - want]:
                    self.event("scale_down", **self.p.destroy(w))
            if self.a.chaos and time.time() - self.last_chaos > self.a.chaos * 60:
                busy = self.p.busy(self.p.list())
                if busy:
                    victim = random.choice(busy)
                    self.event("chaos_kill", **self.p.destroy(victim, kill=True))
                self.last_chaos = time.time()
            self.p.account()
            if self.a.until_empty and queued == 0 and active == 0:
                for w in self.p.list():
                    self.event("scale_down", **self.p.destroy(w))
                self.event("done")
                return
            time.sleep(self.a.interval)

    def event(self, kind: str, **fields) -> None:
        rec = {"t": round(time.time(), 1), "event": kind} | fields
        print(time.strftime("%H:%M:%S"), kind, {k: v for k, v in fields.items() if k != "cmd"}, flush=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


class LocalProvider:
    """Extra worker containers cloned from the main worker's configuration."""

    def __init__(self, a) -> None:
        self.a = a
        cfg = json.loads(docker(f"inspect {BASE_WORKER}"))[0]
        self.image = a.image or cfg["Config"]["Image"]
        keep = ("AUTH_", "PRIVATE_KEY_PATH", "RECONSTRUCTIONS_BUCKET", "CAPTURES_BUCKET", "MINIO_", "API_INTERNAL_URL")
        env = [e for e in cfg["Config"]["Env"] if e.startswith(keep)] + ["RECONSTRUCTOR_CHECKPOINTS=1"]
        self.env_file = STATE_DIR / "local-worker.env"
        fd = os.open(self.env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, ("\n".join(env) + "\n").encode())
        os.close(fd)
        self.network = next(iter(cfg["NetworkSettings"]["Networks"]))
        self.extra_hosts = cfg["HostConfig"].get("ExtraHosts") or []
        self.secrets = next(m["Name"] for m in cfg["Mounts"] if m["Destination"] == "/run/secrets")
        self.mapper_cfg = next((m["Source"] for m in cfg["Mounts"] if m["Destination"] == "/etc/placeframe/mapper.json"),
                               None)

    def base_capacity(self) -> int:
        return 1  # the main worker

    def list(self) -> list[str]:
        out = docker(f"ps --filter name={LOCAL_PREFIX} --format '{{{{.Names}}}}'")
        return sorted(n for n in out.split() if n)

    def create(self) -> dict:
        n = 1
        existing = set(docker(f"ps -a --filter name={LOCAL_PREFIX} --format '{{{{.Names}}}}'").split())
        while f"{LOCAL_PREFIX}{n}" in existing:
            n += 1
        name = f"{LOCAL_PREFIX}{n}"
        mounts = [f"-v {PATCH_SRC / f}:{PATCH_DST}/{f}:ro" for f in PATCHED if (PATCH_SRC / f).exists()]
        if self.mapper_cfg:
            mounts.append(f"-v {self.mapper_cfg}:/etc/placeframe/mapper.json:ro")
        hosts = " ".join(f"--add-host {h}" for h in self.extra_hosts)
        cmd = (f"run -d --name {name} --gpus all --network {self.network} {hosts} --env-file {self.env_file} "
               f"-v {self.secrets}:/run/secrets:ro {' '.join(mounts)} --label pf.fleet=1 {self.image}")
        docker(cmd)
        return {"worker": name, "image": self.image}

    def destroy(self, name: str, kill: bool = False) -> dict:
        docker(f"{'kill' if kill else 'stop -t 30'} {name}")
        docker(f"rm -f {name}")
        return {"worker": name, "killed": kill}

    def busy(self, workers: list[str]) -> list[str]:
        # Search the whole log for the last lease event: progress output floods any fixed-size tail.
        return [w for w in workers if _holds_lease(docker(
            f"logs {w} 2>&1 | grep -E 'Acquired lease|Reconstruction (succeeded|failed)' | tail -1"))]

    def idle(self, workers: list[str]) -> list[str]:
        return [w for w in workers if w not in self.busy(workers)]

    def account(self) -> None:
        pass


class VastProvider:
    """Interruptible Vast.ai instances. Commands are printed (dry run) unless --live."""

    def __init__(self, a) -> None:
        self.a = a
        self.cfg_path = STATE_DIR / "vast.env"
        self.cfg = _read_env(self.cfg_path)
        self.state_path = STATE_DIR / "vast_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"instances": {}, "spent": 0.0}
        missing = [k for k in ("WORKER_IMAGE", "API_INTERNAL_URL", "AUTH_TOKEN_URL", "AUTH_CLIENT_ID",
                               "MINIO_ENDPOINT_URL", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "WORKER_PRIVATE_KEY_B64")
                   if not self.cfg.get(k)]
        if a.live and (missing or not os.environ.get("VAST_API_KEY")):
            sys.exit(f"--live needs VAST_API_KEY and {self.cfg_path} with: {', '.join(missing) or 'ok'}")
        self.missing = missing

    def base_capacity(self) -> int:
        return 0

    def list(self) -> list[str]:
        if not self.a.live:
            return list(self.state["instances"])
        rows = json.loads(vast("show instances --raw") or "[]")
        ours = {str(r["id"]): r for r in rows if (r.get("label") or "").startswith("pf-worker")}
        for iid in list(self.state["instances"]):
            if iid not in ours:
                self.state["instances"].pop(iid)  # destroyed or reclaimed by the host
        for iid, r in ours.items():
            if r.get("actual_status") in ("exited", "offline") or r.get("intended_status") == "stopped":
                vast(f"destroy instance {iid}")  # outbid/paused instances still bill storage: replace, don't wait
                self.state["instances"].pop(iid, None)
        self._save()
        return [i for i in ours if i in self.state["instances"]]

    def create(self) -> dict:
        if self.state["spent"] >= self.a.budget:
            return {"skipped": f"budget {self.a.budget} reached"}
        query = (f"gpu_name in [RTX_3090,RTX_4090,RTX_A5000,L4] num_gpus=1 reliability>0.97 inet_down>300 "
                 f"disk_space>=60 cuda_vers>=12.8 dph<={self.a.max_price} rentable=true")
        env = " ".join(f"-e {k}={shlex.quote(self.cfg.get(k, f'<{k}>'))}" for k in
                       ("API_INTERNAL_URL", "AUTH_TOKEN_URL", "AUTH_CLIENT_ID", "MINIO_ENDPOINT_URL", "MINIO_ACCESS_KEY",
                        "MINIO_SECRET_KEY", "CAPTURES_BUCKET", "RECONSTRUCTIONS_BUCKET", "WORKER_PRIVATE_KEY_B64"))
        onstart = ("mkdir -p /run/secrets && echo $WORKER_PRIVATE_KEY_B64 | base64 -d > /run/secrets/worker-private-key.pem "
                   "&& export PRIVATE_KEY_PATH=/run/secrets/worker-private-key.pem RECONSTRUCTOR_CHECKPOINTS=1 "
                   "&& /usr/local/bin/entrypoint.sh")
        if not self.a.live:
            cmd = (f"vastai search offers '{query}' -i -o dph --raw | <pick cheapest> ; vastai create instance <offer> "
                   f"--image {self.cfg.get('WORKER_IMAGE', '<WORKER_IMAGE>')} --bid_price <~1.1x min bid, <= {self.a.max_price}> "
                   f"--disk 60 --label pf-worker --env '<{len(env.split(' -e '))} vars from vast.env>' --onstart-cmd '<writes key, runs worker>'")
            iid = f"dry-{len(self.state['instances']) + 1}"
            self.state["instances"][iid] = {"created": time.time(), "dph": self.a.max_price, "dry_run": True}
            self._save()
            return {"worker": iid, "dry_run": True, "cmd": cmd, "missing_config": self.missing}
        offers = json.loads(vast(f"search offers {shlex.quote(query)} -i -o dph --raw") or "[]")
        if not offers:
            return {"skipped": "no offers under max price"}
        offer = offers[0]
        bid = round(min(self.a.max_price, max(offer.get("min_bid", offer["dph_total"]) * 1.1, 0.01)), 3)
        out = vast(f"create instance {offer['id']} --image {shlex.quote(self.cfg['WORKER_IMAGE'])} --bid_price {bid} "
                   f"--disk 60 --label pf-worker --env {shlex.quote(env)} --onstart-cmd {shlex.quote(onstart)} --raw")
        iid = str(json.loads(out)["new_contract"])
        self.state["instances"][iid] = {"created": time.time(), "dph": bid, "gpu": offer.get("gpu_name")}
        self._save()
        return {"worker": iid, "bid": bid, "gpu": offer.get("gpu_name")}

    def destroy(self, iid: str, kill: bool = False) -> dict:
        if self.a.live and not iid.startswith("dry-"):
            vast(f"destroy instance {iid}")
        self.state["instances"].pop(iid, None)
        self._save()
        return {"worker": iid}

    def busy(self, workers: list[str]) -> list[str]:
        return workers  # leases are tracked by the API; idle detection uses the queue depth

    def idle(self, workers: list[str]) -> list[str]:
        return sorted(workers, key=lambda i: self.state["instances"].get(i, {}).get("created", 0))

    def account(self) -> None:
        now = time.time()
        for inst in self.state["instances"].values():
            last = inst.get("billed_to", inst["created"])
            self.state["spent"] += inst["dph"] * (now - last) / 3600
            inst["billed_to"] = now
        self._save()

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1))


def _holds_lease(log: str) -> bool:
    acquired = log.rfind("Acquired lease")
    return acquired >= 0 and max(log.rfind("Reconstruction succeeded"), log.rfind("Reconstruction failed")) < acquired


def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def docker(args: str) -> str:
    p = subprocess.run(["sg", "docker", "-c", f"docker {args}"], capture_output=True, text=True)
    if p.returncode and not args.startswith(("kill", "stop", "rm")):
        raise RuntimeError(f"docker {args.split()[0]} failed: {p.stderr.strip()[:300]}")
    return p.stdout


def vast(args: str) -> str:
    p = subprocess.run(f"vastai {args}", shell=True, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"vastai {args.split()[0]} failed: {p.stderr.strip()[:300]}")
    return p.stdout


if __name__ == "__main__":
    main()
