#!/usr/bin/env python3
"""Watchdog for the Placeframe + Insta360 mapping stack. Run every 5 min (user crontab, flock-guarded).

Detects crashes and stalls, kills hung work, applies known recoveries, restarts/resumes, keeps logs bounded,
and records every incident with evidence (log tail / traceback) in ~/pf_watchdog/incidents.jsonl.
Incidents it cannot fix mechanically are marked "needs_attention" for the supervising agent, which does
root-cause fixes and adjusts logging (see ops/README.md).

  python3 ops/watchdog.py            # one pass (what cron runs)
  python3 ops/watchdog.py --dry-run  # report only, change nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
PLACEFRAME = REPO.parent / "placeframe"
STATE_DIR = HOME / "pf_watchdog"
STATE = STATE_DIR / "state.json"
INCIDENTS = STATE_DIR / "incidents.jsonl"
JOBS = HOME / "pf_mapper/jobs"
COST = HOME / "pf_cost"
SCRATCH = Path("/mnt/hdd12t/placeframe_scratch")
VENV_PY = REPO / ".venv/bin/python"

STALL_MIN = 30           # no progress at all for this long ...
IDLE_CPU_PCT = 25        # ... and the worker is (nearly) idle -> hung
LOG_MAX_MB = 300         # truncate (keep tail) above this
LOG_KEEP_MB = 20
DISK_MIN_GB = 15
HOME_WORKER_PAUSED = HOME / "pf_fleet" / "home_worker_paused"  # present while a fleet run has the queue
MEM_MIN_AVAILABLE_GB = 3.0
SWAP_MAX_USED = 0.75
MEM_LOW_WITH_SWAP_GB = 8.0
RECON_IDLE_MAX_GB = 8.0
ONE_SHOT = ("initialize-", "create-database", "migrate-database", "auth-initializer")

sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------- helpers


def sh(cmd: str, timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {cmd}"


def spawn(args: list[str], cwd: Path, log: Path) -> int:
    """Start a long-running background process fully detached (own session, own log, no inherited pipes)."""
    with open(log, "a") as fh:
        p = subprocess.Popen(args, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True)
    return p.pid


def docker(cmd: str, timeout: int = 120) -> tuple[int, str]:
    return sh(f"sg docker -c {json.dumps('docker ' + cmd)}", timeout)


def tail(path: Path, lines: int = 60) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 200_000))
            return "\n".join(fh.read().decode(errors="replace").splitlines()[-lines:])
    except Exception:
        return ""


def traceback_in(text: str) -> str | None:
    i = text.rfind("Traceback (most recent call last)")
    return text[i:][-4000:] if i >= 0 else None


def procs(pattern: str) -> list[tuple[int, float, str]]:
    """(pid, cpu%, args) for processes whose command line matches pattern (excluding this watchdog)."""
    _, out = sh("ps -eo pid=,pcpu=,args=")
    res = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and re.search(pattern, parts[2]) and "watchdog.py" not in parts[2] and "awk" not in parts[2]:
            res.append((int(parts[0]), float(parts[1]), parts[2]))
    return res


class Watchdog:
    def __init__(self, dry: bool) -> None:
        self.dry = dry
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(STATE.read_text()) if STATE.exists() else {}
        self.now = time.time()
        self.actions: list[str] = []
        self._api = None

    @property
    def api(self):
        if self._api is None:
            from add_recording_to_map import PlaceframeApi

            self._api = PlaceframeApi()
        return self._api

    def incident(self, kind: str, summary: str, evidence: str = "", fixed: bool = False, **extra) -> None:
        key = f"{kind}:{extra.get('target', '')}"
        last = self.state.setdefault("last_incident", {}).get(key)
        if last and self.now - last < 3600 and not fixed:  # don't re-file the same open problem every 5 min
            return
        self.state["last_incident"][key] = self.now
        if self.dry:
            self.actions.append(f"(dry-run) incident {kind}: {summary}")
            return
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "summary": summary,
               "status": "auto_fixed" if fixed else "needs_attention", "evidence": evidence[-6000:], **extra}
        with open(INCIDENTS, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.actions.append(f"[{rec['status']}] {kind}: {summary}")

    def launch(self, desc: str, args: list[str], cwd: Path, log: Path) -> bool:
        if self.dry:
            self.actions.append(f"(dry-run) would {desc}")
            return True
        pid = spawn(args, cwd, log)
        self.actions.append(f"{desc}: pid {pid}")
        return True

    def act(self, desc: str, cmd: str, timeout: int = 600) -> bool:
        if self.dry:
            self.actions.append(f"(dry-run) would {desc}")
            return True
        rc, out = sh(cmd, timeout)
        self.actions.append(f"{desc}: {'ok' if rc == 0 else 'FAILED rc=' + str(rc)}")
        return rc == 0

    # ----------------------------------------------------------------------- checks
    def run(self) -> None:
        for check in (self.services, self.containers, self.server_builds, self.merge_jobs, self.study,
                      self.disk, self.memory, self.logs):
            try:
                check()
            except Exception as e:  # a broken check must never stop the others
                self.incident("watchdog-error", f"{check.__name__} raised {e!r}", target=check.__name__)
        self.state["last_run"] = self.now
        self.state["last_actions"] = self.actions[-30:]
        if not self.dry:
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2))
            tmp.replace(STATE)
        print(time.strftime("%H:%M:%S"), "; ".join(self.actions) or "all healthy", flush=True)

    def services(self) -> None:
        rc, code = sh("curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8150/")
        if code != "200":
            ev = tail(HOME / "pf_mapper/service.log", 80)
            ok = self.act("restart map-updater service", f"{REPO}/mapper/restart.sh", 120)
            self.incident("service-down", f"map updater not answering (http {code}); restarted", ev,
                          fixed=ok and not traceback_in(ev), target="mapper")
        rc, code = sh("curl -s -o /dev/null -w '%{http_code}' --max-time 15 "
                      f"$(grep ^PLACEFRAME_URL= {PLACEFRAME}/.credentials | cut -d= -f2)/auth/realms/placeframe-dev")
        if code != "200":
            self.incident("public-endpoint", f"public Placeframe URL returns {code} (Tailscale Funnel / gateway?)",
                          sh("tailscale funnel status 2>&1 | head -5")[1], target="funnel")
        if not procs(r"sampler\.py"):
            self.launch("restart resource sampler", ["python3", str(COST / "sampler.py")], COST, COST / "sampler.log")
        if not procs(r"http\.server 8130"):
            self.launch("restart map viewer", ["python3", "-m", "http.server", "8130"], HOME / "pf_x5/viewer",
                        STATE_DIR / "viewer.log")

    def containers(self) -> None:
        rc, out = docker("ps -a --filter name=placeframe- --format '{{.Names}}\t{{.State}}\t{{.Status}}'")
        bad = [l for l in out.splitlines() if l and not any(s in l.split("\t")[0] for s in ONE_SHOT)
               and l.split("\t")[1] not in ("running",)]
        # The home reconstructor is stopped on purpose while rented GPUs serve the queue (fleet runs).
        if HOME_WORKER_PAUSED.exists():
            bad = [l for l in bad if not l.startswith("placeframe-reconstructor-cuda-1")]
        if bad:
            ev = "\n".join(bad)
            for name in [l.split("\t")[0] for l in bad]:
                ev += f"\n--- {name} ---\n" + docker(f"logs --tail 40 {name}")[1]
            ok = self.act("bring the Placeframe stack back up", f"cd {PLACEFRAME} && sg docker -c 'uv run up --quiet-pull'", 900)
            self.incident("container-down", f"{len(bad)} Placeframe container(s) not running; ran `uv run up`", ev,
                          fixed=ok, target=",".join(l.split('\t')[0] for l in bad))

    def server_builds(self) -> None:
        recs = self.api.get("/reconstructions")
        seen = self.state.setdefault("builds", {})
        active_ids = set()
        for r in recs:
            if r["status"] in ("succeeded", "failed", "cancelled"):
                continue
            rid = r["id"]
            active_ids.add(rid)
            sig = f"{r['status']}|{r.get('progress_current')}|{r.get('updated_at')}"
            s = seen.setdefault(rid, {"sig": sig, "since": self.now})
            if s["sig"] != sig:
                seen[rid] = {"sig": sig, "since": self.now}
                continue
            if r["status"] == "queued":
                continue
            stalled_min = (self.now - s["since"]) / 60
            logs = docker("logs --since 40m placeframe-reconstructor-cuda-1 2>&1", 60)[1]
            if f"[{rid}] Reconstruction succeeded" in logs and "Token expired" in logs:
                self.recover_succeeded(rid, logs)
                continue
            if stalled_min < STALL_MIN:
                continue
            cpu = self.container_cpu("placeframe-reconstructor-cuda-1")
            last_line_age = self.container_log_age("placeframe-reconstructor-cuda-1")
            # Hung means stalled AND idle AND silent. A long bundle adjustment is silent for many minutes
            # but busy, so any CPU use (or an unreadable reading) or recent log output means: leave it alone.
            if cpu is None or cpu > IDLE_CPU_PCT or last_line_age < 20 * 60:
                continue
            ev = f"status={r['status']} progress={r.get('progress_current')}/{r.get('progress_total')} " \
                 f"no change for {stalled_min:.0f} min, reconstructor cpu={cpu}%, last log line {last_line_age / 60:.0f} min ago\n" \
                 + "\n".join(logs.splitlines()[-60:])
            if "Token expired" in logs or "401" in logs:
                ok = self.act("restart reconstructor (auth failure)", "sg docker -c 'docker restart placeframe-reconstructor-cuda-1'")
                self.mark_failed_and_retry(rid)
                self.incident("build-auth-stall", f"build {rid[:8]} stalled with auth errors; restarted worker + retried",
                              ev, fixed=ok, target=rid)
            else:
                ok = self.act("restart hung reconstructor", "sg docker -c 'docker restart placeframe-reconstructor-cuda-1'")
                self.mark_failed_and_retry(rid)
                self.incident("build-hung", f"build {rid[:8]} hung ({stalled_min:.0f} min no progress, idle worker); "
                              "restarted worker + retried. Root cause unknown — check evidence; add logging if unclear.",
                              ev, fixed=False, target=rid)
            seen.pop(rid, None)
        for rid in list(seen):
            if rid not in active_ids:
                seen.pop(rid)

    def recover_succeeded(self, rid: str, logs: str) -> None:
        """Worker finished but its final 'succeed' call was rejected (expired token): outputs are in MinIO."""
        rc, out = docker(f"exec placeframe-minio-1 sh -c 'mc cat l/dev-reconstructions/{rid}/sfm_model/frames.txt | head -3; "
                         f"mc cat l/dev-reconstructions/{rid}/sfm_model/points3D.txt | head -3'")
        if "Number of frames" not in out:
            self.incident("build-unreported", f"build {rid[:8]} logged success but outputs missing", logs[-3000:], target=rid)
            return
        # The localizer needs the map_* metrics (502 without them): recompute them from the stored model
        # with the reconstructor's own MetricsBuilder before the build is marked succeeded.
        if not self.dry:
            p = subprocess.run([sys.executable, str(Path(__file__).with_name("recover_map_metrics.py")), rid],
                               capture_output=True, text=True, timeout=1800)
            if p.returncode:
                self.incident("build-unreported", f"build {rid[:8]} finished unreported; metric recovery failed",
                              (p.stdout + p.stderr)[-3000:], target=rid)
                return
        sql = ("update reconstructions set status='succeeded', progress_current=null, progress_total=null, "
               "progress_attempt=null, manifest = jsonb_set(manifest::jsonb, '{metrics}', "
               "coalesce(manifest::jsonb->'metrics','{}'::jsonb) || "
               "'{\"recovered_by_watchdog\": true}'::jsonb) "
               f"where id='{rid}' and status not in ('succeeded','failed','cancelled');")
        ok = self.act(f"mark build {rid[:8]} succeeded (worker could not report)",
                      f"sg docker -c {json.dumps('docker exec placeframe-postgres-1 psql -U placeframe_owner -d placeframe -c ' + json.dumps(sql))}")
        self.incident("build-unreported", f"build {rid[:8]} finished but its succeed call failed (expired token); recovered",
                      logs[-2000:], fixed=ok, target=rid)

    def mark_failed_and_retry(self, rid: str) -> None:
        if self.dry:
            return
        sql = f"update reconstructions set status='failed' where id='{rid}' and status not in ('succeeded','failed','cancelled');"
        sh(f"sg docker -c {json.dumps('docker exec placeframe-postgres-1 psql -U placeframe_owner -d placeframe -c ' + json.dumps(sql))}")
        try:
            self.api._auth()
            self.api.client.put(f"{self.api.server}/reconstructions/{rid}/retry")
        except Exception as e:
            self.actions.append(f"retry {rid[:8]} failed: {e}")

    def container_cpu(self, name: str) -> float | None:
        rc, out = docker(f"stats --no-stream --format '{{{{.CPUPerc}}}}' {name}", 60)
        try:
            return float(out.strip().rstrip("%"))
        except ValueError:
            return None

    def container_log_age(self, name: str) -> float:
        rc, out = docker(f"logs -t --tail 1 {name} 2>&1", 30)
        m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", out.strip())
        if not m:
            return 0
        t = time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) - time.timezone
        return max(0, self.now - t)

    def merge_jobs(self) -> None:
        for jf in JOBS.glob("*/job.json"):
            try:
                j = json.loads(jf.read_text())
            except ValueError:
                self.incident("job-file-corrupt", f"{jf} unreadable", target=str(jf))
                continue
            if j.get("state") == "failed":
                log = tail(jf.parent / "log.txt", 120)
                self.incident("merge-failed", f"merge job {j['id']} ({j['filename']}) failed", traceback_in(log) or log,
                              target=j["id"])
                continue
            if j.get("state") != "running":
                continue
            wd = jf.parent / "work"
            log = jf.parent / "log.txt"
            age_min = (self.now - log.stat().st_mtime) / 60 if log.exists() else 0
            in_server_stage = (wd / ".done-merge").exists() and not (wd / ".done-reconstruct").exists()
            if in_server_stage or age_min < 45:
                continue  # server builds are covered by server_builds(); local stages print regularly
            kids = procs(r"add_recording_to_map\.py")
            busy = sum(c for _, c, _ in procs(r"insta360_|stitch_dualfisheye|vio_|insv_to|docker run"))
            if busy > 20:
                continue
            ev = f"job log silent for {age_min:.0f} min, pipeline cpu {busy:.0f}%\n" + tail(log, 60)
            for pid, _, _ in kids:
                self.act(f"kill hung merge job process {pid}", f"kill {pid}")
            self.act("restart map-updater (resumes the job after its last finished stage)", f"{REPO}/mapper/restart.sh", 120)
            self.incident("merge-hung", f"merge job {j['id']} hung in a local stage; killed and resumed", ev,
                          fixed=False, target=j["id"])

    def study(self) -> None:
        log = COST / "study.log"
        if not (COST / "study.py").exists() or not log.exists():
            return
        lines = tail(log, 3).splitlines()
        if lines and "study complete" in lines[-1]:
            return
        if not procs(r"pf_cost/study\.py|\bstudy\.py"):
            ev = tail(log, 80)
            self.launch("restart cost-study driver", [str(VENV_PY), "-u", str(COST / "study.py")], COST, COST / "study.log")
            self.incident("study-died", "cost-study driver not running with variants pending; restarted",
                          traceback_in(ev) or ev, fixed=not traceback_in(ev), target="study")
        errs = [l for l in self.new_lines(log) if ": ERROR " in l]
        if errs:
            self.incident("study-error", f"cost-study variant error: {errs[-1][:160]}", tail(log, 80), target="study-err")

    def new_lines(self, path: Path) -> list[str]:
        """Lines appended to a log since the previous watchdog pass (first pass: none — history is not news)."""
        offs = self.state.setdefault("log_offsets", {})
        size = path.stat().st_size if path.exists() else 0
        start = offs.get(str(path), size)
        if start > size:
            start = 0  # log was truncated/rotated
        offs[str(path)] = size
        if size <= start:
            return []
        with open(path, "rb") as fh:
            fh.seek(start)
            return fh.read().decode(errors="replace").splitlines()

    def memory(self) -> None:
        """Memory pressure starves everything (a 7,242-image merge pushed the host into swap for an hour: slow builds,
        failed DNS, an unreachable dashboard, then the kernel killed the reconstructor). Restart an idle reconstructor
        that holds memory between builds; report pressure with the biggest processes."""
        info = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        avail_gb = info["MemAvailable"] / 1e6
        swap_used = 1 - info["SwapFree"] / info["SwapTotal"] if info.get("SwapTotal") else 0.0
        _, mem = docker("stats --no-stream --format '{{.MemUsage}}' placeframe-reconstructor-cuda-1")
        recon_gb = _gb(mem.split("/")[0]) if "/" in mem else 0.0
        _, active = docker("exec placeframe-postgres-1 psql -U postgres -d placeframe -At -c "
                           "\"select count(*) from reconstructions where status not in ('succeeded','failed','cancelled')\"")
        idle = active.strip() == "0"
        if idle and recon_gb > RECON_IDLE_MAX_GB:
            fixed = self.act(f"restart idle reconstructor holding {recon_gb:.1f} GB",
                             "sg docker -c 'docker restart placeframe-reconstructor-cuda-1'", 180)
            self.incident("memory-held", f"idle reconstructor held {recon_gb:.1f} GB between builds; restarted",
                          mem, fixed=fixed, target="reconstructor-memory")
        # Swap stays occupied long after an episode, so a full swap only counts while memory is also short.
        if avail_gb < MEM_MIN_AVAILABLE_GB or (swap_used > SWAP_MAX_USED and avail_gb < MEM_LOW_WITH_SWAP_GB):
            top = sh("ps -eo rss,comm --sort=-rss | head -6")[1]
            self.incident("memory-pressure", f"only {avail_gb:.1f} GB memory available, swap {swap_used:.0%} used "
                          f"(reconstructor {recon_gb:.1f} GB{'' if idle else ', build running'})",
                          top, target="memory")

    def disk(self) -> None:
        free = shutil.disk_usage(HOME).free / 1e9
        if free >= DISK_MIN_GB:
            return
        moved = []
        if SCRATCH.exists():
            for jf in JOBS.glob("*/job.json"):
                try:
                    j = json.loads(jf.read_text())
                except ValueError:
                    continue
                wd = jf.parent / "work"
                if j.get("state") == "succeeded" and wd.exists() and not wd.is_symlink():
                    dst = SCRATCH / "jobs" / j["id"]
                    if self.act(f"archive finished job {j['id']} to HDD",
                                f"mkdir -p {dst.parent} && cp -r {wd} {dst} && rm -rf {wd} && ln -s {dst} {wd}", 3600):
                        moved.append(j["id"])
        # Docker's build cache (multi-GB CUDA compile stages) is the other large, safe-to-drop consumer on this
        # disk: it only makes the next image build slower. Images and volumes (MinIO, Postgres) are never touched.
        pruned = self.act("prune docker build cache", "sg docker -c 'docker builder prune -af'", 900)
        after = shutil.disk_usage(HOME).free / 1e9
        self.incident("disk-low", f"only {free:.0f} GB free; archived {len(moved)} finished job(s) to {SCRATCH}, "
                      f"pruned build cache; now {after:.0f} GB free",
                      sh("du -sh ~/pf_* 2>/dev/null | sort -h | tail -8")[1] + "\n" + sh("df -h /")[1],
                      fixed=after >= DISK_MIN_GB and (bool(moved) or pruned), target="disk")

    def logs(self) -> None:
        """Keep logs bounded; flag a log that grows too fast so its verbosity gets cleaned up at the source."""
        files = [HOME / "pf_mapper/service.log", COST / "study.log", COST / "sampler.log"] + list(JOBS.glob("*/log.txt"))
        sizes = self.state.setdefault("log_sizes", {})
        for f in files:
            if not f.exists():
                continue
            mb = f.stat().st_size / 1e6
            prev = sizes.get(str(f))
            sizes[str(f)] = [mb, self.now]
            if prev and self.now > prev[1]:
                rate = (mb - prev[0]) / ((self.now - prev[1]) / 3600)
                if rate > 100:
                    self.incident("log-noisy", f"{f.name} grows {rate:.0f} MB/h — reduce verbosity at the source",
                                  tail(f, 30), target=str(f))
            if mb > LOG_MAX_MB:
                self.act(f"truncate {f} (keep last {LOG_KEEP_MB} MB)",
                         f"tail -c {LOG_KEEP_MB * 1_000_000} {f} > {f}.tmp && cat {f}.tmp > {f} && rm {f}.tmp")
        # Docker json logs: size is only visible to root; ask docker for a recent-line rate instead.
        rc, n = docker("logs --since 5m placeframe-reconstructor-cuda-1 2>&1 | wc -l", 60)
        if n.strip().isdigit() and int(n) > 200_000:
            self.incident("log-noisy", f"reconstructor printed {int(n):,} lines in 5 min — reduce COLMAP/glog verbosity",
                          target="reconstructor-docker-log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    Watchdog(a.dry_run).run()



def _gb(text: str) -> float:
    text = text.strip()
    for unit, f in (("GiB", 1.074), ("MiB", 1.074e-3), ("KiB", 1.024e-6), ("GB", 1.0), ("MB", 1e-3), ("kB", 1e-6), ("B", 1e-9)):
        if text.endswith(unit):
            return float(text[:-len(unit)]) * f
    return 0.0

if __name__ == "__main__":
    main()
