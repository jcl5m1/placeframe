"""Compute dashboard telemetry for the map updater: machine load, Placeframe server builds, local merge jobs and
the cost-study queue, with progress, rate (items/s) and ETA.

Progress sources (no extra instrumentation needed):
  * Placeframe reconstructions report progress_current/progress_total through the API (reconstructor leases).
  * Local pipeline tools print counters ("stitched 2000/5777", "frame 300/5777", OpenVIO "image=600"); the last
    counter in the current stage's log section is parsed.
  * Finished stages leave .done-<stage> markers with their seconds; seconds per recording-second from finished
    jobs give the remaining-stage estimate for new jobs.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

SERVER_STAGES = ["queued", "downloading", "extracting_features", "training_opq_matrix", "training_product_quantizer",
                 "matching_features", "verifying_geometry", "reconstructing", "uploading"]
TERMINAL = {"succeeded", "failed", "cancelled"}
COST = Path.home() / "pf_cost"
COUNTER = re.compile(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)")
OPENVIO_IMG = re.compile(r"\bimage=(\d+)")


class Rate:
    """items/s over a sliding window of (time, done) samples."""

    def __init__(self) -> None:
        self.s: dict[str, deque] = {}
        self.changed: dict[str, tuple[float, float]] = {}  # key -> (value, time it last changed)

    def since_change(self, key: str) -> float | None:
        c = self.changed.get(key)
        return time.time() - c[1] if c else None

    def update(self, key: str, done: float) -> float | None:
        q = self.s.setdefault(key, deque(maxlen=60))
        now = time.time()
        if key not in self.changed or self.changed[key][0] != done:
            self.changed[key] = (done, now)
        if q and done < q[-1][1]:
            q.clear()  # counter restarted (new stage)
        q.append((now, done))
        old = [p for p in q if now - p[0] <= 180] or list(q)
        if len(old) < 2 or old[-1][0] - old[0][0] < 8:
            return None
        return (old[-1][1] - old[0][1]) / (old[-1][0] - old[0][0])


class Telemetry:
    def __init__(self, jobs_dir: Path, api_factory) -> None:
        self.jobs_dir = jobs_dir
        self.api_factory = api_factory
        self.api = None
        self.rate = Rate()
        self.history: deque = deque(maxlen=180)  # 15 min at 5 s
        self.snapshot: dict = {}
        self.containers: dict = {}
        self.names: dict[str, str] = {}
        self.cams: dict[str, int] = {}  # capture session -> cameras per rig frame
        self._cpu_prev = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loop
    gpu_util: float | None = None

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._docker_loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                machine = self._machine()
                self.gpu_util = machine.get("gpu_util_pct")
                snap = {"t": time.time(), "machine": machine, "containers": self.containers,
                        "server": self._server(), "jobs": self._jobs(), "study": self._study(),
                        "watchdog": _watchdog()}
                m = snap["machine"]
                self.history.append({"t": snap["t"], "cpu": m["cpu_pct"], "gpu": m["gpu_util_pct"],
                                     "ram": m["ram_pct"], "gpu_mem": m["gpu_mem_pct"]})
                snap["history"] = list(self.history)
                with self._lock:
                    self.snapshot = snap
            except Exception as e:  # keep the dashboard alive whatever happens
                # Keep the last good snapshot but say it is stale (the page shows a warning), and drop the build
                # lists, so a finished build can never linger on screen as "in progress".
                if getattr(getattr(e, "response", None), "status_code", None) == 401:
                    self.api = None  # re-authenticate on the next pass
                with self._lock:
                    self.snapshot = {**self.snapshot, "server": [], "jobs": [], "error": repr(e)[:300],
                                     "error_t": time.time()}
            time.sleep(5)

    def _docker_loop(self) -> None:
        while True:
            try:
                out = subprocess.run(["docker", "stats", "--no-stream", "--format",
                                      "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"], capture_output=True, text=True,
                                     timeout=60).stdout
                c = {}
                for line in out.splitlines():
                    name, cpu, mem = line.split("\t")
                    if name.startswith("placeframe-") or name.startswith("ov"):
                        c[name.replace("placeframe-", "").rsplit("-1", 1)[0]] = {
                            "cpu_pct": float(cpu.rstrip("%") or 0), "mem": mem.split("/")[0].strip()}
                self.containers = c
            except Exception:
                pass
            time.sleep(15)

    def get(self) -> dict:
        with self._lock:
            return self.snapshot

    # ------------------------------------------------------------------ machine
    def _machine(self) -> dict:
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        tot, idle = sum(v), v[3] + v[4]
        cpu = 0.0
        if self._cpu_prev:
            dt, di = tot - self._cpu_prev[0], idle - self._cpu_prev[1]
            cpu = 100 * (1 - di / max(1, dt))
        self._cpu_prev = (tot, idle)
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split()[1].isdigit()}
        ram_tot, ram_av = mi["MemTotal"] / 1048576, mi["MemAvailable"] / 1048576
        g = {"gpu_util_pct": None, "gpu_mem_pct": None, "gpu_mem_gb": None, "gpu_mem_total_gb": None,
             "gpu_power_w": None, "gpu_temp_c": None, "gpu_name": None}
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            name, util, mu, mt, pw, tc = [x.strip() for x in o.splitlines()[0].split(",")]
            g = {"gpu_name": name, "gpu_util_pct": float(util), "gpu_mem_gb": float(mu) / 1024,
                 "gpu_mem_total_gb": float(mt) / 1024, "gpu_mem_pct": 100 * float(mu) / float(mt),
                 "gpu_power_w": float(pw), "gpu_temp_c": float(tc)}
        except Exception:
            pass
        du = shutil.disk_usage(Path.home())
        load1 = float(open("/proc/loadavg").read().split()[0])
        return {"cpu_pct": round(cpu, 1), "cpu_cores": os.cpu_count(), "load1": load1,
                "ram_used_gb": round(ram_tot - ram_av, 1), "ram_total_gb": round(ram_tot, 1),
                "ram_pct": round(100 * (1 - ram_av / ram_tot), 1),
                "disk_free_gb": round(du.free / 1e9, 1), "disk_pct": round(100 * du.used / du.total, 1), **g}

    # ------------------------------------------------------------------ Placeframe server builds
    def _server(self) -> list[dict]:
        if self.api is None:
            self.api = self.api_factory()
        recs = self.api.get("/reconstructions")
        active = [r for r in recs if r["status"] not in TERMINAL]
        recent = sorted([r for r in recs if r["status"] in TERMINAL], key=lambda r: r.get("updated_at") or "")[-3:]
        missing = [r["capture_session_id"] for r in active + recent if r["capture_session_id"] not in self.names]
        if missing:
            for cs in self.api.get("/capture_sessions"):
                self.names[cs["id"]] = cs.get("name") or cs["id"][:8]
        out = []
        for r in active + recent[::-1]:
            st = r["status"]
            cur, tot = r.get("progress_current"), r.get("progress_total")
            if st == "reconstructing" and tot:
                # The incremental mapper reports registered rig FRAMES against a total of IMAGES: convert the total
                # to frames using the capture's cameras per rig frame. The fast prior mapper reports images, so
                # only convert when the count fits in the frame total.
                ncam = self._cameras(r["capture_session_id"])
                if ncam > 1 and (cur is None or cur <= tot // ncam):
                    tot = tot // ncam
            key = f"srv-{r['id']}-{st}"
            rate = self.rate.update(key, cur) if (cur is not None and st not in TERMINAL) else None
            eta = (tot - cur) / rate if (rate and rate > 0 and tot and cur is not None) else None
            health = None
            if st not in TERMINAL:
                act = self._worker_activity()
                health = _health(rate, self.rate.since_change(key) if cur is not None else None, act["cpu_cores"],
                                 act["log_age_s"], self.gpu_util) | act
            out.append({"id": r["id"], "name": self.names.get(r["capture_session_id"], "?"), "status": st,
                        "stage_index": SERVER_STAGES.index(st) + 1 if st in SERVER_STAGES else None,
                        "stage_count": len(SERVER_STAGES), "current": cur, "total": tot, "rate": rate,
                        "stage_eta_s": eta, "created": r.get("created_at"), "updated": r.get("updated_at"),
                        "active": st not in TERMINAL, "health": health})
        return out

    def _worker_activity(self) -> dict:
        """What the reconstructor is doing: CPU in use, last meaningful log step and how long ago it logged."""
        cpu = (self.containers.get("reconstructor-cuda") or {}).get("cpu_pct")
        step, step_age, log_age = None, None, None
        try:
            out = subprocess.run(["docker", "logs", "--timestamps", "--tail", "400", "placeframe-reconstructor-cuda-1"],
                                 capture_output=True, text=True, timeout=10)
            lines = (out.stdout + out.stderr).splitlines() if out.returncode == 0 else []
            now = time.time()
            if lines:
                log_age = now - _docker_ts(lines[-1])
            for ln in reversed(lines):
                msg = ln.split(" ", 1)[1] if " " in ln else ln
                if STEP_LINE.search(msg):
                    step, step_age = _step_text(msg), now - _docker_ts(ln)
                    break
        except Exception:
            pass
        return {"cpu_cores": None if cpu is None else round(cpu / 100, 1), "last_step": step,
                "last_step_age_s": step_age, "log_age_s": log_age}

    def _cameras(self, cs_id: str) -> int:
        if cs_id not in self.cams:
            try:
                self.api._auth()
                r = self.api.client.get(f"{self.api.server}/capture_sessions/{cs_id}/manifest.json")
                r.raise_for_status()
                rigs = r.json().get("rigs") or []
                self.cams[cs_id] = max(1, len(rigs[0]["cameras"])) if rigs else 1
            except Exception:
                self.cams[cs_id] = 1
        return self.cams[cs_id]

    # ------------------------------------------------------------------ local merge jobs
    def _stage_ratios(self) -> dict[str, float]:
        """Seconds of processing per second of recording, per stage, from finished jobs."""
        acc: dict[str, list[float]] = {}
        for wd in self.jobs_dir.glob("*/work"):
            rec = _recording_s(wd)
            if not rec:
                continue
            for m in wd.glob(".done-*"):
                try:
                    acc.setdefault(m.name[6:], []).append(json.loads(m.read_text())["seconds"] / rec)
                except Exception:
                    pass
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def _jobs(self) -> list[dict]:
        ratios = self._stage_ratios()
        out = []
        for jf in sorted(self.jobs_dir.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
            j = _json(jf)
            if not j:
                continue
            wd = jf.parent / "work"
            odo = j.get("odometry") or "openvio"
            names = ["calibrate", f"{odo}-export", f"{odo}-run", "stitch", "views", "capture", "register", "merge",
                     "reconstruct", "publish", "geolocate"]
            if j.get("new_map"):  # a recording built as a map of its own skips registration and merging
                names = [n for n in names if n not in ("register", "merge")]
            done = [s for s in names if (wd / f".done-{s}").exists()]
            cur = next((s for s in names if s not in done), None) if j["state"] == "running" else None
            rec_s = _recording_s(wd) or _guess_recording_s(Path(j["file"]))
            item = {"id": j["id"], "file": j["filename"], "state": j["state"], "stages": names, "done": done,
                    "current": cur, "recording_s": rec_s, "current_progress": None, "rate": None, "stage_eta_s": None,
                    "elapsed_s": (time.time() - j["started"]) if j.get("started") and j["state"] == "running" else None}
            if cur:
                prog = _stage_progress(jf.parent / "log.txt", cur, wd)
                if cur == "reconstruct":
                    rid = _json(wd / "reconstruct.json").get("reconstruction_id")
                    srv = next((s for s in (self.snapshot.get("server") or []) if s["id"] == rid), None)
                    if srv:
                        prog = (srv["current"], srv["total"], f"server: {srv['status']}")
                if prog and prog[0] is not None and prog[1]:
                    item["current_progress"] = {"done": prog[0], "total": prog[1], "label": prog[2]}
                    key = f"job-{j['id']}-{cur}-{re.sub(r'[0-9,./ ]+', '', prog[2])}"
                    rate = self.rate.update(key, prog[0])
                    item["rate"] = rate
                    if cur == "reconstruct" and srv:
                        item["health"] = srv.get("health")
                    else:
                        log = jf.parent / "log.txt"
                        log_age = time.time() - log.stat().st_mtime if log.exists() else None
                        item["health"] = _health(rate, self.rate.since_change(key), None, log_age)
                    if rate and rate > 0:
                        item["stage_eta_s"] = (prog[1] - prog[0]) / rate
                # remaining stages after the current one, from measured per-recording-second ratios
                if rec_s and ratios:
                    rest = [s for s in names[names.index(cur) + 1:] if s not in ("reconstruct", "publish")]
                    item["after_stage_eta_s"] = sum(ratios.get(s, 0) * rec_s for s in rest)
            out.append(item)
        return out

    # ------------------------------------------------------------------ cost study queue
    def _study(self) -> dict | None:
        study = COST / "study.py"
        if not study.exists():
            return None
        m = re.search(r"ORDER = \[(.*?)\]", study.read_text())
        order = [x.strip().strip("\"'") for x in m.group(1).split(",")] if m else []
        done = {p.stem for p in (COST / "results").glob("*.json")}
        log = (COST / "study.log").read_text(errors="replace").splitlines() if (COST / "study.log").exists() else []
        current = None
        for line in reversed(log):
            mm = re.search(r"\s(\w+): capture \S+, reconstruction (\S+),", line)
            if mm and mm.group(1) not in done:
                current = {"variant": mm.group(1), "reconstruction_id": mm.group(2)}
                break
        items = []
        for v in order:
            vj = _json(COST / "variants" / v / "variant.json")
            st = "done" if v in done else ("running" if current and current["variant"] == v else "pending")
            r = _json(COST / "results" / f"{v}.json")
            items.append({"variant": v, "state": st, "images": vj.get("images"), "frames": vj.get("frames"),
                          "total_s": r.get("total_s")})
        waiting = (not current) and any(i["state"] == "pending" for i in items) and not (log and "study complete" in log[-1])
        return {"items": items, "current": current, "waiting_for_idle_server": bool(waiting), "last_log": log[-1] if log else ""}


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _recording_s(wd: Path) -> float | None:
    meta = _json(wd / "fisheye/work/meta.json")
    ts = meta.get("ts")
    return (ts[-1] - ts[0]) / 1000 + 0.5 if ts else None


def _guess_recording_s(f: Path) -> float | None:
    try:
        return f.stat().st_size / 7.6e6  # X5 .insv ~7.6 MB per second of video
    except Exception:
        return None


def _stage_progress(log: Path, stage: str, wd: Path):
    """(done, total, label) from the latest counter printed in the current stage's log section."""
    if not log.exists():
        return None
    with open(log, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 64000))
        text = fh.read().decode(errors="replace")
    sec = text.rsplit(f"== STAGE {stage}", 1)[-1].splitlines()
    if stage.endswith("-run"):
        total = sum(1 for _ in open(wd / "openvio/cam_timestamps.txt")) if (wd / "openvio/cam_timestamps.txt").exists() else None
        for line in reversed(sec):
            m = OPENVIO_IMG.search(line)
            if m and total:
                return int(m.group(1)), total, "frames"
    for line in reversed(sec[-200:]):
        if "registered" in line and "/" in line and stage == "register":
            pass
        m = COUNTER.search(line)
        if m:
            d, t = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
            if 0 <= d <= t and t > 1:
                label = re.sub(r"\s+", " ", line.strip())[:60]
                return d, t, label
    return None


def _watchdog() -> dict:
    d = Path.home() / "pf_watchdog"
    st = _json(d / "state.json")
    open_inc, recent = 0, []
    inc = d / "incidents.jsonl"
    if inc.exists():
        rows = [json.loads(l) for l in inc.read_text().splitlines()[-200:] if l.strip()]
        cutoff = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 86400))
        day = [r for r in rows if r.get("ts", "") >= cutoff]
        open_inc = sum(1 for r in day if r.get("status") == "needs_attention")
        recent = [{"ts": r["ts"], "kind": r["kind"], "status": r["status"], "summary": r["summary"][:140]} for r in day[-5:]]
        # Open problems for the dashboard banner: unresolved, one per kind + summary (re-raised incidents collapse).
        latest: dict[tuple, dict] = {}
        for r in day:
            latest[(r.get("kind"), r.get("summary"))] = r
        problems = [{"ts": r["ts"], "kind": r["kind"], "summary": r["summary"][:200],
                     "evidence": (r.get("evidence") or "")[-600:]}
                    for r in latest.values() if r.get("status") == "needs_attention"]
    else:
        problems = []
    return {"last_run": st.get("last_run"), "last_actions": (st.get("last_actions") or [])[-5:],
            "open_24h": open_inc, "recent": recent[::-1], "problems": sorted(problems, key=lambda r: r["ts"], reverse=True)}


STEP_LINE = re.compile(r"\[prior-mapper\]|\[mapper\]|\[reconstructing\]|Global bundle adjustment|Retriangulation|"
                       r"Registering image|Finding good initial|Extracting colors|Bundle adjustment report|"
                       r"Acquired lease|Reconstruction (succeeded|failed)")


def _health(rate, idle_s, cpu_cores, log_age_s, gpu_util=None) -> dict:
    """progressing: the counter is moving. working: no counter movement, but the worker is busy (CPU or GPU) or
    still logging (normal during long optimizations that report no progress). stalled: no movement, idle CPU and
    GPU and silent logs for 20 min (the watchdog's rule)."""
    if rate is not None and rate > 0:
        return {"state": "progressing", "idle_s": idle_s}
    # GPU bundle adjustment leaves the CPU nearly idle and logs nothing for minutes: count the GPU as busy too.
    busy = (cpu_cores is not None and cpu_cores >= 0.25) or (gpu_util is not None and gpu_util >= 20)
    logging = log_age_s is not None and log_age_s < 20 * 60
    if busy or logging or idle_s is None or idle_s < 20 * 60:
        return {"state": "working", "idle_s": idle_s}
    return {"state": "stalled", "idle_s": idle_s}


def _docker_ts(line: str) -> float:
    try:
        stamp = line.split(" ", 1)[0]
        base, frac = stamp.rstrip("Z").split(".")
        return calendar.timegm(time.strptime(base, "%Y-%m-%dT%H:%M:%S")) + float("0." + frac[:6])
    except Exception:
        return time.time()


def _step_text(msg: str) -> str:
    msg = re.sub(r"^[IWE]\d{8} [\d:.]+ \d+ [\w.]+:\d+\] ", "", msg)  # glog prefix
    return msg.strip()[:140]
