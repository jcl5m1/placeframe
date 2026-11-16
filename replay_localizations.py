#!/usr/bin/env python3
"""Replay recorded phone /localize requests against the Placeframe server.

The API records every client localization (image + camera intrinsics + map ids + response + latency) to
/mnt/hdd12t/placeframe_scratch/localization_recordings/<date>/ (LOCALIZATION_RECORD_DIR in compose.local.yml).

  summary   metrics of what the clients actually experienced (success rate, latency, confidence) per session
  replay    re-send each request once and compare to the recording -> regression report (exit 1 on regression)
  load      N simulated clients replaying sessions concurrently at their original pace (or --speed x) ->
            throughput, latency percentiles, error rate and server GPU/CPU per request

Sessions = consecutive requests from the same client/user-agent with gaps < --session-gap seconds.
Replayed requests carry X-Placeframe-Replay so they are not recorded again.

  python replay_localizations.py summary
  python replay_localizations.py replay --date 2026-11-15 --out report.json
  python replay_localizations.py replay --map-id <new map>      # same place, rebuilt map: success/confidence only
  python replay_localizations.py load --clients 8 --speed 1 --duration 120
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_recording_to_map import PlaceframeApi  # noqa: E402

ROOT = Path("/mnt/hdd12t/placeframe_scratch/localization_recordings")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["summary", "replay", "load"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--date", action="append", help="YYYY-MM-DD (repeatable; default: all)")
    ap.add_argument("--client", help="only requests whose client IP or user agent contains this")
    ap.add_argument("--session-gap", type=float, default=120.0)
    ap.add_argument("--map-id", help="replay against this map instead of the recorded map ids")
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    # regression thresholds
    ap.add_argument("--max-success-drop", type=float, default=0.05, help="allowed drop in success rate")
    ap.add_argument("--max-pose-dt", type=float, default=0.25, help="m; median translation change vs recording")
    ap.add_argument("--max-pose-dr", type=float, default=2.0, help="deg; median rotation change vs recording")
    ap.add_argument("--max-latency-x", type=float, default=1.5, help="allowed p50 latency growth factor")
    # load
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--speed", type=float, default=1.0, help="pace multiplier (2 = twice as many requests/s)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds")
    a = ap.parse_args()

    reqs = load_requests(a.root, a.date, a.client)
    if not reqs:
        raise SystemExit(f"no recorded requests under {a.root}")
    sessions = split_sessions(reqs, a.session_gap)
    if a.mode == "summary":
        report = summary(sessions)
    elif a.mode == "replay":
        report = replay(sessions, a)
    else:
        report = load_test(sessions, a)
    text = json.dumps(report, indent=2, default=float)
    if a.out:
        a.out.write_text(text)
    print(text)
    if a.mode == "replay" and report["regressions"]:
        sys.exit(1)


def load_requests(root: Path, dates: list[str] | None, client: str | None) -> list[dict]:
    out = []
    for day in sorted(root.glob("*/requests.jsonl")):
        if dates and day.parent.name not in dates:
            continue
        for line in day.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if client and client not in f"{r.get('client')} {r.get('user_agent')}":
                continue
            r["_path"] = day.parent / r["image"]
            r["_t"] = datetime.fromisoformat(r["received_at"]).timestamp()
            out.append(r)
    return sorted(out, key=lambda r: r["_t"])


def split_sessions(reqs: list[dict], gap: float) -> list[list[dict]]:
    sessions: dict[str, list[list[dict]]] = {}
    for r in reqs:
        key = f"{r.get('client')}|{r.get('user_agent')}"
        groups = sessions.setdefault(key, [])
        if not groups or r["_t"] - groups[-1][-1]["_t"] > gap:
            groups.append([])
        groups[-1].append(r)
    return sorted((s for g in sessions.values() for s in g), key=lambda s: s[0]["_t"])


def summary(sessions: list[list[dict]]) -> dict:
    return {"sessions": [describe(s) for s in sessions], "overall": stats([r for s in sessions for r in s])}


def describe(s: list[dict]) -> dict:
    return {"start": s[0]["received_at"], "client": s[0].get("client"), "user_agent": s[0].get("user_agent"),
            "duration_s": round(s[-1]["_t"] - s[0]["_t"], 1),
            "requests_per_s": round(len(s) / max(1.0, s[-1]["_t"] - s[0]["_t"]), 2),
            "maps": sorted({m for r in s for m in r["request"]["map_ids"]})} | stats(s)


def stats(rs: list[dict], key: str = "response", lat: str = "latency_ms", status: str = "status") -> dict:
    ok = [r for r in rs if r.get(key)]
    lats = np.array([r[lat] for r in rs if r.get(lat) is not None], float)
    conf = np.array([confidence(r[key]) for r in ok], float)
    return {"requests": len(rs), "localized": len(ok), "success_rate": round(len(ok) / max(1, len(rs)), 3),
            "no_fix_422": sum(1 for r in rs if r.get(status) == 422),
            "errors": sum(1 for r in rs if (r.get(status) or 200) >= 400 and r.get(status) != 422),
            "latency_ms_p50": pct(lats, 50), "latency_ms_p95": pct(lats, 95), "latency_ms_max": pct(lats, 100),
            "confidence_p50": pct(conf, 50), "image_kB_avg": round(np.mean([r["image_bytes"] for r in rs]) / 1e3, 1)}


def replay(sessions: list[list[dict]], a) -> dict:
    api = PlaceframeApi()
    api._auth()
    rs = [r for s in sessions for r in s]
    deltas = []
    for i, r in enumerate(rs):
        res, ms, status = send(api, r, a.map_id)
        r["replay"], r["replay_ms"], r["replay_status"] = res, ms, status
        if not a.map_id and res and r.get("response"):
            deltas.append(pose_delta(r["response"][0], res[0]))
        print(f"\r{i + 1}/{len(rs)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    before = stats(rs)
    after = stats(rs, "replay", "replay_ms", "replay_status")
    d = np.array(deltas) if deltas else np.zeros((0, 2))
    report = {"recorded": before, "replayed": after,
              "pose_change": {"n": len(d), "dt_m_p50": pct(d[:, 0], 50), "dt_m_p95": pct(d[:, 0], 95),
                              "dr_deg_p50": pct(d[:, 1], 50), "dr_deg_p95": pct(d[:, 1], 95)} if len(d) else None,
              "changed": [{"received_at": r["received_at"], "recorded": bool(r.get("response")), "replayed": bool(r["replay"]),
                           "image": str(r["_path"])} for r in rs if bool(r.get("response")) != bool(r["replay"])][:50]}
    reg = []
    if before["success_rate"] - after["success_rate"] > a.max_success_drop:
        reg.append(f"success rate {before['success_rate']:.3f} -> {after['success_rate']:.3f}")
    if report["pose_change"] and report["pose_change"]["dt_m_p50"] > a.max_pose_dt:
        reg.append(f"median translation change {report['pose_change']['dt_m_p50']:.3f} m")
    if report["pose_change"] and report["pose_change"]["dr_deg_p50"] > a.max_pose_dr:
        reg.append(f"median rotation change {report['pose_change']['dr_deg_p50']:.2f} deg")
    if before["latency_ms_p50"] and after["latency_ms_p50"] and after["latency_ms_p50"] > a.max_latency_x * before["latency_ms_p50"]:
        reg.append(f"p50 latency {before['latency_ms_p50']:.0f} -> {after['latency_ms_p50']:.0f} ms")
    report["regressions"] = reg
    return report


def load_test(sessions: list[list[dict]], a) -> dict:
    results, lock, stop = [], threading.Lock(), time.time() + a.duration

    def client(k: int) -> None:
        s = sessions[k % len(sessions)]
        cli = PlaceframeApi()  # one login per simulated client, like real phones
        while time.time() < stop:
            t0 = time.time()
            for r in s:
                due = t0 + (r["_t"] - s[0]["_t"]) / a.speed
                if due > stop:
                    return
                time.sleep(max(0.0, due - time.time()))
                res, ms, status = send(cli, r, a.map_id)
                with lock:
                    results.append({"t": time.time(), "ms": ms, "ok": bool(res), "status": status})

    sampler = Sampler()
    sampler.start()
    threads = [threading.Thread(target=client, args=(k,), daemon=True) for k in range(a.clients)]
    t_start = time.time()
    for k, th in enumerate(threads):
        th.start()
        time.sleep(min(1.0, 5.0 / max(1, a.clients)))  # stagger session starts
    for th in threads:
        th.join()
    wall = time.time() - t_start
    sampler.stop()
    ms = np.array([r["ms"] for r in results], float)
    n = len(results)
    util = sampler.summary()
    return {"clients": a.clients, "speed": a.speed, "duration_s": round(wall, 1), "requests": n,
            "throughput_rps": round(n / wall, 2), "offered_rps_per_client": offered_rate(sessions, a.speed),
            "success_rate": round(sum(r["ok"] for r in results) / max(1, n), 3),
            "errors": sum(1 for r in results if r["status"] >= 400),
            "latency_ms_p50": pct(ms, 50), "latency_ms_p95": pct(ms, 95), "latency_ms_p99": pct(ms, 99),
            "server": util,
            "gpu_seconds_per_request": round(util["gpu_util_avg"] * wall / max(1, n), 3) if util else None,
            "localizer_cpu_core_seconds_per_request": round(util["localizer_cpu_cores_avg"] * wall / max(1, n), 3) if util else None}


def send(api: PlaceframeApi, r: dict, map_id: str | None):
    q = r["request"]
    data = {"map_ids": json.dumps([map_id] if map_id else q["map_ids"]), "camera_config": json.dumps(q["camera_config"]),
            "axis_convention": q["axis_convention"]}
    for k in ("retrieval_top_k", "ransac_threshold"):
        if q.get(k) is not None:
            data[k] = str(q[k])
    api._auth()
    t0 = time.monotonic()
    try:
        resp = api.client.post(api.server + "/localize", data=data, headers={"X-Placeframe-Replay": "1"},
                               files={"image": ("q.jpg", r["_path"].read_bytes(), r.get("image_content_type") or "image/jpeg")})
    except Exception:
        return None, (time.monotonic() - t0) * 1000, 599
    ms = (time.monotonic() - t0) * 1000
    if resp.status_code == 401:
        api._token_at = 0.0  # force a fresh login on the next request
    return (resp.json() or None) if resp.status_code < 400 else None, ms, resp.status_code


def pose_delta(a: dict, b: dict) -> tuple[float, float]:
    def cam(m):
        tr = m["camera_from_map_transform"]
        R = Rotation.from_quat([tr["rotation"][k] for k in "xyzw"])
        t = np.array([tr["translation"][k] for k in "xyz"])
        return R, -(R.inv().apply(t))  # rotation, camera centre in map

    Ra, Ca = cam(a)
    Rb, Cb = cam(b)
    return float(np.linalg.norm(Ca - Cb)), float(np.degrees((Ra.inv() * Rb).magnitude()))


def confidence(resp: list[dict]) -> float:
    return float(max((m.get("metrics") or {}).get("confidence_loose") or 0.0 for m in resp))


def offered_rate(sessions, speed) -> float:
    rates = [len(s) / max(1.0, s[-1]["_t"] - s[0]["_t"]) for s in sessions if len(s) > 1]
    return round(float(np.mean(rates)) * speed, 2) if rates else None


def pct(x, q):
    x = np.asarray(x, float)
    return round(float(np.percentile(x, q)), 3) if len(x) else None


class Sampler(threading.Thread):
    """GPU utilisation + localizer container CPU every 2 s while a load test runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.gpu, self.cpu, self.mem, self._stop = [], [], [], threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout.split(",")
                self.gpu.append(float(g[0]))
                self.mem.append(float(g[1]))
                c = subprocess.run(["sg", "docker", "-c", "docker stats --no-stream --format '{{.CPUPerc}}' placeframe-localizer-cuda-1"],
                                   capture_output=True, text=True, timeout=20).stdout.strip().rstrip("%")
                self.cpu.append(float(c))
            except Exception:
                pass
            self._stop.wait(2.0)

    def stop(self):
        self._stop.set()
        self.join(timeout=30)

    def summary(self):
        if not self.gpu:
            return None
        return {"gpu_util_avg": round(np.mean(self.gpu) / 100, 3), "gpu_util_max": round(max(self.gpu) / 100, 3),
                "gpu_mem_mb_max": max(self.mem), "localizer_cpu_cores_avg": round(np.mean(self.cpu) / 100, 2) if self.cpu else None}


if __name__ == "__main__":
    main()
