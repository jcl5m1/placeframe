#!/usr/bin/env python3
"""Where a tracked localization request's time goes, and what limits throughput, without the API or HTTP.

Runs the localizer's own tracked path in-process (tracked.localize_candidates + the per-session tracker, as
POST /localization/track does) on a recorded phone walk against one map:

  profile     one session; timers around each stage (GPU stages synchronised so their time is theirs)
  throughput  P processes x 2 worker threads (the server's LOCALIZER_TRACK_WORKERS), S concurrent sessions, each
              replaying its own stretch of the walk frame after frame; queries/s, latency, GPU and CPU use

Comparing in-process throughput with the throughput measured through the API on the same GPU model separates the
localizer's limit from the API's.

  python bench/localizer_profile.py BUNDLE OUT.json [--configs 1x1 1x2 1x4 2x4 2x8 2x16] [--frames 50]

BUNDLE: code/pf_localizer (docker/localizer/src), map/<id>/ (features.h5, global_descriptors.h5, opq, pq,
sfm_model/), map/metrics.json, calibration/global.json, replay/session.json + replay/img/.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHA = os.environ.get("LOCALIZER_SHA", "tree-37154ed4b6d1b4193e4b1f69bdb225010b373f50")
TRACK_WORKERS = 2


# ------------------------------------------------------------------ localizer setup (in each process)
def set_threads() -> None:
    """LP_THREADS caps PyTorch's and faiss's CPU threads (0 = their defaults: one per core)."""
    n = int(os.environ.get("LP_THREADS", "0"))
    if n:
        import faiss
        import torch

        torch.set_num_threads(n)
        faiss.omp_set_num_threads(n)


def setup(bundle: Path):
    sys.path.insert(0, str(bundle / "code"))
    set_threads()
    from uuid import UUID

    from core.calibration import load_global_calibration
    from core.reconstruction_metrics import ReconstructionMetrics
    from pf_localizer import localize
    from pf_localizer import map as mapmod

    localize.load_models()
    calibration = load_global_calibration(bundle / "calibration/global.json", SHA)
    map_dir = bundle / "map"
    map_id = next(p.name for p in map_dir.iterdir() if p.is_dir())

    class NoDownloads:  # files are already in map_dir: load_map's download loop sees no objects
        def get_paginator(self, _name):
            return type("P", (), {"paginate": lambda self, **_kw: []})()

    metrics = ReconstructionMetrics.model_validate(json.loads((map_dir / "metrics.json").read_text()))
    m = mapmod.load_map(UUID(map_id), NoDownloads(), "unused", map_dir, metrics)
    return m, calibration


def load_frames(bundle: Path) -> list[dict]:
    from core.camera_config import PinholeCameraConfig

    frames = []
    for r in json.loads((bundle / "replay/session.json").read_text()):
        q = r["request"]
        t, o = q["odometry_from_camera"]["translation"], q["odometry_from_camera"]["rotation"]
        frames.append({"image": (bundle / "replay/img" / r["image"]).read_bytes(),
                       "camera": PinholeCameraConfig.model_validate(q["camera_config"]),
                       "t": [t["x"], t["y"], t["z"]], "q": [o["x"], o["y"], o["z"], o["w"]]})
    return frames


def one_request(m, calibration, tracker, f) -> None:
    from pf_localizer.tracked import localize_candidates
    from pf_localizer.tracker import pose_from_quaternion

    odom = pose_from_quaternion(f["t"], f["q"])
    cands = localize_candidates(m, f["camera"], f["image"], tracker.predicted_map_from_camera(odom),
                                tracker.confident, SHA, calibration)
    tracker.update(odom, cands)


# ------------------------------------------------------------------ profile
def profile(bundle: Path, frames_n: int) -> dict:
    sys.path.insert(0, str(bundle / "code"))
    import torch
    from pf_localizer import localize, tracked
    from pf_localizer.tracker import Sessions

    m, calibration = setup(bundle)
    frames = load_frames(bundle)
    stages: dict[str, float] = {}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def timed(name, fn, gpu=False):
        def wrap(*a, **k):
            if gpu:
                sync()
            t = time.perf_counter()
            out = fn(*a, **k)
            if gpu:
                sync()
            stages[name] = stages.get(name, 0.0) + time.perf_counter() - t
            return out
        return wrap

    localize.local_feature_extractor = timed("local features (ALIKED, GPU)", localize.local_feature_extractor, True)
    localize.global_descriptor_extractor = timed("global descriptors (GPU)", localize.global_descriptor_extractor, True)
    localize.local_feature_matcher = timed("matching (LightGlue, GPU)", localize.local_feature_matcher, True)
    tracked.canonicalize_image = timed("decode + resize (CPU)", tracked.canonicalize_image)
    tracked.estimate_and_refine_absolute_pose = timed("pose RANSAC + refine (CPU)",
                                                      tracked.estimate_and_refine_absolute_pose)
    tracked.build_localization_metrics = timed("localization metrics (CPU)", tracked.build_localization_metrics)
    tracked.decode_descriptors = timed("descriptor decode, PQ (CPU)", tracked.decode_descriptors)
    tracked.release_gpu_memory = timed("release cached GPU memory", tracked.release_gpu_memory, True)
    tracked.tile_image = timed("query tiles (CPU)", tracked.tile_image)
    from pf_localizer import tracker as trk
    trk.Tracker.update = timed("tracker update (CPU)", trk.Tracker.update)
    sessions = Sessions()
    tracker = sessions.get("profile", "map")
    for f in frames[:5]:  # warm up (CUDA kernels, map descriptors on the GPU)
        one_request(m, calibration, tracker, f)
    stages.clear()
    tracker = sessions.get("profile-2", "map")
    totals = []
    for f in frames[:frames_n]:
        sync()
        t = time.perf_counter()
        one_request(m, calibration, tracker, f)
        sync()
        totals.append(time.perf_counter() - t)
    n = len(totals)
    per = {k: round(v / n * 1000, 1) for k, v in sorted(stages.items(), key=lambda kv: -kv[1])}
    per["other (retrieval, transfers, 2D-3D correspondence loop)"] = round(statistics.mean(totals) * 1000 - sum(per.values()), 1)
    return {"frames": n, "mean_ms": round(statistics.mean(totals) * 1000, 1),
            "p50_ms": round(statistics.median(totals) * 1000, 1), "stages_ms": per}


# ------------------------------------------------------------------ throughput
def _proc(bundle: str, sessions: int, frames_n: int, offset0: int, ready, go, out) -> None:
    sys.path.insert(0, str(Path(bundle) / "code"))
    from pf_localizer.tracker import Sessions

    m, calibration = setup(Path(bundle))
    frames = load_frames(Path(bundle))
    pool = ThreadPoolExecutor(max_workers=TRACK_WORKERS)
    trackers = Sessions()
    warm = trackers.get("warm", "map")
    for f in frames[:3]:
        pool.submit(one_request, m, calibration, warm, f).result()
    ready.set()
    go.wait()
    lat: list[float] = []
    lock = threading.Lock()

    def session(k: int) -> None:
        tr = trackers.get(f"s{offset0 + k}", "map")
        start = ((offset0 + k) * 37) % max(1, len(frames) - frames_n)
        for f in frames[start:start + frames_n]:
            t = time.perf_counter()
            pool.submit(one_request, m, calibration, tr, f).result()
            with lock:
                lat.append(time.perf_counter() - t)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=session, args=(k,)) for k in range(sessions)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    out.put({"lat": lat, "wall": time.perf_counter() - t0})


class Sampler:
    """GPU utilization/memory (nvidia-smi) and host CPU busy share (/proc/stat), every 0.5 s."""

    def __init__(self) -> None:
        self.gpu, self.mem, self.cpu, self.stop = [], [], [], threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _cpu_times():
        v = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        return sum(v), v[3] + v[4]

    def _run(self) -> None:
        last = self._cpu_times()
        while not self.stop.is_set():
            out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits"]).split(",")
            if len(out) == 2:
                self.gpu.append(float(out[0]))
                self.mem.append(float(out[1]))
            now = self._cpu_times()
            if now[0] > last[0]:
                self.cpu.append(100 * (1 - (now[1] - last[1]) / (now[0] - last[0])))
            last = now
            time.sleep(0.5)


def throughput(bundle: Path, procs: int, sessions: int, frames_n: int) -> dict:
    ctx = mp.get_context("spawn")
    ready = [ctx.Event() for _ in range(procs)]
    go, out = ctx.Event(), ctx.Queue()
    split = [sessions // procs + (1 if i < sessions % procs else 0) for i in range(procs)]
    offs = [sum(split[:i]) for i in range(procs)]
    ps = [ctx.Process(target=_proc, args=(str(bundle), split[i], frames_n, offs[i], ready[i], go, out))
          for i in range(procs) if split[i]]
    for p in ps:
        p.start()
    for e in ready[:len(ps)]:
        e.wait()
    s = Sampler()
    t0 = time.perf_counter()
    go.set()
    res = [out.get() for _ in ps]
    wall = time.perf_counter() - t0
    s.stop.set()
    for p in ps:
        p.join()
    lat = sorted(x for r in res for x in r["lat"])
    pct = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))] * 1000, 1)  # noqa: E731
    return {"processes": len(ps), "sessions": sessions, "queries": len(lat), "qps": round(len(lat) / wall, 2),
            "p50_ms": pct(0.5), "p95_ms": pct(0.95),
            "gpu_util_mean": round(statistics.mean(s.gpu), 1) if s.gpu else None,
            "gpu_mem_max_MiB": max(s.mem) if s.mem else None,
            "cpu_busy_mean_pct": round(statistics.mean(s.cpu), 1) if s.cpu else None}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def host() -> dict:
    lscpu = _run(["lscpu"])
    model = next((ln.split(":", 1)[1].strip() for ln in lscpu.splitlines() if ln.startswith("Model name")), "")
    gpu = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip()
    return {"cpu": model, "cpus_available": len(os.sched_getaffinity(0)), "gpu": gpu}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--configs", nargs="+", default=["1x1", "1x2", "1x4", "1x8", "2x4", "2x8", "2x16"],
                    help="PROCESSESxSESSIONS")
    ap.add_argument("--frames", type=int, default=50, help="frames per session (and for the profile)")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--threads", nargs="+", type=int, default=[0],
                    help="CPU thread caps to compare (0 = library defaults); each runs the profile and configs")
    a = ap.parse_args()
    result: dict = {"host": host(), "frames_per_session": a.frames, "track_workers_per_process": TRACK_WORKERS,
                    "runs": []}
    for n in a.threads:
        os.environ["LP_THREADS"] = str(n)  # spawned processes inherit it
        run: dict = {"threads": n or "default", "throughput": []}
        result["runs"].append(run)
        if not a.skip_profile:
            run["profile"] = profile(a.bundle, a.frames) if n == a.threads[0] else _profile_in_child(a.bundle, a.frames)
            print(json.dumps(run["profile"]), flush=True)
            a.out.write_text(json.dumps(result, indent=2))
        for c in a.configs:
            p, s = (int(x) for x in c.split("x"))
            r = throughput(a.bundle, p, s, a.frames)
            print(json.dumps(r), flush=True)
            run["throughput"].append(r)
            a.out.write_text(json.dumps(result, indent=2))


def _profile_child(bundle: str, frames_n: int, out) -> None:
    out.put(profile(Path(bundle), frames_n))


def _profile_in_child(bundle: Path, frames_n: int) -> dict:
    """Later thread settings profile in a fresh process: the first profile patched this process's modules."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_profile_child, args=(str(bundle), frames_n, q))
    p.start()
    r = q.get()
    p.join()
    return r


if __name__ == "__main__":
    main()
