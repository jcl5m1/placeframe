"""Localizer serving throughput: N concurrent tracked sessions replaying a recorded walk through the API
(POST /localize/track), each sending its next frame as soon as the previous answer arrives. Reports queries/s,
latency percentiles and GPU utilization/memory.

  python bench/serve_throughput.py REPLAY_DIR MAP_ID --sessions 1 2 4 --frames 60
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from add_recording_to_map import PlaceframeApi  # noqa: E402


def session(api: PlaceframeApi, url: str, frames: list[dict], d: Path, map_id: str, lat: list[float]) -> None:
    sid = str(uuid.uuid4())
    for r in frames:
        q = r["request"]
        t = time.monotonic()
        resp = api.client.post(url, headers={"X-Placeframe-Replay": "1"},
                               data={"map_ids": json.dumps([map_id]), "camera_config": json.dumps(q["camera_config"]),
                                     "session_id": sid, "odometry_from_camera": json.dumps(q["odometry_from_camera"])},
                               files={"image": ("q.jpg", (d / "img" / r["image"]).read_bytes(), "image/jpeg")}, timeout=120)
        if resp.status_code < 300:
            lat.append((time.monotonic() - t) * 1000)


def gpu_sampler(stop: threading.Event, out: list[tuple[float, float]]) -> None:
    while not stop.is_set():
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.split(",")
        out.append((float(o[0]), float(o[1])))
        time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path)
    ap.add_argument("map_id")
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()
    frames = json.loads((a.replay / "session.json").read_text())
    api = PlaceframeApi()
    api._auth()
    url = api.server.rstrip("/") + "/localize/track"
    session(api, url, frames[:3], a.replay, a.map_id, [])  # warm up: load the map
    results = []
    for n in a.sessions:
        lat: list[float] = []
        gpu: list[tuple[float, float]] = []
        stop = threading.Event()
        sampler = threading.Thread(target=gpu_sampler, args=(stop, gpu))
        sampler.start()
        t0 = time.monotonic()
        # Each session starts at a different point of the walk, as different people would.
        threads = [threading.Thread(target=session, args=(api, url, frames[(i * 37) % len(frames):][: a.frames]
                                                         or frames[: a.frames], a.replay, a.map_id, lat))
                   for i in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.monotonic() - t0
        stop.set()
        sampler.join()
        lat.sort()
        res = {"sessions": n, "queries": len(lat), "qps": len(lat) / wall, "p50_ms": statistics.median(lat),
               "p95_ms": lat[int(0.95 * (len(lat) - 1))], "gpu_util_mean": statistics.mean(g for g, _ in gpu),
               "gpu_mem_max_MiB": max(m for _, m in gpu)}
        results.append(res)
        print(json.dumps(res), flush=True)
    (a.replay / "throughput.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
