#!/usr/bin/env python3
"""Export recorded phone localization sessions for the map viewer (~/pf_x5/viewer).

Writes, under the viewer folder:
  maps.json                      the maps the viewer can show: [{id, name, file}]
  maps/<map id>/map.json         a viewer map for every map a session localized against (built on demand)
  sessions/index.json            one entry per client session
  sessions/<sid>.json            per request: time, result, camera position + heading in the viewer frame,
                                 confidence, latency, thumbnail; plus "replays" against other maps
  sessions/<sid>/<n>.jpg         upright 360 px thumbnails of what the phone saw

Positions: the viewer frame is Placeframe's OpenCV world with y negated, which is exactly Placeframe's
UNITY world (basis diag(1, -1, 1)), so UNITY responses are used as-is and OPENCV ones get y negated.

--relocalize re-sends each finished session to every other map that has a viewer map (cached per session and
map; replayed requests are not re-recorded), so a session can be watched on maps it was not recorded against.

  python export_viewer_sessions.py --relocalize          # run by cron every 5 min
  python export_viewer_sessions.py --map <map id>        # add a map to the viewer
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_localizations as rl  # noqa: E402
from add_recording_to_map import PlaceframeApi  # noqa: E402

VIEWER = Path.home() / "pf_x5/viewer"
DEFAULT_MAP = {"id": "20afa6e7-0150-45ae-b041-f23971b46d30", "file": "map.json"}  # first build (no hallway)
# Listed first in the viewer's map picker, best first: the full X5 walk (panoramic pairing, VIO priors), then the
# merged two-recording map, then the OpenVIO-prior build.
FEATURED = ["4c17aa5e-aaf8-453a-bfd7-99d2bc176fda", "f3913957-1a89-4a29-928d-b05ff3cdc4ef", "19917a7a-cb22-46d9-963c-20732257d6f5",
            "8bc254e3-854b-42df-85cc-dd064689ed78"]
SETTLE_S = 180  # a session is finished once its last request is this old
THUMB_W = 360
# Image orientation reported by the client -> PIL transpose that makes the thumbnail upright.
UPRIGHT = {"LEFT_TOP": Image.Transpose.ROTATE_270, "RIGHT_TOP": Image.Transpose.ROTATE_90,
           "BOTTOM_RIGHT": Image.Transpose.ROTATE_180}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", type=Path, default=VIEWER)
    ap.add_argument("--relocalize", action="store_true")
    ap.add_argument("--map", action="append", default=[], help="also build a viewer map for this localization map id")
    ap.add_argument("--session-gap", type=float, default=120.0)
    a = ap.parse_args()

    reqs = rl.load_requests(rl.ROOT, None, None)
    sessions = [s for s in rl.split_sessions(reqs, a.session_gap) if not is_tool_traffic(s)]
    api = PlaceframeApi()
    maps = ensure_maps(api, a.viewer, {m for s in sessions for r in s for m in r["request"]["map_ids"]} | set(a.map))
    out = a.viewer / "sessions"
    out.mkdir(exist_ok=True)
    index = []
    for s in sessions:
        sid = session_id(s)
        doc = export_session(sid, s, out)
        finished = time.time() - s[-1]["_t"] > SETTLE_S
        if a.relocalize and finished:
            for m in maps:
                if m["id"] not in doc["map_ids"]:
                    doc["replays"][m["id"]] = relocalize(api, sid, s, m["id"], out)
        (out / f"{sid}.json").write_text(json.dumps(doc, separators=(",", ":")))
        index.append({k: doc[k] for k in ("id", "start", "duration_s", "user_agent", "map_ids", "requests", "localized")}
                     | {"replayed_on": sorted(doc["replays"]), "finished": finished})
    (out / "index.json").write_text(json.dumps(sorted(index, key=lambda x: x["start"], reverse=True), indent=1))
    print(f"{len(index)} sessions, {len(maps)} maps exported to {a.viewer}")


def is_tool_traffic(s: list[dict]) -> bool:
    return all((r.get("user_agent") or "").startswith("python-httpx") for r in s)


def session_id(s: list[dict]) -> str:
    key = f"{s[0]['received_at']}|{s[0].get('client')}|{s[0].get('user_agent')}"
    return s[0]["received_at"][:19].replace(":", "").replace("-", "") + "-" + hashlib.sha1(key.encode()).hexdigest()[:6]


def ensure_maps(api: PlaceframeApi, viewer: Path, needed: set[str]) -> list[dict]:
    known = {m["id"]: m for m in api.get("/localization-maps")}
    wanted = [m for m in FEATURED if m in known] + sorted(needed - set(FEATURED) - {DEFAULT_MAP["id"]})
    if (viewer / "maps").exists():
        wanted += sorted(d.parent.name for d in (viewer / "maps").glob("*/map.json")
                         if d.parent.name in known and d.parent.name not in wanted)
    entries = []
    for mid in wanted:
        if mid not in known:
            continue
        dest = viewer / "maps" / mid / "map.json"
        if not dest.exists() or not (dest.parent / ".v2").exists():
            build_viewer_map(known[mid], dest, viewer / DEFAULT_MAP["file"])
            (dest.parent / ".v2").touch()
        entries.append({"id": mid, "name": known[mid].get("name") or mid[:8], "file": f"maps/{mid}/map.json"})
    entries.append({"id": DEFAULT_MAP["id"], "file": DEFAULT_MAP["file"],
                    "name": known.get(DEFAULT_MAP["id"], {}).get("name", "original map") + " (first build — no hallway)"})
    (viewer / "maps.json").write_text(json.dumps(entries, indent=1))
    return entries


def build_viewer_map(m: dict, dest: Path, reference: Path) -> None:
    """Viewer map from the stored COLMAP model: points with their real observation counts, per-frame video
    timestamps (rig + ms from the image names), mapping gaps, and the odometry overlays of the reference viewer
    map carried over by a similarity fit on frames matched by timestamp."""
    rid = m["reconstruction_id"]
    frames = model_frames(minio_text(rid, "images.txt"))
    pts = model_points(minio_text(rid, "points3D.txt"))
    rigs = sorted({r for r, _ in frames})
    cams, gaps, offset = [], [], 0
    for rig in rigs:
        ts = sorted(t for r, t in frames if r == rig)
        for t in ts:
            c = frames[(rig, t)]
            cams.append([round(c[0], 3), round(c[1], 3), round(c[2], 3), t + offset])
        gaps += [[(a + offset) / 1000, (b + offset) / 1000] for a, b in zip(ts, ts[1:]) if b - a > 1000]
        offset += ts[-1] + 5000  # later recordings follow on the time axis, after a visible break
    data = {"name": m.get("name") or m["id"], "map_id": m["id"], "points": pts, "cams": cams, "camsOrig": cams,
            "gaps": gaps, "rigs": len(rigs)}
    data |= carry_overlays(reference, {t: c for (r, t), c in frames.items() if r == rigs[0]})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, separators=(",", ":")))
    print(f"viewer map for {data['name']}: {len(pts)} points, {len(cams)} frames, {len(gaps)} gaps, "
          f"overlays: {data.get('overlayNote', 'none')}")


def minio_text(rid: str, name: str) -> str:
    import subprocess
    return subprocess.run(["sg", "docker", "-c", f"docker exec placeframe-minio-1 mc cat l/dev-reconstructions/{rid}/sfm_model/{name}"],
                          capture_output=True, text=True, check=True).stdout


def model_frames(images_txt: str) -> dict[tuple[str, int], list[float]]:
    """(rig, video ms) -> camera centre in viewer axes, averaged over the rig's views."""
    acc: dict[tuple[str, int], list] = {}
    lines = [ln for ln in images_txt.splitlines() if ln and not ln.startswith("#")]
    for ln in lines[0::2]:
        f = ln.split()
        q = [float(x) for x in f[1:5]]
        t = np.array([float(x) for x in f[5:8]])
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        c = -R.T @ t
        rig, _, stem = f[9].split("/")
        acc.setdefault((rig, int(stem.split(".")[0])), []).append(c)
    return {k: list(np.mean(v, axis=0) * [1, -1, 1]) for k, v in acc.items()}


def model_points(points_txt: str, limit: int = 300_000) -> list[list]:
    rows = []
    for ln in points_txt.splitlines():
        if not ln or ln.startswith("#"):
            continue
        f = ln.split()
        rows.append((float(f[1]), -float(f[2]), float(f[3]), int(f[4]), int(f[5]), int(f[6]), (len(f) - 8) // 2))
    if len(rows) > limit:  # keep the page light: drop the least-observed points first
        rows.sort(key=lambda r: -r[6])
        rows = rows[:limit]
    return [[round(r[0], 3), round(r[1], 3), round(r[2], 3), r[3], r[4], r[5], r[6]] for r in rows]


def carry_overlays(reference: Path, new_cams: dict[int, list[float]]) -> dict:
    if not reference.exists():
        return {}
    ref = json.loads(reference.read_text())
    if not any(ref.get(k) for k in ("vio", "ovio", "ovins")):
        return {}
    times = np.array(sorted(new_cams))
    src, dst = [], []
    for c in ref["cams"]:
        k = int(np.argmin(np.abs(times - c[3])))
        if abs(times[k] - c[3]) <= 300:
            src.append(c[:3])
            dst.append(new_cams[int(times[k])])
    if len(src) < 10:
        return {}
    src, dst = np.array(src), np.array(dst)
    keep = np.ones(len(src), bool)
    for _ in range(3):  # similarity fit, then drop frames the old map got wrong
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        keep = res < max(0.3, 3 * np.median(res))
    out = {}
    for k in ("vio", "ovio", "ovins"):
        if ref.get(k):
            A = np.array(ref[k], float)
            P = (s * (R @ A[:, :3].T)).T + t
            out[k] = [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3), *a[3:]] for p, a in zip(P, ref[k])]
    out["vioNote"] = ref.get("vioNote", "")
    out["overlayNote"] = (f"odometry carried over from the first-build frame: {keep.sum()}/{len(src)} matched frames, "
                          f"scale {s:.3f}, RMS {np.sqrt(np.mean(res[keep] ** 2)):.2f} m")
    return out


def umeyama(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean())
    return s, R, mb - s * R @ ma


def fetch_array(api: PlaceframeApi, path: str, widths: list[tuple[int, str]]) -> list[np.ndarray]:
    api._auth()
    r = api.client.get(api.server + path, params={"axis_convention": "OPENCV"})
    r.raise_for_status()
    buf = r.content
    n = int.from_bytes(buf[:4], "little")
    out, off = [], 4
    for width, dtype in widths:
        size = n * width * np.dtype(dtype).itemsize
        out.append(np.frombuffer(buf[off:off + size], dtype).reshape(n, width))
        off += size
    return out


def export_session(sid: str, s: list[dict], out: Path) -> dict:
    prev = out / f"{sid}.json"
    replays = json.loads(prev.read_text()).get("replays", {}) if prev.exists() else {}
    thumbs = out / sid
    thumbs.mkdir(exist_ok=True)
    t0 = s[0]["_t"]
    frames = []
    for n, r in enumerate(s):
        thumb = thumbs / f"{n}.jpg"
        if not thumb.exists():
            write_thumb(r, thumb)
        f = {"t": round(r["_t"] - t0, 2), "status": r["status"], "ms": r["latency_ms"], "img": f"sessions/{sid}/{n}.jpg"}
        if r.get("response"):
            f |= pose(r["response"][0], r["request"]["axis_convention"])
        frames.append(f)
    return {"id": sid, "start": s[0]["received_at"], "duration_s": round(s[-1]["_t"] - t0, 1),
            "client": s[0].get("client"), "user_agent": s[0].get("user_agent"),
            "map_ids": sorted({m for r in s for m in r["request"]["map_ids"]}),
            "camera": s[0]["request"]["camera_config"], "requests": len(s),
            "localized": sum(1 for f in frames if "p" in f), "frames": frames, "replays": replays}


def write_thumb(r: dict, dest: Path) -> None:
    try:
        im = Image.open(io.BytesIO(r["_path"].read_bytes())).convert("RGB")
    except Exception:
        return
    op = UPRIGHT.get(r["request"]["camera_config"].get("orientation"))
    if op is not None:
        im = im.transpose(op)
    im.thumbnail((THUMB_W, THUMB_W))
    im.save(dest, quality=72)


def pose(m: dict, axis_convention: str) -> dict:
    tr = m["camera_from_map_transform"]
    R = Rotation.from_quat([tr["rotation"][k] for k in "xyzw"]).as_matrix()
    t = np.array([tr["translation"][k] for k in "xyz"])
    centre, forward = -R.T @ t, R.T @ np.array([0.0, 0.0, 1.0])
    if axis_convention == "OPENCV":  # to the viewer/UNITY frame
        centre, forward = centre * [1, -1, 1], forward * [1, -1, 1]
    met = m.get("metrics") or {}
    return {"p": [round(float(x), 3) for x in centre], "f": [round(float(x), 3) for x in forward],
            "conf": round(float(met.get("confidence_loose") or 0), 3), "inl": met.get("num_inliers")}


def relocalize(api: PlaceframeApi, sid: str, s: list[dict], map_id: str, out: Path) -> list[dict]:
    cache = out / sid / f"replay-{map_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    res = []
    for r in s:
        resp, ms, status = rl.send(api, r, map_id)
        f = {"status": status, "ms": round(ms, 1)}
        if resp:
            f |= pose(resp[0], r["request"]["axis_convention"])
        res.append(f)
    cache.write_text(json.dumps(res, separators=(",", ":")))
    print(f"session {sid} replayed on {map_id[:8]}: {sum('p' in f for f in res)}/{len(res)} localized")
    return res


if __name__ == "__main__":
    main()
