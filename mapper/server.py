#!/usr/bin/env python3
"""Map updater service: resumable uploads of Insta360 recordings + jobs that merge them into an existing map.

  sg docker -c "uv run uvicorn mapper.server:app --host 127.0.0.1 --port 8150 --root-path /mapper"

Upload protocol (resumable, chunked):
  POST  /api/uploads              {filename, size, last_modified, map_id, new_name, replace, odometry}
                                  -> {id, offset, size}   (same file + same user resumes the existing upload)
  GET   /api/uploads/{id}         -> {offset, size, complete, job_id}
  PATCH /api/uploads/{id}         body = raw chunk; headers Upload-Offset, X-Chunk-SHA256
                                  -> {offset}; 409 + {offset} if the client's offset is stale
When the last byte arrives the upload is sealed and a merge job is queued.

Hold: while $MAPPER_DATA/HOLD exists, queued jobs stay queued (its text is shown as the reason); uploads still work.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import struct
import asyncio
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = Path(os.environ.get("MAPPER_DATA", Path.home() / "pf_mapper"))
UPLOADS, JOBS = DATA / "uploads", DATA / "jobs"
HOLD = DATA / "HOLD"
# Map folders ("experiments/datasets/7-Scenes"): a view-only grouping kept by this service, so Placeframe map names
# (seen by every client) stay unchanged. Map id -> folder path; maps not listed are at the top level.
FOLDERS = DATA / "map_folders.json"
# Pending maps: a new map exists here from the moment its first upload starts ("group:<id>"); its recordings build
# it and merge into it. Group id -> {name, folder, user, created}. Once built, the group's latest Placeframe map
# stands in for it.
PENDING = DATA / "pending_maps.json"
# The map visualizer is a static page (another server, tailnet only): it gets the folder tree as a file in its folder.
VIEWER_DIR = Path(os.environ.get("MAPPER_VIEWER_DIR", Path.home() / "pf_x5" / "viewer"))
CHUNK_MAX = 64 * 1024 * 1024
UPLOAD_ACTIVE_SECONDS = 120  # a partial upload that received data this recently is still in progress
ALLOWED_EXT = {".insv"}

sys.path.insert(0, str(REPO))
from upload_to_placeframe import saved_credentials  # noqa: E402

import gps_placement  # noqa: E402

from .project_pages import router as project_router  # noqa: E402
from . import rentals  # noqa: E402
from .telemetry import Telemetry  # noqa: E402

CREDS = saved_credentials(REPO.parent / "placeframe")
SERVER = CREDS["PLACEFRAME_URL"].rstrip("/")
REALM = "placeframe-dev"
ODOMETRY = ("openvio", "openvins")  # add_recording_to_map.py --odometry; first = default


def stages(odometry: str, new_map: bool = False) -> list[str]:  # mirrors add_recording_to_map.stages()
    merge = [] if new_map else ["register", "merge"]
    return ["calibrate", f"{odometry}-export", f"{odometry}-run", "stitch", "views", "capture", *merge,
            "reconstruct", "publish", "geolocate"]

app = FastAPI(title="Placeframe map updater")
_token_cache: dict[str, tuple[float, str]] = {}
_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()


# --------------------------------------------------------------------------- auth


def user(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "login required")
    hit = _token_cache.get(token)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    r = httpx.get(f"{SERVER}/auth/realms/{REALM}/protocol/openid-connect/userinfo",
                  headers={"Authorization": f"Bearer {token}"}, timeout=15)
    if r.status_code != 200:
        raise HTTPException(401, "session expired — log in again")
    name = r.json().get("preferred_username", "user")
    _token_cache[token] = (time.time(), name)
    return name


@app.post("/api/login")
async def login(body: dict):
    r = httpx.post(f"{SERVER}/auth/realms/{REALM}/protocol/openid-connect/token",
                   data={"grant_type": "password", "client_id": "placeframe-api", "scope": "openid",
                         "username": body.get("username", ""), "password": body.get("password", "")}, timeout=15)
    if r.status_code != 200:
        raise HTTPException(401, "wrong username or password")
    tok = r.json()
    return {"token": tok["access_token"], "expires_in": tok.get("expires_in", 3600)}


@app.get("/api/maps")
async def maps(request: Request, who: str = Depends(user)):
    token = request.headers["authorization"]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{SERVER}/localization-maps", headers={"Authorization": token})
    r.raise_for_status()
    produced = _lineage()
    folders = _folders()
    _base_names.update({m["id"]: m.get("name") or m["id"][:8] for m in r.json()})
    items = []
    for m in r.json():
        chain = _chain(m["id"], produced)
        items.append({"id": m["id"], "name": m.get("name") or m["id"][:8], "active": m.get("active"),
                      "created_at": m.get("created_at"), "base_map_id": chain["base_map_id"],
                      "recordings": len(chain["recordings"]), "folder": folders.get(m["id"], "")})
    group_of = {j["result"]["map_id"]: j.get("group") or _upload_meta(j).get("group") for j in produced.values()}
    for m in items:
        m["group"] = group_of.get(m["id"])
    built_groups = {g for g in group_of.values() if g}
    for g, pm in _pending().items():
        if g not in built_groups:
            n = sum(1 for j in _group_items(g))
            items.append({"id": f"group:{g}", "name": pm["name"], "active": False, "created_at": None, "pending": True,
                          "base_map_id": f"group:{g}", "recordings": n, "folder": pm.get("folder", ""), "group": g})
    by_id = {m["id"]: m for m in items}
    for m in items:
        base = by_id.get(m["base_map_id"])
        m["base_name"] = base["name"] if base else m["base_map_id"][:8]
    return sorted(items, key=lambda m: (not m["active"], m["created_at"] or ""), reverse=False)


@app.post("/api/new-maps")
async def create_pending_map(body: dict, who: str = Depends(user)):
    group = "g" + uuid.uuid4().hex[:11]
    name = str(body.get("name") or "").strip()[:120] or f"New map {time.strftime('%Y-%m-%d')}"
    folder = _clean_folder(str(body.get("folder") or ""))
    with _lock:
        pending = _pending()
        pending[group] = {"name": name, "folder": folder, "user": who, "created": time.time()}
        _write_json(PENDING, pending)
    return {"id": f"group:{group}", "name": name, "folder": folder, "group": group}


def _pending() -> dict[str, dict]:
    try:
        return json.loads(PENDING.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(path)


def _group_items(group: str) -> list[dict]:
    """The recordings of a group: its jobs, plus uploads still in progress (no job yet)."""
    jobs = [j for j in _all_jobs() if (j.get("group") or _upload_meta(j).get("group")) == group]
    have = {j["id"] for j in jobs}
    for mp in UPLOADS.glob("*.json"):
        try:
            m = json.loads(mp.read_text())
        except ValueError:
            continue
        if m.get("group") == group and m["id"] not in have:
            jobs.append({"id": m["id"], "filename": m["filename"], "state": "uploading", "map_id": None,
                         "created": m.get("created"), "user": m.get("user"), "group": group})
    return jobs


@app.post("/api/incidents/dismiss")
async def dismiss_incident(body: dict, who: str = Depends(user)):
    """Mark a watchdog problem as handled: every unresolved incident with the same kind and summary."""
    path = Path.home() / "pf_watchdog" / "incidents.jsonl"
    kind, summary = body.get("kind"), body.get("summary")
    with _lock:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        n = 0
        for r in rows:
            if r.get("kind") == kind and r.get("summary", "")[:200] == summary and r.get("status") == "needs_attention":
                r.update(status="dismissed", resolution=f"dismissed by {who} on the dashboard",
                         resolved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                n += 1
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(path)
    return {"dismissed": n}


@app.post("/api/maps/{map_id}/folder")
async def set_map_folder(map_id: str, body: dict, who: str = Depends(user)):
    folder = _clean_folder(str(body.get("folder") or ""))
    if map_id.startswith("group:"):
        with _lock:
            pending = _pending()
            if map_id[6:] in pending:
                pending[map_id[6:]]["folder"] = folder
                _write_json(PENDING, pending)
    else:
        _set_folder(map_id, folder)
    return {"id": map_id, "folder": folder}


def _folders() -> dict[str, str]:
    try:
        return json.loads(FOLDERS.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _set_folder(map_id: str, folder: str) -> None:
    with _lock:
        f = _folders()
        if folder:
            f[map_id] = folder
        else:
            f.pop(map_id, None)
        tmp = FOLDERS.with_suffix(".tmp")
        tmp.write_text(json.dumps(f, indent=1, sort_keys=True))
        tmp.replace(FOLDERS)
    _export_folders()


def _export_folders() -> None:
    if VIEWER_DIR.is_dir():
        _write_json(VIEWER_DIR / "folders.json", _folders())


def _clean_folder(folder: str) -> str:
    return "/".join(part.strip() for part in folder.replace("\\", "/").split("/") if part.strip())


@app.get("/api/maps/{map_id}/points")
async def map_points(map_id: str, request: Request, limit: int = 80_000, who: str = Depends(user)):
    """The map's 3D points (reconstruction frame, the frame localizations are returned in) for the phone app's
    top-down view: little-endian float32 x, y, z triples, the most-observed points first, at most `limit`."""
    async with httpx.AsyncClient(timeout=30) as c:  # the caller must be able to see the map
        r = await c.get(f"{SERVER}/localization-maps/{map_id}", headers={"Authorization": request.headers["authorization"]})
    if r.status_code != 200:
        raise HTTPException(r.status_code, "map not found")
    src = VIEWER_DIR / "maps" / map_id / "map.json"
    if not src.is_file():
        raise HTTPException(404, "no point cloud exported for this map yet")
    key = (map_id, src.stat().st_mtime, limit)
    if key not in _points_cache:
        pts = json.loads(src.read_text())["points"]  # rows: x, y, z, r, g, b, observations
        pts.sort(key=lambda p: -p[6])
        _points_cache.clear()  # one map at a time is enough; the files are several MB
        _points_cache[key] = struct.pack(f"<{3 * min(limit, len(pts))}f", *(v for p in pts[:limit] for v in p[:3]))
    return Response(_points_cache[key], media_type="application/octet-stream")


_points_cache: dict[tuple, bytes] = {}


@app.get("/api/maps/{map_id}/contents")
async def map_contents(map_id: str, request: Request, who: str = Depends(user)):
    """What a map is made of: the original capture it started from, plus every recording merged into it through
    this tool (oldest first), and recordings currently queued or processing against it."""
    if map_id.startswith("group:"):
        pm = _pending().get(map_id[6:], {})
        items = [_recording(j) for j in _group_items(map_id[6:])]
        return {"base": {"map_id": map_id, "map_name": pm.get("name"), "pending": True}, "recordings": [],
                "pending": items}
    produced = _lineage()
    chain = _chain(map_id, produced)
    base = {"map_id": chain["base_map_id"]}
    async with httpx.AsyncClient(timeout=30) as c:
        h = {"Authorization": request.headers["authorization"]}
        try:
            lm = (await c.get(f"{SERVER}/localization-maps/{chain['base_map_id']}", headers=h)).json()
            rec = (await c.get(f"{SERVER}/reconstructions/{lm['reconstruction_id']}", headers=h)).json()
            cs = (await c.get(f"{SERVER}/capture_sessions/{rec['capture_session_id']}", headers=h)).json()
            base |= {"map_name": lm.get("name"), "capture_name": cs.get("name"),
                     "captured_at": cs.get("recorded_at") or cs.get("created_at")}
        except (httpx.HTTPError, KeyError, ValueError):
            pass
    pending = [_recording(j) for j in _all_jobs() if j["map_id"] == map_id and j["state"] in ("queued", "running", "failed")]
    return {"base": base, "recordings": chain["recordings"], "pending": pending}


def _all_jobs() -> list[dict]:
    out = []
    for jf in JOBS.glob("*/job.json"):
        try:
            out.append(json.loads(jf.read_text()))
        except ValueError:
            continue
    return sorted(out, key=lambda j: j.get("created") or 0)


def _lineage() -> dict[str, dict]:
    """map id -> the succeeded job that produced it."""
    return {j["result"]["map_id"]: j for j in _all_jobs()
            if j.get("state") == "succeeded" and (j.get("result") or {}).get("map_id")}


def _chain(map_id: str, produced: dict[str, dict]) -> dict:
    """Walk back through the jobs that produced map_id to the map it all started from."""
    recs, seen, cur = [], set(), map_id
    while cur in produced and cur not in seen:
        seen.add(cur)
        j = produced[cur]
        recs.append(_recording(j))
        if not j.get("map_id"):  # built as a new map: the recording is the map's own original capture
            break
        cur = j["map_id"]
    return {"base_map_id": cur, "recordings": recs[::-1]}


def _recording(j: dict) -> dict:
    """A recording as shown in the UI; the recording time is parsed from Insta360's VID_YYYYMMDD_HHMMSS name."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", j["filename"])
    recorded = f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}" if m else None
    r = j.get("result") or {}
    return {"job_id": j["id"], "filename": j["filename"], "recorded": recorded, "uploaded": j.get("created"),
            "state": j["state"], "user": j.get("user"), "odometry": j.get("odometry"),
            "into_map": j["map_id"], "produced_map": r.get("map_id"), "produced_name": r.get("name")}


def _combined_map(j: dict) -> dict | None:
    """All new-map recordings of one group (the upload page's 'Create a new map' group) become one map: the first to
    finish builds it, each later one is merged into the map the group has most recently produced. Returns
    {map_id, job} to merge into, or None when no recording of the group has built a map yet."""
    group = j.get("group") or _upload_meta(j).get("group")
    if not group:
        return None
    built = [o for o in _all_jobs() if o["id"] != j["id"] and o.get("state") == "succeeded"
             and (o.get("group") or _upload_meta(o).get("group")) == group and (o.get("result") or {}).get("map_id")]
    if not built:
        return None
    last = max(built, key=lambda o: o.get("finished") or 0)
    return {"map_id": last["result"]["map_id"], "job": last["id"]}


def _upload_meta(j: dict) -> dict:
    try:
        return json.loads(_meta_path(j.get("upload") or j["id"]).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _new_map_name(filename: str, folder: str = "") -> str:
    recorded = _recording({"id": "", "filename": filename, "map_id": None, "state": ""})["recorded"]
    leaf = folder.rsplit("/", 1)[-1]
    return f"{leaf} {recorded[:10]}" if leaf and recorded else f"X5 recording {recorded or Path(filename).stem}"


def _default_name(map_id: str, filename: str) -> str:
    """Name for a merged map when the user gives none: '<original map> + N recordings (dates)'. When the original map
    was itself built from a recording here (a new map), that recording is part of the original, not an addition."""
    produced = _lineage()
    chain = _chain(map_id, produced)
    recs = chain["recordings"]
    base_job = produced.get(chain["base_map_id"])
    if recs and base_job and not base_job.get("map_id"):  # chain starts with a new-map build
        base_name, recs = (base_job.get("result") or {}).get("name"), recs[1:]
    else:
        base_name = _base_names.get(chain["base_map_id"])
    new = _recording({"id": "", "filename": filename, "map_id": map_id, "state": ""})["recorded"]
    dates = sorted({r["recorded"][:10] if r["recorded"] else r["filename"] for r in recs} | {new[:10] if new else filename})
    n = len(recs) + 1
    return f"{base_name or chain['base_map_id'][:8]} + {n} recording{'s' if n > 1 else ''} ({', '.join(dates)})"


_base_names: dict[str, str] = {}  # map id -> name, refreshed by /api/maps


# --------------------------------------------------------------------------- resumable uploads


def _meta_path(uid: str) -> Path:
    return UPLOADS / f"{uid}.json"


def _load(uid: str) -> dict:
    p = _meta_path(uid)
    if not p.exists():
        raise HTTPException(404, "unknown upload")
    return json.loads(p.read_text())


def _save(meta: dict) -> None:
    tmp = _meta_path(meta["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(_meta_path(meta["id"]))


def _offset(meta: dict) -> int:
    p = UPLOADS / f"{meta['id']}.part"
    return p.stat().st_size if p.exists() else 0


@app.post("/api/inspect")
async def inspect_recording(request: Request, x_file_size: int = Header(...), x_head_len: int = Header(...),
                            who: str = Depends(user)):
    """Check a recording for GPS before uploading it. Insta360 keeps its metadata (gyro, exposure, GPS) in a trailer
    at the end of the .insv; the browser sends the first x_head_len bytes and the trailer (the rest of the body),
    which are written into a sparse file of the original size (no disk used for the gap) and parsed as usual."""
    body = await request.body()
    if not 0 < x_head_len <= len(body) or x_file_size < len(body) or len(body) > 96 * 2**20:
        raise HTTPException(400, "bad inspection request")
    return await asyncio.to_thread(_inspect, body[:x_head_len], body[x_head_len:], x_file_size)


def _inspect(head: bytes, tail: bytes, size: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".insv", dir=DATA) as t:
        t.write(head)
        t.truncate(size)
        t.seek(size - len(tail))
        t.write(tail)
        t.flush()
        try:
            fixes = gps_placement.insv_gps(Path(t.name))
            meta = gps_placement.insv_metadata(Path(t.name))
        except Exception as e:  # noqa: BLE001 - report unreadable files to the page instead of failing
            return {"ok": False, "error": f"could not read the recording's metadata ({type(e).__name__})"}
    return {"ok": True, "gps_fixes": len(fixes), "gps": gps_placement.median_fix(fixes), **meta}


@app.post("/api/uploads")
async def create_upload(body: dict, who: str = Depends(user)):
    name = Path(str(body.get("filename", ""))).name
    size = int(body.get("size", 0))
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, "upload the raw .insv recording")
    target = str(body.get("map_id") or "")
    if target.startswith("group:"):  # a pending map: the recording joins its group
        body["new_map"], body["group"] = True, target[6:]
    body["new_map"] = bool(body.get("new_map"))
    if body["new_map"]:
        body["map_id"] = None
    else:
        body["group"] = None
    if size <= 0 or not (body.get("map_id") or body["new_map"]):
        raise HTTPException(400, "size and either map_id or new_map are required")
    body["odometry"] = body.get("odometry") or ODOMETRY[0]
    if body["odometry"] not in ODOMETRY:
        raise HTTPException(400, f"odometry must be one of {', '.join(ODOMETRY)}")
    fp = hashlib.sha1(f"{who}|{name}|{size}|{body.get('last_modified')}".encode()).hexdigest()
    with _lock:
        for p in UPLOADS.glob("*.json"):
            m = json.loads(p.read_text())
            if m["fingerprint"] == fp and not m.get("complete"):
                m.update({k: body[k] for k in ("map_id", "new_map", "group", "folder", "new_name", "replace",
                                                "odometry") if k in body})
                _save(m)
                return {"id": m["id"], "offset": _offset(m), "size": m["size"], "resumed": True}
        uid = uuid.uuid4().hex[:12]
        meta = {"id": uid, "fingerprint": fp, "user": who, "filename": name, "size": size,
                "map_id": body["map_id"], "new_map": body["new_map"], "folder": _clean_folder(str(body.get("folder") or "")),
                # New-map uploads carry the page's group id: every recording of a group goes into one map.
                # Without one (e.g. scripted uploads) each new-map upload is its own group.
                "group": (str(body.get("group") or "")[:64] or uuid.uuid4().hex[:12]) if body["new_map"] else None,
                "new_name": body.get("new_name") or None,
                "replace": bool(body.get("replace")), "odometry": body["odometry"], "created": time.time(),
                "complete": False}
        _save(meta)
    return {"id": uid, "offset": 0, "size": size, "resumed": False}


@app.post("/api/uploads/check")
async def check_uploads(body: dict, who: str = Depends(user)):
    """Which of these recordings (name + size, as on the camera's card) this service already has: 'uploaded' (complete,
    with its processing job), 'uploading' (partial and receiving data now), 'paused' (partial, resumable) or 'new'.
    Matched on file name and size, which stay the same however the card is read."""
    known: dict[tuple[str, int], dict] = {}
    for mp in UPLOADS.glob("*.json"):
        try:
            m = json.loads(mp.read_text())
        except ValueError:
            continue
        key = (m["filename"], int(m["size"]))
        if m.get("complete") or key not in known:  # a complete upload wins over stale partial ones
            known[key] = m
    out = []
    for f in body.get("files") or []:
        name, size = Path(str(f.get("name", ""))).name, int(f.get("size") or 0)
        m = known.get((name, size))
        if not m:
            out.append({"name": name, "size": size, "status": "new"})
            continue
        item = {"name": name, "size": size, "upload_id": m["id"]}
        if m.get("complete"):
            jf = JOBS / m["id"] / "job.json"
            job = json.loads(jf.read_text()) if jf.exists() else {}
            item |= {"status": "uploaded", "job_state": job.get("state"),
                     "map_name": (job.get("result") or {}).get("name"), "new_map": bool(m.get("new_map"))}
        else:
            part = UPLOADS / f"{m['id']}.part"
            active = part.exists() and time.time() - part.stat().st_mtime < UPLOAD_ACTIVE_SECONDS
            item |= {"status": "uploading" if active else "paused", "offset": _offset(m)}
        out.append(item)
    return out


@app.get("/api/uploads/{uid}")
async def upload_status(uid: str, who: str = Depends(user)):
    m = _load(uid)
    return {"id": uid, "offset": _offset(m), "size": m["size"], "complete": m.get("complete", False),
            "job_id": m.get("job_id")}


@app.patch("/api/uploads/{uid}")
async def upload_chunk(uid: str, request: Request, upload_offset: int = Header(...),
                       x_chunk_sha256: str = Header(default=""), who: str = Depends(user)):
    m = _load(uid)
    if m.get("complete"):
        return {"offset": m["size"], "complete": True, "job_id": m.get("job_id")}
    data = await request.body()
    if len(data) > CHUNK_MAX:
        raise HTTPException(413, "chunk too large")
    if x_chunk_sha256 and hashlib.sha256(data).hexdigest() != x_chunk_sha256.lower():
        raise HTTPException(422, "chunk checksum mismatch — resend")
    with _lock:
        cur = _offset(m)
        if upload_offset != cur:
            return JSONResponse({"offset": cur}, status_code=409)
        if cur + len(data) > m["size"]:
            raise HTTPException(400, "chunk overruns declared size")
        with open(UPLOADS / f"{uid}.part", "ab") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        cur += len(data)
        if cur == m["size"]:
            final = UPLOADS / f"{uid}_{m['filename']}"
            (UPLOADS / f"{uid}.part").rename(final)
            m.update({"complete": True, "path": str(final), "completed": time.time()})
            m["job_id"] = _create_job(m)
            _save(m)
            return {"offset": cur, "complete": True, "job_id": m["job_id"]}
    return {"offset": cur, "complete": False}


# --------------------------------------------------------------------------- jobs


def _create_job(upload: dict) -> str:
    jid = upload["id"]
    jd = JOBS / jid
    jd.mkdir(parents=True, exist_ok=True)
    job = {"id": jid, "upload": upload["id"], "file": upload["path"], "filename": upload["filename"],
           "map_id": upload["map_id"], "new_map": bool(upload.get("new_map")), "group": upload.get("group"),
           "folder": upload.get("folder") or "",
           "new_name": upload.get("new_name"), "replace": upload.get("replace"),
           "odometry": upload.get("odometry") or ODOMETRY[0], "user": upload["user"], "state": "queued", "created": time.time()}
    (jd / "job.json.tmp").write_text(json.dumps(job, indent=2))
    (jd / "job.json.tmp").replace(jd / "job.json")
    _queue.put(jid)
    return jid


def _job(jid: str) -> dict:
    return json.loads((JOBS / jid / "job.json").read_text())


def _set(jid: str, **kw) -> None:
    j = _job(jid) | kw
    tmp = JOBS / jid / "job.json.tmp"
    tmp.write_text(json.dumps(j, indent=2))
    tmp.replace(JOBS / jid / "job.json")  # atomic: a kill mid-write never leaves an empty job file


def _held() -> str | None:
    if not HOLD.exists():
        return None
    return HOLD.read_text(errors="replace").strip() or "held by the operator"


# Jobs run up to this many at a time: one recording's preprocessing (CPU: decode, odometry, stitching) overlaps
# another's reconstruction (GPU, on the server). Recordings of the same group, or merges into the same map, still
# run one after another: each must merge into the map the previous one produced.
PARALLEL_JOBS = int(os.environ.get("MAPPER_PARALLEL_JOBS", "2"))
_running_keys: set[str] = set()


def _exclusive_key(j: dict) -> str:
    return "group:" + (j.get("group") or _upload_meta(j).get("group") or "") if (
        j.get("group") or _upload_meta(j).get("group")) else ("map:" + j["map_id"] if j.get("map_id") else "job:" + j["id"])


def _worker() -> None:
    while True:
        jid = _queue.get()
        while _held():  # jobs stay "queued" until the HOLD file is removed
            time.sleep(10)
        j = _job(jid)
        if j["state"] == "succeeded":
            continue
        key = _exclusive_key(j)
        with _lock:
            busy = key in _running_keys
            if not busy:
                _running_keys.add(key)
        if busy:  # a job of the same group/map is running: try again once it may have finished
            threading.Timer(15, _queue.put, args=(jid,)).start()
            continue
        try:
            _run_job(jid, j)
        finally:
            with _lock:
                _running_keys.discard(key)


def _run_job(jid: str, j: dict) -> None:
    _set(jid, state="running", started=time.time(), error=None)
    if j.get("new_map"):
        base = _combined_map(j)
        if base:  # an earlier recording of the same upload group already built the map: merge into it
            j = _set(jid, new_map=False, map_id=base["map_id"], combined_into=base["job"]) or _job(jid)
    target = ["--new-map"] if j.get("new_map") else ["--map", j["map_id"]]
    cmd = [sys.executable, "-u", str(REPO / "add_recording_to_map.py"), j["file"], *target,
           "--job-dir", str(JOBS / jid / "work"), "--odometry", j.get("odometry") or ODOMETRY[0]]
    pending_name = _pending().get(j.get("group") or "", {}).get("name")
    default = ((pending_name or _new_map_name(j["filename"], j.get("folder") or "")) if j.get("new_map")
               else _default_name(j["map_id"], j["filename"]))
    cmd += ["--name", j.get("new_name") or default]
    if j.get("replace"):
        cmd.append("--replace")
    with open(JOBS / jid / "log.txt", "a") as log:
        log.write(f"\n### run started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.flush()
        # Own process group: a restart stops the job with everything it started (converters, ffmpeg).
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
    result = JOBS / jid / "work" / "result.json"
    if rc == 0 and result.exists():
        res = json.loads(result.read_text())
        # A merged map goes in its base map's folder; a new map in the folder chosen at upload.
        folder = ((_pending().get(j.get("group") or "", {}).get("folder") or j.get("folder")) if j.get("new_map")
                  else _folders().get(j["map_id"], ""))
        if folder and res.get("map_id"):
            _set_folder(res["map_id"], folder)
        _set(jid, state="succeeded", finished=time.time(), result=res)
    else:
        _set(jid, state="failed", finished=time.time(), error=f"exit code {rc} — see log")


@app.get("/api/jobs")
async def jobs(who: str = Depends(user)):
    out = []
    held = _held()
    for jd in sorted(JOBS.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        try:
            j = json.loads(jd.read_text())
        except ValueError:
            continue
        log = jd.parent / "log.txt"
        lines = log.read_text(errors="replace").splitlines() if log.exists() else []
        names = stages(j.get("odometry") or ODOMETRY[0], bool(j.get("new_map")))
        done = [s for s in names if (jd.parent / "work" / f".done-{s}").exists()]
        current = next((s for s in names if s not in done), None) if j["state"] == "running" else None
        out.append(j | {"stages": names, "done": done, "current": current, "log": lines[-60:],
                        "held": f"held: {held}" if held and j["state"] == "queued" else None})
    return out


@app.post("/api/jobs/{jid}/retry")
async def retry(jid: str, who: str = Depends(user)):
    if _job(jid)["state"] in ("failed",):
        _set(jid, state="queued")
        _queue.put(jid)
    return {"ok": True}


def _api_factory():
    from add_recording_to_map import PlaceframeApi

    return PlaceframeApi()


TELEMETRY = Telemetry(JOBS, _api_factory)


@app.get("/api/status")
async def status(who: str = Depends(user)):
    snap = dict(TELEMETRY.get())
    snap["hold"] = _held()
    return snap


# Rented GPUs (Vast.ai): running instances with their cost, account credit, and the fleet broker's recent rentals.
@app.get("/api/rentals")
async def rentals_status(who: str = Depends(user)):
    return await asyncio.to_thread(rentals.snapshot)


COST_REPORT = Path.home() / "pf_cost" / "report"


@app.get("/cost/{name:path}")
async def cost_report(name: str, request: Request):
    _tailnet_only(request, "the cost report")
    path = (COST_REPORT / (name or "index.html")).resolve()
    if not path.is_relative_to(COST_REPORT.resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


# Generated videos (track replays, renders): every video under the video folder. A "<name>.txt" next to a video is
# shown as its description. The folder is on the SSD: listing the HDD scratch folder took ~10 s cold (and nothing
# while the HDD was offline); the HDD keeps archives only.
VIDEO_ROOT = Path(os.environ.get("MAPPER_VIDEO_ROOT", str(Path.home() / "pf_data" / "videos")))
VIDEO_EXT = {".mp4", ".webm", ".mov", ".m4v"}
VIDEO_SKIP = {"uploads", "datasets", "docker_archive"}


def _tailnet_only(request: Request, what: str) -> None:
    # <img>/<video> tags can't send the bearer token, so public (Funnel) requests are refused instead.
    if request.headers.get("Tailscale-Funnel-Request"):
        raise HTTPException(403, f"{what} is only available on the tailnet")


@app.get("/videos/api/list")
async def video_list(request: Request):
    _tailnet_only(request, "the video gallery")
    out = []
    for dirpath, dirs, files in os.walk(VIDEO_ROOT):
        if Path(dirpath) == VIDEO_ROOT:
            dirs[:] = [d for d in dirs if d not in VIDEO_SKIP and not d.startswith("git_backup")]
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() not in VIDEO_EXT:
                continue
            st = p.stat()
            note = p.with_suffix(".txt")
            rel = p.relative_to(VIDEO_ROOT).as_posix()
            out.append({"path": rel, "name": p.stem, "folder": Path(rel).parent.as_posix().removeprefix("."),
                        "size": st.st_size, "mtime": st.st_mtime,
                        "note": note.read_text(errors="replace").strip() if note.is_file() else ""})
    out.sort(key=lambda v: -v["mtime"])
    return {"root": str(VIDEO_ROOT), "videos": out}


@app.get("/videos/file/{name:path}")
async def video_file(name: str, request: Request):
    _tailnet_only(request, "the video gallery")
    path = (VIDEO_ROOT / name).resolve()
    if not path.is_relative_to(VIDEO_ROOT.resolve()) or not path.is_file() or path.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(404)
    return FileResponse(path)  # Starlette answers Range requests, so the player can seek


@app.get("/videos")
async def videos_redirect():
    return Response(status_code=308, headers={"Location": "videos/"})


@app.get("/videos/")
async def videos_page():
    return FileResponse(HERE / "static" / "videos.html", headers={"Cache-Control": "no-cache"})


app.include_router(project_router)  # /placeframe/ home, visualizer, vendor, reports (mapper/project_pages.py)


@app.get("/pf_ui.js")
async def shared_ui():
    return FileResponse(REPO / "web" / "pf_ui.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/")
async def index():
    # no-cache: browsers revalidate on every load, so dashboard changes show up without a hard refresh
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.on_event("startup")
def _startup() -> None:
    _export_folders()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    # Resume interrupted work: running/queued jobs restart after their last completed stage.
    for jd in sorted(JOBS.glob("*/job.json"), key=lambda p: p.stat().st_mtime):
        try:
            j = json.loads(jd.read_text())
        except ValueError:
            print(f"skipping unreadable {jd}", flush=True)
            continue
        if j["state"] in ("queued", "running"):
            _set(j["id"], state="queued")
            _queue.put(j["id"])
    TELEMETRY.start()
    if os.environ.get("MAPPER_NO_JOBS"):
        return
    for _ in range(PARALLEL_JOBS):
        threading.Thread(target=_worker, daemon=True).start()
