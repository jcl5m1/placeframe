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
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = Path(os.environ.get("MAPPER_DATA", Path.home() / "pf_mapper"))
UPLOADS, JOBS = DATA / "uploads", DATA / "jobs"
HOLD = DATA / "HOLD"
CHUNK_MAX = 64 * 1024 * 1024
ALLOWED_EXT = {".insv"}

sys.path.insert(0, str(REPO))
from upload_to_placeframe import saved_credentials  # noqa: E402

from .telemetry import Telemetry  # noqa: E402

CREDS = saved_credentials(REPO.parent / "placeframe")
SERVER = CREDS["PLACEFRAME_URL"].rstrip("/")
REALM = "placeframe-dev"
ODOMETRY = ("openvio", "openvins")  # add_recording_to_map.py --odometry; first = default


def stages(odometry: str) -> list[str]:  # mirrors add_recording_to_map.stages()
    return ["calibrate", f"{odometry}-export", f"{odometry}-run", "stitch", "views", "capture", "register", "merge",
            "reconstruct", "publish"]

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
    _base_names.update({m["id"]: m.get("name") or m["id"][:8] for m in r.json()})
    items = []
    for m in r.json():
        chain = _chain(m["id"], produced)
        items.append({"id": m["id"], "name": m.get("name") or m["id"][:8], "active": m.get("active"),
                      "created_at": m.get("created_at"), "base_map_id": chain["base_map_id"],
                      "recordings": len(chain["recordings"])})
    by_id = {m["id"]: m for m in items}
    for m in items:
        base = by_id.get(m["base_map_id"])
        m["base_name"] = base["name"] if base else m["base_map_id"][:8]
    return sorted(items, key=lambda m: (not m["active"], m["created_at"] or ""), reverse=False)


@app.get("/api/maps/{map_id}/contents")
async def map_contents(map_id: str, request: Request, who: str = Depends(user)):
    """What a map is made of: the original capture it started from, plus every recording merged into it through
    this tool (oldest first), and recordings currently queued or processing against it."""
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


def _default_name(map_id: str, filename: str) -> str:
    """Name for a merged map when the user gives none: '<original map> + N recordings (dates)'."""
    chain = _chain(map_id, _lineage())
    dates = [r["recorded"][:10] if r["recorded"] else r["filename"] for r in chain["recordings"]]
    new = _recording({"id": "", "filename": filename, "map_id": map_id, "state": ""})["recorded"]
    dates.append(new[:10] if new else filename)
    base = _base_names.get(chain["base_map_id"]) or chain["base_map_id"][:8]
    n = len(dates)
    return f"{base} + {n} recording{'s' if n > 1 else ''} ({', '.join(dates)})"


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


@app.post("/api/uploads")
async def create_upload(body: dict, who: str = Depends(user)):
    name = Path(str(body.get("filename", ""))).name
    size = int(body.get("size", 0))
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, "upload the raw .insv recording")
    if size <= 0 or not body.get("map_id"):
        raise HTTPException(400, "size and map_id are required")
    body["odometry"] = body.get("odometry") or ODOMETRY[0]
    if body["odometry"] not in ODOMETRY:
        raise HTTPException(400, f"odometry must be one of {', '.join(ODOMETRY)}")
    fp = hashlib.sha1(f"{who}|{name}|{size}|{body.get('last_modified')}".encode()).hexdigest()
    with _lock:
        for p in UPLOADS.glob("*.json"):
            m = json.loads(p.read_text())
            if m["fingerprint"] == fp and not m.get("complete"):
                m.update({k: body[k] for k in ("map_id", "new_name", "replace", "odometry") if k in body})
                _save(m)
                return {"id": m["id"], "offset": _offset(m), "size": m["size"], "resumed": True}
        uid = uuid.uuid4().hex[:12]
        meta = {"id": uid, "fingerprint": fp, "user": who, "filename": name, "size": size,
                "map_id": body["map_id"], "new_name": body.get("new_name") or None,
                "replace": bool(body.get("replace")), "odometry": body["odometry"], "created": time.time(),
                "complete": False}
        _save(meta)
    return {"id": uid, "offset": 0, "size": size, "resumed": False}


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
           "map_id": upload["map_id"], "new_name": upload.get("new_name"), "replace": upload.get("replace"),
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


def _worker() -> None:
    while True:
        jid = _queue.get()
        while _held():  # jobs stay "queued" until the HOLD file is removed
            time.sleep(10)
        j = _job(jid)
        if j["state"] == "succeeded":
            continue
        _set(jid, state="running", started=time.time(), error=None)
        cmd = [sys.executable, "-u", str(REPO / "add_recording_to_map.py"), j["file"], "--map", j["map_id"],
               "--job-dir", str(JOBS / jid / "work"), "--odometry", j.get("odometry") or ODOMETRY[0]]
        cmd += ["--name", j.get("new_name") or _default_name(j["map_id"], j["filename"])]
        if j.get("replace"):
            cmd.append("--replace")
        with open(JOBS / jid / "log.txt", "a") as log:
            log.write(f"\n### run started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        result = JOBS / jid / "work" / "result.json"
        if rc == 0 and result.exists():
            _set(jid, state="succeeded", finished=time.time(), result=json.loads(result.read_text()))
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
        names = stages(j.get("odometry") or ODOMETRY[0])
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


COST_REPORT = Path.home() / "pf_cost" / "report"


@app.get("/cost/{name:path}")
async def cost_report(name: str, request: Request):
    # Tailnet only: <img> tags can't send the bearer token, so public (Funnel) requests are refused instead.
    if request.headers.get("Tailscale-Funnel-Request"):
        raise HTTPException(403, "cost report is only available on the tailnet")
    path = (COST_REPORT / (name or "index.html")).resolve()
    if not path.is_relative_to(COST_REPORT.resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@app.get("/")
async def index():
    # no-cache: browsers revalidate on every load, so dashboard changes show up without a hard refresh
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.on_event("startup")
def _startup() -> None:
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
    threading.Thread(target=_worker, daemon=True).start()
