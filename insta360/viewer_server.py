#!/usr/bin/env python3
"""Map review UI server: static viewer + resumable chunked uploads + the merge job queue.

  uv run viewer_server.py                       # http://0.0.0.0:8131
  uv run viewer_server.py --port 8131 --maps ~/pf_x5/viewer --data ~/pf_x5/merge_ui

--maps is the existing viewer folder (map.json); it is only read. Everything this server writes goes under --data:
  maps/<name>/map.json and current_map.json, written when a merge finishes
  uploads/  <upload id>/{meta.json, data.part, chunks.log}  ->  <upload id>/<original name> when complete
  jobs/     jobs.json, <job id>/{job.log, ...merge_video.py work...}

Uploads: the browser cuts the file into chunks and PUTs each one separately, several at a time. Every chunk
is written straight to its offset and recorded in chunks.log once its length and CRC32 check out, so an
upload survives dropped connections, a closed tab or a server restart — the client asks which chunks
arrived and sends only the rest. A finished upload is queued for merge_video.py; jobs run one at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "viewer"
CHUNK_SIZE = 16 * 1024 * 1024
ALLOWED_EXT = {".insv"}
STAGE_RE = re.compile(r"^### STAGE (\d+)/(\d+) (\w+)", re.M)


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- uploads


class Upload:
    def __init__(self, folder: Path, meta: dict) -> None:
        self.folder = folder
        self.meta = meta
        self.lock = threading.Lock()
        self.received: set[int] = set()
        log = folder / "chunks.log"
        if log.exists():
            self.received = {int(x) for x in log.read_text().split() if x.isdigit() and int(x) < self.chunk_count}

    @property
    def size(self) -> int:
        return self.meta["size"]

    @property
    def chunk_count(self) -> int:
        return max(1, -(-self.size // CHUNK_SIZE))

    @property
    def part(self) -> Path:
        return self.folder / "data.part"

    @property
    def final(self) -> Path:
        return self.folder / self.meta["name"]

    def chunk_length(self, index: int) -> int:
        return min(CHUNK_SIZE, self.size - index * CHUNK_SIZE)

    def received_bytes(self) -> int:
        with self.lock:
            return sum(self.chunk_length(i) for i in self.received)

    def describe(self) -> dict:
        with self.lock:
            received = sorted(self.received)
        return {"id": self.meta["id"], "name": self.meta["name"], "size": self.size, "chunkSize": CHUNK_SIZE,
                "chunks": self.chunk_count, "received": received, "complete": self.final.exists(),
                "job": self.meta.get("job"), "created": self.meta["created"]}


class Uploads:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.items: dict[str, Upload] = {}
        for meta_path in root.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                self.items[meta["id"]] = Upload(meta_path.parent, meta)
            except (ValueError, KeyError):
                continue

    def get(self, upload_id: str) -> Upload:
        with self.lock:
            up = self.items.get(upload_id)
        if up is None:
            raise ApiError(404, "Unknown upload — start it again")
        return up

    def start(self, name: str, size: int, last_modified: int) -> Upload:
        name = os.path.basename(name.replace("\\", "/")).strip()
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
        if not name or Path(name).suffix.lower() not in ALLOWED_EXT:
            raise ApiError(400, "Only raw Insta360 .insv recordings can be added (the pipeline needs the "
                                "dual-fisheye video and its IMU track)")
        if size <= 0:
            raise ApiError(400, "Empty file")
        # The same file (name, size, modified time) always maps to the same upload, which is what lets a
        # re-selected file resume — from another tab, after a reload, or after a server restart.
        upload_id = hashlib.sha1(f"{name}|{size}|{last_modified}".encode()).hexdigest()[:16]
        with self.lock:
            up = self.items.get(upload_id)
            if up is None:
                missing = size
                free = shutil.disk_usage(self.root).free
                # The upload itself plus roughly 1.5x its size of extracted frames while the job runs.
                needed = int(missing * 2.5) + 2 * 1024**3
                if free < needed:
                    raise ApiError(507, f"Not enough disk space on the server: {free / 1e9:.0f} GB free, about "
                                        f"{needed / 1e9:.0f} GB needed for this recording and its processing")
                folder = self.root / upload_id
                folder.mkdir(parents=True, exist_ok=True)
                meta = {"id": upload_id, "name": name, "size": size, "lastModified": last_modified,
                        "created": time.time()}
                write_json(folder / "meta.json", meta)
                up = Upload(folder, meta)
                if not up.final.exists():
                    with open(up.part, "ab") as fh:
                        fh.truncate(size)  # sparse; chunks land at their own offsets in any order
                self.items[upload_id] = up
        return up

    def discard(self, upload_id: str) -> None:
        with self.lock:
            up = self.items.pop(upload_id, None)
        if up is not None:
            shutil.rmtree(up.folder, ignore_errors=True)

    def unfinished(self) -> list[dict]:
        with self.lock:
            ups = list(self.items.values())
        out = []
        for up in ups:
            if not up.final.exists():
                out.append({"id": up.meta["id"], "name": up.meta["name"], "size": up.size,
                            "receivedBytes": up.received_bytes(), "created": up.meta["created"]})
        return sorted(out, key=lambda u: -u["created"])


def receive_chunk(up: Upload, index: int, rfile, length: int, crc_header: str | None) -> None:
    if not 0 <= index < up.chunk_count:
        raise ApiError(400, "Chunk index out of range")
    if length != up.chunk_length(index):
        raise ApiError(400, f"Chunk {index} should be {up.chunk_length(index)} bytes, got {length}")
    if up.final.exists():
        raise ApiError(409, "Upload already complete")
    fd = os.open(up.part, os.O_WRONLY)
    try:
        offset, remaining, crc = index * CHUNK_SIZE, length, 0
        while remaining:
            data = rfile.read(min(1 << 20, remaining))
            if not data:
                raise ConnectionError("connection closed mid-chunk")
            crc = zlib.crc32(data, crc)
            os.pwrite(fd, data, offset)
            offset += len(data)
            remaining -= len(data)
        os.fsync(fd)
    finally:
        os.close(fd)
    if crc_header is not None and int(crc_header) != crc:
        raise ApiError(422, f"Chunk {index} was corrupted in transit (checksum mismatch)")
    with up.lock:
        if index not in up.received:
            up.received.add(index)
            with open(up.folder / "chunks.log", "a") as fh:
                fh.write(f"{index}\n")
                fh.flush()
                os.fsync(fh.fileno())


# --------------------------------------------------------------------------- jobs


class Jobs:
    def __init__(self, root: Path, data_root: Path, maps_root: Path, extra_args: list[str]) -> None:
        self.root = root
        self.data_root = data_root
        self.maps_root = maps_root
        self.extra_args = extra_args
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "jobs.json"
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.jobs: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []
        self.proc: subprocess.Popen | None = None
        for job in self.jobs:
            # A job that was running when the server stopped either still is (it runs in its own session)
            # or died with it; the worker checks the pid and resumes the pipeline if it is gone.
            if job["status"] == "running":
                job["status"] = "orphaned"
        threading.Thread(target=self._worker, daemon=True).start()

    def _save(self) -> None:
        write_json(self.path, self.jobs)

    def base_map(self) -> str:
        current = self.data_root / "current_map.json"
        if current.exists():
            return json.loads(current.read_text())["name"]
        return json.loads((self.maps_root / "map.json").read_text())["name"]

    def add(self, video: Path, upload_id: str) -> dict:
        with self.lock:
            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{upload_id[:6]}"
            job = {"id": job_id, "video": str(video), "name": video.name, "upload": upload_id, "status": "queued",
                   "created": time.time(), "started": None, "finished": None, "pid": None, "error": None}
            self.jobs.append(job)
            self._save()
        self.wake.set()
        return job

    def find(self, job_id: str) -> dict:
        for job in self.jobs:
            if job["id"] == job_id:
                return job
        raise ApiError(404, "Unknown job")

    def retry(self, job_id: str) -> None:
        with self.lock:
            job = self.find(job_id)
            if job["status"] not in ("failed", "cancelled"):
                raise ApiError(409, f"Job is {job['status']}")
            job.update(status="queued", error=None, finished=None)
            self._save()
        self.wake.set()

    def cancel(self, job_id: str) -> None:
        with self.lock:
            job = self.find(job_id)
            if job["status"] == "queued":
                job.update(status="cancelled", finished=time.time())
            elif job["status"] == "running" and job.get("pid"):
                job["cancel_requested"] = True
                try:
                    os.killpg(job["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                raise ApiError(409, f"Job is {job['status']}")
            self._save()

    def log_path(self, job_id: str) -> Path:
        return self.root / job_id / "job.log"

    def describe(self) -> list[dict]:
        with self.lock:
            jobs = [dict(j) for j in self.jobs]
        for job in jobs:
            tail = self._tail(self.log_path(job["id"]), 256 * 1024)
            stages = STAGE_RE.findall(tail)
            if stages:
                job["stageIndex"], job["stageCount"], job["stage"] = int(stages[-1][0]), int(stages[-1][1]), stages[-1][2]
            lines = [ln for ln in tail.splitlines() if ln.strip()]
            job["lastLine"] = lines[-1][-300:] if lines else ""
            job.pop("pid", None)
        return jobs[::-1]

    @staticmethod
    def _tail(path: Path, size: int) -> str:
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - size))
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""

    @staticmethod
    def _alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
        return b"merge_video.py" in cmdline

    def _worker(self) -> None:
        while True:
            with self.lock:
                job = next((j for j in self.jobs if j["status"] in ("orphaned", "queued")), None)
            if job is None:
                self.wake.wait(5)
                self.wake.clear()
                continue
            try:
                self._run(job)
            except Exception as e:  # noqa: BLE001 — the worker must outlive any one job
                with self.lock:
                    job.update(status="failed", error=f"{type(e).__name__}: {e}", finished=time.time())
                    self._save()

    def _run(self, job: dict) -> None:
        job_dir = self.root / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_path(job["id"])
        if job["status"] == "orphaned" and self._alive(job.get("pid")):
            with self.lock:
                job["status"] = "running"
                self._save()
            while self._alive(job["pid"]):
                time.sleep(5)
            code = 0 if (job_dir / ".done" / "viewer").exists() else 1
        else:
            cmd = [sys.executable, "-u", str(HERE / "merge_video.py"), job["video"], "--job-dir", str(job_dir),
                   "--base-map", self.base_map(), "--viewer-root", str(self.data_root), *self.extra_args]
            with open(log_path, "ab") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting\n".encode())
                log.flush()
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True)
            with self.lock:
                job.update(status="running", started=job.get("started") or time.time(), pid=proc.pid, error=None)
                job.pop("cancel_requested", None)
                self._save()
            code = proc.wait()
        with self.lock:
            if job.pop("cancel_requested", False):
                job.update(status="cancelled", error=None)
            elif code == 0:
                job.update(status="succeeded", error=None)
            else:
                lines = [ln for ln in self._tail(log_path, 8192).splitlines() if ln.strip()]
                job.update(status="failed", error=lines[-1][-400:] if lines else f"exit code {code}")
            job.update(finished=time.time(), pid=None)
            self._save()


# --------------------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 120  # a stalled connection is dropped; the browser re-sends that chunk
    server_version = "PlaceframeViewer/1"
    uploads: Uploads
    jobs: Jobs
    maps_root: Path
    data_root: Path

    def log_message(self, fmt, *args) -> None:  # quiet: chunk PUTs would flood the log
        if not self.path.startswith("/api/uploads/") and not self.path.startswith("/api/state"):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    # ---- helpers
    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1 << 20:
            raise ApiError(413, "Request too large")
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError as e:
            raise ApiError(400, "Invalid JSON") from e

    def dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        try:
            if parts[:1] == ["api"]:
                self.api(method, parts[1:], parse_qs(url.query))
            elif method == "GET":
                self.static(url.path)
            else:
                raise ApiError(405, "Method not allowed")
        except ApiError as e:
            self.close_connection = True  # the request body may be unread
            self.send_json({"error": e.message}, e.status)
        except (ConnectionError, TimeoutError, BrokenPipeError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        self.dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self.dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self.dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self.dispatch("DELETE")

    # ---- static files: the page from the repo, map data from the data folder
    def static(self, path: str) -> None:
        rel = unquote(path).lstrip("/") or "index.html"
        for root in (STATIC, self.data_root, self.maps_root):
            if root is self.data_root and not rel.startswith("maps/"):
                continue  # only finished maps are public from the data folder, not uploads or job logs
            target = (root / rel).resolve()
            if root.resolve() in target.parents and target.is_file():
                break
        else:
            raise ApiError(404, "Not found")
        size = target.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(target, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    # ---- API
    def api(self, method: str, parts: list[str], query: dict) -> None:
        if parts == ["state"] and method == "GET":
            self.send_json(self.state())
        elif parts == ["uploads"] and method == "POST":
            body = self.read_json()
            up = self.uploads.start(str(body.get("name", "")), int(body.get("size", 0)), int(body.get("lastModified", 0)))
            self.send_json(up.describe())
        elif len(parts) == 4 and parts[0] == "uploads" and parts[2] == "chunks" and method == "PUT":
            up = self.uploads.get(parts[1])
            receive_chunk(up, int(parts[3]), self.rfile, int(self.headers.get("Content-Length") or 0),
                          self.headers.get("X-Chunk-Crc32"))
            self.send_json({"ok": True})
        elif len(parts) == 3 and parts[0] == "uploads" and parts[2] == "complete" and method == "POST":
            self.read_json()
            self.send_json(self.complete(self.uploads.get(parts[1])))
        elif len(parts) == 2 and parts[0] == "uploads" and method == "DELETE":
            up = self.uploads.get(parts[1])
            if up.final.exists():
                raise ApiError(409, "Upload already complete")
            self.uploads.discard(parts[1])
            self.send_json({"ok": True})
        elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "log" and method == "GET":
            self.jobs.find(parts[1])
            body = Jobs._tail(self.jobs.log_path(parts[1]), 512 * 1024).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif len(parts) == 3 and parts[0] == "jobs" and parts[2] in ("retry", "cancel") and method == "POST":
            self.read_json()
            getattr(self.jobs, parts[2])(parts[1])
            self.send_json({"ok": True})
        else:
            raise ApiError(404, "No such endpoint")

    def complete(self, up: Upload) -> dict:
        with up.lock:
            if not up.final.exists():
                missing = [i for i in range(up.chunk_count) if i not in up.received]
                if missing:
                    raise ApiError(409, f"{len(missing)} chunks still missing")
                if up.part.stat().st_size != up.size:
                    raise ApiError(500, "Assembled file has the wrong size")
                os.replace(up.part, up.final)
            if not up.meta.get("job"):
                up.meta["job"] = self.jobs.add(up.final, up.meta["id"])["id"]
                write_json(up.folder / "meta.json", up.meta)
        return up.describe()

    def state(self) -> dict:
        maps = []
        root_map = self.maps_root / "map.json"
        if root_map.exists():
            maps.append({"path": "map.json", "label": "original map"})
        for p in sorted(self.data_root.glob("maps/*/map.json"), key=lambda p: p.stat().st_mtime):
            maps.append({"path": f"maps/{p.parent.name}/map.json", "label": p.parent.name})
        current = self.data_root / "current_map.json"
        return {"maps": maps, "current": json.loads(current.read_text()) if current.exists() else None,
                "baseMap": self.jobs.base_map(), "uploads": self.uploads.unfinished(), "jobs": self.jobs.describe(),
                "diskFree": shutil.disk_usage(self.uploads.root).free, "chunkSize": CHUNK_SIZE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--maps", default="~/pf_x5/viewer", help="existing viewer folder with map.json (read only)")
    ap.add_argument("--data", default="~/pf_x5/merge_ui", help="folder for uploads/, jobs/ and merged maps/")
    ap.add_argument("--merge-args", default="", help="extra arguments passed to merge_video.py, e.g. '--keep-work'")
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()
    data.mkdir(parents=True, exist_ok=True)
    Handler.maps_root = Path(args.maps).expanduser().resolve()
    Handler.data_root = data
    Handler.uploads = Uploads(data / "uploads")
    Handler.jobs = Jobs(data / "jobs", data, Handler.maps_root, args.merge_args.split())
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"viewer on http://{args.host}:{args.port}  (data: {data})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
