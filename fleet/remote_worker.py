#!/usr/bin/env python3
"""Runs one Placeframe reconstruction on a rented GPU with no credentials: everything it may read or write is a
pre-signed R2 URL in its ticket (issued and logged by fleet/broker.py), and it never talks to the Placeframe API.

  python remote_worker.py <ticket URL>

Placeframe's own reconstruction code runs unchanged; two stand-ins replace its connections:
  * storage: reads (the capture, earlier checkpoints) come from the ticket's GET URLs; writes go to a local results
    folder, except stage checkpoints, which are uploaded at once so a preempted job can resume elsewhere;
  * progress: phase/progress are written to the ticket's status URL; the lease agent relays them to the API (and
    keeps the lease alive).
At the end the results folder is uploaded as one tar, then the final status (metrics or error).
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from uuid import UUID

WORK = Path("/tmp/pf_remote")
RESULTS = WORK / "results"
STATUS_EVERY_S = 20


def main() -> None:
    ticket = json.loads(_get(sys.argv[1]))
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    status = Status(ticket["put"]["status"], ticket["dispatch"])
    status.update(state="starting", phase="downloading")
    try:
        metrics = run(ticket, status)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(RESULTS, arcname=".")
        _put(ticket["put"]["results"], buf.getvalue())
        status.update(state="succeeded", metrics=metrics, results_bytes=buf.tell(), force=True)
    except Exception as e:  # noqa: BLE001 - report every failure to the broker instead of dying silently
        status.update(state="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:],
                      force=True)
        raise


def run(ticket: dict, status: "Status") -> dict:
    # Settings the reconstructor validates at import; nothing here is used to connect anywhere.
    os.environ.update({"API_INTERNAL_URL": "http://unused.invalid", "AUTH_TOKEN_URL": "http://unused.invalid",
                       "AUTH_CLIENT_ID": "remote", "PRIVATE_KEY_PATH": "/dev/null", "CAPTURES_BUCKET": "captures",
                       "RECONSTRUCTIONS_BUCKET": "reconstructions", "RECONSTRUCTOR_CHECKPOINTS": "1"})
    if ticket.get("mapper_config") is not None:
        cfg = WORK / "mapper.json"
        cfg.write_text(json.dumps(ticket["mapper_config"]))
        os.environ["RECONSTRUCTOR_MAPPER_CONFIG"] = str(cfg)
    os.chdir(WORK)
    import reconstructor.run_reconstruction as rr  # after the environment above: settings and checkpoints read it
    from core.reconstruction_options import ReconstructionOptions

    rr.create_s3_client = lambda **_: PresignedStorage(ticket)
    rr.load_models()
    metrics = rr.run_reconstruction(UUID(ticket["reconstruction_id"]), UUID(ticket["capture_id"]),
                                    ReconstructionOptions.model_validate(ticket["options"] or {}), Publisher(status))
    return metrics.model_dump(mode="json")


class PresignedStorage:
    """The subset of the S3 client the reconstructor uses, backed by the ticket's pre-signed URLs."""

    def __init__(self, ticket: dict) -> None:
        self.rid, self.cid = ticket["reconstruction_id"], ticket["capture_id"]
        self.gets, self.puts = ticket["get"], ticket["put"]

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803 - boto3's argument names
        name = "capture.tar" if Key == f"{self.cid}.tar" else Key.removeprefix(f"{self.rid}/")
        if name not in self.gets:
            raise KeyError(f"no access to {Key}")  # checkpoint loaders treat any error as "no checkpoint"
        return {"Body": io.BytesIO(_get(self.gets[name]))}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> dict:  # noqa: N803
        rel = Key.removeprefix(f"{self.rid}/")
        path = RESULTS / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Body)
        if rel in self.puts:  # stage checkpoints leave the machine immediately
            _put(self.puts[rel], Body)
        return {}


class Publisher:
    """Stand-in for ReconstructionPublisher: phase and progress go to the status object."""

    def __init__(self, status: "Status") -> None:
        self.status = status

    def set_phase(self, status, total: int | None = None) -> None:
        self.status.update(state="running", phase=getattr(status, "value", str(status)), current=None, total=total,
                           force=True)

    def on_progress(self, current: int, attempt: int | None = None) -> None:
        self.status.update(state="running", current=current, attempt=attempt)

    def heartbeat(self) -> None:
        self.status.update(force=True)


class Status:
    def __init__(self, url: str, dispatch: str) -> None:
        self.url, self.data, self.last = url, {"dispatch": dispatch}, 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._beat, daemon=True).start()

    def update(self, force: bool = False, **kw) -> None:
        with self.lock:
            self.data.update(kw)
            self.data["t"] = time.time()
            if not force and time.time() - self.last < STATUS_EVERY_S:
                return
            self.last = time.time()
            body = json.dumps(self.data).encode()
        try:
            _put(self.url, body)
        except Exception as e:  # noqa: BLE001 - a missed status write is retried on the next update
            print(f"[remote] status upload failed: {e}", flush=True)

    def _beat(self) -> None:  # keeps the status fresh during long silent stages
        while True:
            time.sleep(STATUS_EVERY_S)
            self.update(force=True)


def _get(url: str) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _put(url: str, body: bytes) -> None:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=body, method="PUT")
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
            return
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    main()
