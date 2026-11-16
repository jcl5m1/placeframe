#!/usr/bin/env python3
"""Fleet lease agent: the only fleet component that holds Placeframe credentials. Runs on the home server inside the
worker image (Placeframe's network, worker key mounted read-only). It never ships credentials anywhere.

For each job it leases from the Placeframe API (while ~/pf_fleet/broker/capacity allows):
  1. stages the capture from MinIO to R2 (jobs/<reconstruction id>/capture.tar) and announces the job to the broker
     (broker/jobs/<id>.json); the broker issues pre-signed URLs and runs it on a rented GPU;
  2. keeps the lease alive, relaying the remote status (jobs/<id>/status.json in R2) to the API every 30 s;
  3. on success copies the results tar from R2 into MinIO under the reconstruction and reports the metrics; on
     failure (or when the broker gives up) fails the lease with the error.
Every R2/MinIO transfer is appended to ~/pf_fleet/access_log.jsonl.

  python lease_agent.py          (in the worker image; see fleet/README.md)
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import tarfile
import time
from pathlib import Path

from common.boto_clients import create_s3_client
from common.token_manager import TokenManager
from placeframe_api_client import (
    ApiClient,
    Configuration,
    DefaultApi,
    ProgressUpdate,
    ReconstructionStatus,
)
from placeframe_api_client import ReconstructionMetrics as ClientReconstructionMetrics
from placeframe_api_client.exceptions import ApiException
from reconstructor.settings import get_settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2  # noqa: E402

FLEET = Path.home() / "pf_fleet"
BROKER = FLEET / "broker"
LOG = FLEET / "access_log.jsonl"
POLL_S, RELAY_S = 10, 30


def log(event: str, **kw) -> None:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "by": "lease_agent", "event": event, **kw}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


async def main() -> None:
    settings = get_settings()
    minio = create_s3_client(minio_endpoint_url=settings.minio_endpoint_url, minio_access_key=settings.minio_access_key,
                             minio_secret_key=settings.minio_secret_key)
    r2s3, bucket = r2.client()
    auth = TokenManager(str(settings.auth_token_url), settings.auth_client_id, Path(settings.private_key_path))
    configuration = Configuration(host=str(settings.api_internal_url))
    (BROKER / "jobs").mkdir(parents=True, exist_ok=True)
    active: dict[str, asyncio.Task] = {}
    async with ApiClient(configuration) as client:
        api = DefaultApi(client)
        print("[lease_agent] started", flush=True)
        while True:
            token = await auth.get_token()
            configuration.access_token = token
            client.default_headers["Authorization"] = f"Bearer {token}"
            for rid in [r for r, t in active.items() if t.done()]:
                active.pop(rid)
            if len(active) < _capacity():
                try:
                    lease = await api.request_lease()
                except ApiException as e:
                    if e.status != 404:
                        print(f"[lease_agent] lease request failed: {e.status}", flush=True)
                    lease = None
                if lease is not None:
                    rid = str(lease.reconstruction_id)
                    active[rid] = asyncio.create_task(_run_job(api, minio, r2s3, bucket, settings, lease))
            await asyncio.sleep(POLL_S)


def _capacity() -> int:
    try:
        return int((BROKER / "capacity").read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


async def _run_job(api, minio, r2s3, bucket, settings, lease) -> None:
    rid, cid = str(lease.reconstruction_id), str(lease.capture_session_id)
    prefix = f"jobs/{rid}"
    try:
        await _progress(api, rid, {"phase": "downloading"})
        body = await asyncio.to_thread(lambda: minio.get_object(Bucket=settings.captures_bucket, Key=f"{cid}.tar")["Body"].read())
        await asyncio.to_thread(r2s3.put_object, Bucket=bucket, Key=f"{prefix}/capture.tar", Body=body)
        log("staged_capture", reconstruction=rid, capture=cid, r2_key=f"{prefix}/capture.tar", bytes=len(body))
        mapper_cfg = Path("/etc/placeframe/mapper.json")
        job = {"reconstruction_id": rid, "capture_id": cid, "options": lease.options.model_dump(mode="json"),
               "mapper_config": json.loads(mapper_cfg.read_text()) if mapper_cfg.exists() else None,
               "r2_prefix": prefix, "leased": time.time()}
        (BROKER / "jobs" / f"{rid}.json").write_text(json.dumps(job, indent=1))
        while True:
            await asyncio.sleep(RELAY_S)
            st = await asyncio.to_thread(_r2_json, r2s3, bucket, f"{prefix}/status.json")
            outcome = _broker_outcome(rid)
            if st and st.get("state") == "succeeded":
                await _collect(api, minio, r2s3, bucket, settings, rid, prefix, st)
                return
            if st and st.get("state") == "failed":
                raise RuntimeError(f"remote worker failed: {st.get('error')}")
            if outcome == "abandoned":
                raise RuntimeError("fleet gave up on this job (see broker log)")
            await _progress(api, rid, st or {"phase": "downloading"})
    except Exception as e:  # noqa: BLE001 - every failure must reach the API, or the job looks stuck
        log("job_failed", reconstruction=rid, error=str(e)[:500])
        try:
            await api.fail_lease(lease.reconstruction_id, str(e)[:2000])
        except ApiException as api_error:
            print(f"[lease_agent] fail_lease failed: {api_error.status}", flush=True)
    finally:
        (BROKER / "jobs" / f"{rid}.json").unlink(missing_ok=True)
        (BROKER / "done").mkdir(exist_ok=True)
        (BROKER / "done" / rid).write_text(str(time.time()))


async def _collect(api, minio, r2s3, bucket, settings, rid, prefix, st) -> None:
    data = await asyncio.to_thread(lambda: r2s3.get_object(Bucket=bucket, Key=f"{prefix}/results.tar")["Body"].read())
    log("collected_results", reconstruction=rid, r2_key=f"{prefix}/results.tar", bytes=len(data))
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            key = f"{rid}/{m.name.removeprefix('./')}"
            await asyncio.to_thread(minio.put_object, Bucket=settings.reconstructions_bucket, Key=key,
                                    Body=tar.extractfile(m).read())
            n += 1
    log("stored_results", reconstruction=rid, minio_bucket=settings.reconstructions_bucket, files=n)
    await api.succeed_lease(rid, ClientReconstructionMetrics.model_validate(st["metrics"]))
    log("job_succeeded", reconstruction=rid)


async def _progress(api, rid: str, st: dict) -> None:
    try:
        phase = ReconstructionStatus(st.get("phase") or "downloading")
    except ValueError:
        phase = ReconstructionStatus("downloading")
    await api.update_progress(rid, ProgressUpdate(status=phase, progress_current=st.get("current"),
                                                  progress_total=st.get("total"), progress_attempt=st.get("attempt")))


def _r2_json(r2s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(r2s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:  # noqa: BLE001 - not written yet
        return None


def _broker_outcome(rid: str) -> str | None:
    p = BROKER / "outcome" / rid
    return p.read_text().strip() if p.exists() else None


if __name__ == "__main__":
    asyncio.run(main())
