# Fleet: tiled map builds on many (interruptible) workers

## Approach

This is partition-and-parallelize, not classic map-reduce, built on what Placeframe already has:

- **Partition.** `tiler.py` splits a posed walk into balanced, overlapping tiles: recursive bisection of frame
  positions, plus every frame within `--overlap` metres of a core frame. All tiles keep the capture's world frame
  (the VIO, ARKit or map priors), so tile maps line up with no merge step.
- **Map.** Each tile is an ordinary Placeframe reconstruction job. The API's lease queue is already a
  worker-pull job queue (Postgres-backed, with a reaper for dead workers), so any number of workers can serve it:
  the home reconstructor, extra local containers, or rented GPUs.
- **Reduce (optional).** The localizer already accepts a list of map ids, so a client localizes against the tiles
  around it and no global merge is needed. A global consistency pass across tile overlaps is a possible later step.

Why not a classic map-reduce framework? SfM jobs are minutes-long, GPU-bound and stateful, and the API already
provides at-most-once leases. What was missing was preemption tolerance and scale-out, added here:

| Piece | Where | Status |
|---|---|---|
| Lease heartbeat every 60 s during long silent stages | `reconstructor/progress_publisher.py`, `main.py` | deployed |
| Lease timeout configurable (`LEASE_TIMEOUT_MINUTES`, now 10); timed-out jobs re-queued (`LEASE_REQUEUE=1`); reap committed in its own transaction (upstream rolled it back when the queue was empty) | `api/routers/leases.py`, `compose.local.yml` | deployed |
| Stage checkpoints (features, matches) to object storage; resume on re-lease | `reconstructor/checkpoints.py`, `run_reconstruction.py` (on with `RECONSTRUCTOR_CHECKPOINTS=1`) | deployed, off on the home worker |
| GPU bundle adjustment (pycolmap + Ceres CUDA/cuDSS) | `gpu/Dockerfile.pycolmap-cuda` | deployed on the home worker |
| Worker image (upstream image + CUDA pycolmap + patches) | `gpu/Dockerfile.worker`, `build_worker_image.py` | built locally (`placeframe-worker:local`, 16 GB) |
| Tiler / submit / wait / publish | `tiler.py`, `build.py` | tested on the house |
| Autoscaling controller: local + Vast.ai, chaos mode | `controller.py` | local: chaos-tested (4 kills of busy workers, all tiles finished, killed jobs re-queued and resumed from checkpoints); Vast dry run only |

## Running

```bash
python fleet/tiler.py capture.tar tiles/ --images-per-tile 750 --overlap 3
python fleet/build.py submit tiles/ --name my-venue          # one job per tile on the queue
python fleet/controller.py --provider local --max-workers 2 --chaos 6 --until-empty   # extra local workers
python fleet/build.py wait tiles/ && python fleet/build.py publish tiles/
python datasets/eval_queries.py queries.jsonl --map-id <tile 1> --map-id <tile 2> ... --camera conversion.json
```

## Before renting interruptible GPUs (needs decisions and credentials)

1. **Storage reachable from the internet.** Workers download capture tars and upload results and checkpoints.
   MinIO is internal-only. Options:
   - **Cloudflare R2** (recommended in `research/rentable_compute.md`): S3 API, no egress fees,
     $0.015/GB-month. Point the stack's S3 settings, or only the fleet's, at an R2 bucket.
   - **MinIO over Tailscale:** each instance runs tailscale in userspace mode with an ephemeral, tagged key.
     It's free, but it's untested on Vast (no privileges) and the home uplink limits throughput.
2. **A registry for the worker image:** GHCR, private, since the image includes model weights.
3. **Vast.ai:** `VAST_API_KEY`, plus `~/pf_fleet/vast.env` with `WORKER_IMAGE`, the public `API_INTERNAL_URL` and
   `AUTH_TOKEN_URL`, `AUTH_CLIENT_ID`, the storage endpoint and keys, and `WORKER_PRIVATE_KEY_B64` (the worker's
   signing key). Keep it `chmod 600`.
4. **First live test:** one instance, a $2 budget. Measure the 16 GB image pull time, behaviour when outbid,
   checkpoint resume and throughput, then scale out.

`controller.py --provider vast` without `--live` prints what it would do and never rents anything.

## Measured: provisioning a rented worker (2026-11-16)

One interruptible RTX 3090 on Vast.ai (Quebec, bid $0.152/h), stock `ubuntu:24.04` image, no registry:

| Step | Time |
|---|---|
| `create instance` to container running | ~24 s |
| `apt-get install curl zstd` | 10 s |
| Worker environment from R2 (4.3 GB zstd: interpreter, venv, model weights, 16 system libraries) download + decompress + extract | 103 s |
| Imports + GPU check (torch CUDA, pycolmap `has_cuda`, reconstructor modules) | 3 s |
| **Total, create to GPU-ready worker** | **~2 min 20 s** |

- The payload is built from the worker image (`placeframe-worker:local`) and fetched with a pre-signed GET URL, so no
  R2 keys go to the rented host (Vast hosts can see instance environment variables).
- Offers go stale quickly: 3 of the 4 cheapest returned `no_such_ask`. The launcher must walk down the offer list.
- `vastai destroy instance` asks for confirmation; pipe `y` into it in scripts.
- Home to R2: upload 39 Mbit/s (home uplink), download 676 Mbit/s.
