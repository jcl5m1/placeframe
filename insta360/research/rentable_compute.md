# Rentable interruptible GPU compute for tiled SfM builds

Research date: 2026-11-15. Sources are cited inline. **[UNVERIFIED]** marks claims I could not confirm in official docs. Treat these as hypotheses and test them with one cheap instance first.

Workload assumptions: each tile is an independent job that takes 5–30 min of GPU+CPU time. Inputs are 50–500 MB of JPEGs and outputs are 20–200 MB. The Docker image is 5–10 GB with CUDA, COLMAP/pycolmap and PyTorch.

---

## 1. Vast.ai mechanics

**CLI / API.** Install the CLI with `pip install vastai` and set your key with `vastai set api-key ...`. The REST base is `https://console.vast.ai/api/v0` with `Authorization: Bearer $VAST_API_KEY` ([API guide](https://docs.vast.ai/api-reference/creating-instances-with-api.md)). Offer search is `POST /bundles/`, create is `PUT /asks/{offer_id}/` (it returns `new_contract` = instance id), and status is `GET /instances/{id}/`.

**Searching offers** ([search-offers ref](https://docs.vast.ai/cli/reference/search-offers)):
- `-i` / `-b` / `--type=bid` shows interruptible pricing. The default type is on-demand.
- Documented fields: `gpu_name` (use underscores, e.g. `RTX_3090`), `num_gpus`, `gpu_ram`, `reliability`, `inet_down` and `inet_up` (Mb/s), `disk_space` (GB), `dph` ($/h), `min_bid`, `cuda_vers`, `verified`, `datacenter`, `geolocation`, `direct_port_count`.
- Sort with `-o 'dph'`, or `-o 'dlperf_usd-'` for descending order. `--raw` returns JSON.
- The default query is `external=false rentable=true verified=true`. Pass `-n` to disable it.

```bash
vastai search offers -i --raw \
  'gpu_name in [RTX_3090,RTX_4090,RTX_A5000] num_gpus=1 reliability>0.98 \
   inet_down>500 inet_up>200 disk_space>=60 cuda_vers>=12.4 direct_port_count>=1' \
  -o 'dph'
```

(The `-i` + `--raw` combination and `in [...]` on gpu_name are built from the documented syntax but not copied from an example **[lightly unverified]**.) Per-host bandwidth prices also exist as offer fields, possibly `inet_up_cost` / `inet_down_cost` **[UNVERIFIED field names]**. Read them from `--raw` output, because bandwidth is billed per byte and varies by host ([pricing](https://docs.vast.ai/pricing)).

**Creating interruptible vs on-demand** ([create-instance ref](https://docs.vast.ai/cli/reference/create-instance)). Add `--bid_price <$/h>` to create an interruptible instance. Leave it out for on-demand. The documented example:

```bash
vastai create instance 384826 --template_hash 661d... --disk 64 --bid_price 0.1
```

Our variant:

```bash
vastai create instance $OFFER --image ghcr.io/ORG/sfm-worker:2026.11 --disk 60 \
  --bid_price 0.20 --label sfm-worker --ssh --direct \
  --env '-e WORKER_POOL=prod -e R2_ENDPOINT=...' \
  --onstart-cmd 'env >> /etc/environment; /opt/worker/run.sh >> /var/log/worker.log 2>&1 &'
```

In the REST body, the bid is the `price` field. Other body fields are `image`, `disk`, `runtype` (`ssh_direct`, `jupyter_direct`, `args`), `onstart`, `env`, `label`, `image_login` and `cancel_unavail` ([API guide](https://docs.vast.ai/api-reference/creating-instances-with-api.md)). Set `cancel_unavail: true` so a create on an offer that just got taken fails fast instead of hanging.

**Outbid / preemption.** Interruptible instances are "paused if outbid or if on-demand requested", and "among interruptible instances, highest bid wins". They "resume automatically when priority returns" ([instance types](https://docs.vast.ai/documentation/instances/choosing/instance-types)). The FAQ says the instance "is stopped (killing running processes)" and may wait a long time to resume ([rental types FAQ](https://docs.vast.ai/guides/reference/faq/rental-types)).
- **Assume no SIGTERM and no grace period.** No Vast doc mentions one **[UNVERIFIED either way]**.
- A paused instance still exists, keeps its disk, and **keeps billing storage**. For our use case, the controller should treat "not running" as lost: destroy the instance and re-queue its lease. Do not wait for it to resume.

**Instance states.** `actual_status` takes the values null (provisioning), `created`, `loading` (image pull), `running`, `stopped`, `frozen`, `exited`, `rebooting`, `unknown` and `offline`. `intended_status` is the state you asked for ([show-instances ref](https://docs.vast.ai/cli/reference/show-instances)). Vast's hello-world guide warns that `exited`, `unknown` and `offline` never reach `running`. Destroy instances in those states.

**Launch modes, onstart and environment** ([docker environment](https://docs.vast.ai/guides/instances/docker-environment)):
- There are three launch modes: SSH, Jupyter and Entrypoint (args). SSH and Jupyter modes **replace the image entrypoint** with Vast's setup scripts, so start the worker from onstart. `--onstart-cmd` is limited to 16 KB ([CLI get-started](https://docs.vast.ai/cli/get-started)).
- Predefined environment variables: `CONTAINER_ID`, `CONTAINER_API_KEY` (a per-instance key that can stop or destroy *this* instance), `PUBLIC_IPADDR`, `GPU_COUNT`, `VAST_CONTAINERLABEL`, `VAST_TCP_PORT_<n>` and `VAST_UDP_PORT_<n>`.
- SSH sessions don't see custom environment variables unless onstart runs `env >> /etc/environment`.
- A worker can **self-destroy** with `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` ([instances FAQ](https://docs.vast.ai/guides/reference/faq/instances)). The FAQ documents `stop` with this key. That the key also covers `destroy` comes from a secondary source **[verify]**.
- Docker-in-Docker is not supported. Disk size is fixed at creation. CPU and RAM are guaranteed only in proportion to your GPU share. Up to 64 ports.

**Time to running.** Vast publishes no number. Expect roughly 1–3 min when the image layers are cached on the host, and 5–15+ min to pull 5–10 GB on a slow host **[UNVERIFIED, community experience]**. Measure this yourself and filter on `inet_down`.

**Billing** ([pricing](https://docs.vast.ai/pricing)):
- GPU compute is billed **per second** while running.
- Storage is billed for as long as the instance exists, **including when stopped or paused**, and stopped-state storage can cost more.
- Upload and download bandwidth is billed per byte at host-specific rates.
- `destroy` deletes all data and stops all billing. `stop` keeps the disk and keeps billing it.

## 2. RunPod (for comparison): spot is effectively gone from the API

- **Change this month.** REST v1 (`https://rest.runpod.io/v1`) "is deprecated and will be retired on November 15, 2026", which is today.
- **No spot in v2.** The v2 migration table lists **"Spot Pods | `interruptible`, `podRentInterruptable`, `podBidResume` | None."** ([migrate from v1](https://docs.runpod.io/api-reference-v2/migrate-from-v1)). The v2 API at `https://api.runpod.io/v2` has no spot pods.
- The pricing page now lists only on-demand pods (savings plans are deprecated) ([pod pricing](https://docs.runpod.io/pods/pricing)). Whether the console still offers spot is **[UNVERIFIED]**.
- Older docs described spot pods as getting a 5-second SIGTERM then SIGKILL **[from a docs PR diff, not current]**.
- What still holds: per-second billing for compute and storage, **no ingress or egress fees**, and a stopped volume disk costs $0.20/GB/mo.

**Conclusion:** use RunPod only as an *on-demand* overflow pool, not as an interruptible pool.

## 3. Docker image strategy

- **Base image.** Vast hosts "cache commonly-used Docker image layers". Vast recommends building on popular bases so "most of the image content is already present on host machines" ([vast-ai/base-image](https://github.com/vast-ai/base-image)). Two choices:
  - Base on `vastai/base-image:cuda-12.x-cudnn-devel-ubuntu22.04` (or the matching `nvidia/cuda` tag). This is the most likely cache hit.
  - Base on an official `pytorch/pytorch:*-cuda*-runtime` tag. This is likely popular too **[UNVERIFIED cache]**.
- **Layer order:**
  1. CUDA/cuDNN base (cached).
  2. apt deps.
  3. torch/torchvision, pinned to a version that matches the base CUDA.
  4. COLMAP/pycolmap build or wheel.
  5. Other pip deps.
  6. Model weights (LightGlue/SuperPoint/DISK and similar, only tens to hundreds of MB).
  7. Your code, last and tiny.

  This way a code change re-pulls only the last few MB. Use a separate `-runtime` final stage and avoid `-devel` unless COLMAP needs nvcc at runtime.
- **Registry choice:**
  - **GHCR** is free for public images and has no pull rate limit for authenticated or public pulls in practice **[verify current limits]**.
  - **Docker Hub** applies anonymous pull rate limits per IP, and many Vast hosts are shared IPs. If you use it, authenticate via `--login`.
  - **GCP Artifact Registry** charges internet egress (~$0.12/GB, see §4) for every pull of a 5–10 GB image. At that size, every new worker costs about $1 just to pull, so avoid it.
  - Recommendation: GHCR, with a private image and a read-only PAT passed via `--login '-u USER -p TOKEN ghcr.io'`.
- **Weights.** Bake small feature-model weights into the image. Pull at start only things that change per build, such as vocab trees and config.
- **Templates.** A Vast template (`--template_hash`) just stores image, env, onstart and launch mode. It does not pre-warm hosts. It is a convenience, not a speedup.
- **Host affinity.** Prefer hosts you have used before (record `machine_id`). They will already hold your layers.

## 4. Data movement

**Throughput.** In practice the bottleneck is your home uplink and the host's `inet_down`, not the storage service. At 100 Mb/s, a 500 MB tile takes about 40 s, which is fine against 5–30 min of compute. The home uplink caps how many workers home-direct can feed at once **[estimate]**.

| Option | Cost | Notes |
|---|---|---|
| **Cloudflare R2** | $0.015/GB-mo, Class A $4.50/M, Class B $0.36/M, **egress free**. Free tier: 10 GB-mo, 1M A, 10M B ([R2 pricing](https://developers.cloudflare.com/r2/pricing/)) | S3 API (`rclone`, `aws s3 --endpoint-url`, boto3). Presigned URLs work. A 100-tile build is about 40 GB of transient data, which costs cents. Decouples workers from home uptime and uplink. |
| **GCS** | Internet egress ~$0.12/GB for the first TiB (NA) **[from secondary sources; official page not parsed]**. Storage ~$0.02/GB-mo | Signed URLs (V4) are good for scoped access. Egress fees make it strictly worse than R2 here. |
| **HTTPS from home** (Caddy/nginx behind Cloudflare Tunnel, or open port) | $0 | Simple. Every worker hits your uplink. Large-file serving through Cloudflare's proxy may conflict with Cloudflare's terms **[UNVERIFIED for Tunnel; check ToS]**. |
| **Tailscale userspace** | $0 | Works without privileges or `/dev/net/tun` ([userspace networking](https://tailscale.com/docs/concepts/userspace-networking)). The container is not a full tailnet peer: only processes that use the proxy can reach tailnet IPs. Use an **ephemeral, tagged, pre-authorized auth key** with ACLs limiting the tag to the file server. Whether Vast containers block this is **[UNVERIFIED; test once]**. |

Commands for the Tailscale userspace option:

```bash
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 \
  --outbound-http-proxy-listen=localhost:1055 &
tailscale up --auth-key=$TS_AUTHKEY --hostname=sfm-$CONTAINER_ID
ALL_PROXY=socks5://localhost:1055/ HTTP_PROXY=http://localhost:1055/ worker ...
```

**Recommendation:**
- **Primary: R2.**
  - The home server uploads each tile's inputs once as a single tar or zstd archive. Use few large objects to save on operations.
  - Workers download inputs and upload checkpoints and outputs to R2.
  - The home server pulls the results.
  - Give workers a bucket-scoped R2 API token, or better, per-task presigned GET/PUT URLs minted by the queue server. Then workers hold no long-lived credentials.
- **Fallback: direct HTTP pull from the home server over Tailscale userspace mode** (ephemeral key). Use it if R2 is down, or for small and debug runs.

## 5. Work queue and checkpoint pattern

**Queue server.** Run it on the home server and expose it via Tailscale and Cloudflare Tunnel. A small HTTP service with a SQLite or Postgres table is enough at this scale (hundreds of tasks). Redis Streams or Postgres `SELECT ... FOR UPDATE SKIP LOCKED` would also work.

- **Lease with expiry (visibility timeout).**
  - `POST /lease` returns `{task_id, attempt, lease_token, expires_at, input_urls, output_urls}`.
  - The worker heartbeats every 30–60 s (`POST /heartbeat`), which extends the lease to about 3 min.
  - The reaper puts expired leases back on the queue.
  - Because Vast gives no preemption signal, **a missed heartbeat *is* the preemption detector**. The controller cross-checks `actual_status` from `vastai show instances --raw`.
- **Idempotent tasks.**
  - Outputs go to `results/{build}/{tile}/attempt-{n}/`, then the worker writes a `_SUCCESS` manifest containing hashes.
  - `POST /complete` with the lease token is accepted only if the token is current. Duplicate completions are ignored.
  - The reducer reads only the manifests.
- **Checkpoint chunks inside a tile.** Split each tile into stages, for example: (a) feature extraction, (b) matching, (c) mapping/BA.
  - Each stage writes its artifacts (`features.h5`, `matches.h5`, `database.db`, `sparse/`) to `ckpt/{build}/{tile}/{stage}.tar.zst`.
  - A new worker skips any stage whose checkpoint already exists.
  - With 5–30 min tiles, stage-level checkpoints keep the loss per preemption to a few minutes. Don't checkpoint more finely.
- **Retries.** Use `max_attempts` (e.g. 4). Send poison tasks (crash, not preemption) to a dead-letter state. Distinguish the two causes by whether the worker reported an error versus silently vanished.
- **Worker lifecycle.** onstart runs a loop of lease, run, complete. If the queue stays empty for N minutes (e.g. 3), the worker self-destroys via `CONTAINER_API_KEY`. That way idle instances never accrue storage or compute charges.
- **Autoscaler** (home cron, every 1–2 min):
  - Target = min(`pending_tasks / tasks_per_worker`, `MAX_INSTANCES`).
  - Search offers, create up to the deficit with `--bid_price` = min(`MAX_BID`, offer `min_bid` × 1.2), and label everything `sfm-worker`.
  - Destroy instances that are `exited` / `unknown` / `offline`, stopped or paused (outbid), stuck in `loading` longer than 15 min, or that haven't heartbeated in 5 min.
- **Cost guardrails:**
  - `MAX_BID` $/h per instance and `MAX_INSTANCES`.
  - A per-build budget computed from per-second spend (`vastai show instances --raw` cost fields plus bandwidth), with a hard stop that destroys everything labelled `sfm-worker`.
  - Keep a modest Vast credit balance (prepaid) as the ultimate cap.
  - Run a nightly sweep that destroys any labelled instance older than X hours.

## 6. Existing tools

- **SkyPilot** supports Vast with `use_spot: true` and `bid_price` via `create_instance_kwargs` ([Vast article, Mar 2026](https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot)). `pip install "skypilot[vast]"`.
  - Managed jobs (`sky jobs launch`) auto-recover from preemption and can fall back from spot to on-demand via `any_of`. `--num-jobs N` exposes `$SKYPILOT_JOB_RANK`, and Pools keep warm workers ([managed jobs](https://docs.skypilot.ai/en/latest/examples/managed-jobs.html)).
  - Limits on Vast: no object-store mounting, no multi-node, ports fixed at launch. You still write checkpoint/R2 logic yourself.
  - Recovery restarts the whole job on a fresh instance, which means a fresh image pull. That is acceptable for 5–30 min tiles.
  - SkyPilot added RunPod spot in v0.8.0, but given RunPod's API change (§2) that path is likely broken now **[UNVERIFIED]**.
- **dstack** has native RunPod and Vast backends, `spot_policy: spot|auto|on-demand`, and Vast spot since 0.20.28 ([dstack backends](https://dstack.ai/docs/concepts/backends/)). Container backends don't support some features (e.g. `idle_duration`) ([protips](https://dstack.ai/docs/guides/protips/)).
- **Verdict.** Both tools solve offer search, provisioning, preemption detection and relaunch. Neither gives you a per-tile work queue with stage checkpoints, which is the part that actually matters here.
  - With one provider and ~100 short tasks, a ~300-line custom controller (Vast CLI + lease queue + R2) is simple and transparent.
  - **SkyPilot managed jobs (one job per tile, or `--num-jobs` with a rank-to-tile map) are a reasonable shortcut** if you'd rather not write the autoscaler. Prototype both on one small build.

### dstack pilot (2026-11-17, measured)

We ran the serving benchmark (`bench/localizer_profile.py`) as a dstack task on Vast (`fleet/dstack_bench.py`;
dstack 0.22.8, local server, Vast backend; config in `~/.dstack/server/config.yml`, mode 600).

| | dstack | Our launcher (`fleet/remote_bench.py`, `broker.py`) |
|---|---|---|
| Set-up | `uv tool install "dstack[server]"`, a server process, the Vast API key in its config | Vast CLI only |
| Before the first run | An elastic fleet is necessary (`nodes: 0..1`); without one the run waits with a hint | Nothing |
| Offer filters | GPU, CPU count, RAM, disk, price, spot policy, regions | Any Vast field: also CPU speed (`cpu_ghz`), reliability, download speed, country |
| Bid | dstack's spot price (top offer: $0.43/h at a $0.24 minimum bid); this run paid $0.277/h | 1.5x the minimum bid |
| Submit to running | 2.8 min (1 min waiting for the fleet, 1.4 min provisioning) | ~2.3 min |
| Retry | `retry: on_events [interruption, no-capacity, error]` (configured; no interruption happened) | Own code: relaunch, resume from R2 checkpoints |
| End of run | Instance destroyed by itself (`all_jobs_done`); cost recorded ($0.023 for 6.6 min) | Own code destroys it; we estimate the cost |

Result on the host it chose (RTX 3090, Ryzen 5 5600): 9.6 q/s at 2 processes x 8 sessions, as at home. That host
was a fast CPU by chance: dstack cannot ask for CPU speed, and on a slow-CPU host the same GPU gives half that.

**Verdict.** dstack can replace our launch, watch and destroy code (`broker._launch`/`_watch`, `remote_bench.py`,
`reattach.py`) and adds cost records, logs and more backends. Keep the lease agent, the pre-signed R2 tickets and the
stage checkpoints: dstack does not supply them. Before we change over: (1) find a way to require a fast CPU (a check
at task start that exits on a slow host, then let dstack's retry pick another offer), and (2) test a real
interruption. SkyPilot was not tested.

## Recommended architecture

1. **Single provider: Vast.ai interruptible** (`--bid_price`), 1× RTX 3090/4090/A5000-class, filtered on `reliability>0.98 inet_down>500 inet_up>200 disk_space>=60 cuda_vers>=<your CUDA>`, sorted by `dph`. RunPod on-demand only as an optional overflow.
2. **Image on GHCR**, built `FROM` a popular CUDA base (`vastai/base-image` or `nvidia/cuda`). Heavy layers first, code last, weights baked in, launched in SSH mode with the worker started from `--onstart-cmd`.
3. **R2 as the data plane:** `inputs/{build}/{tile}.tar.zst`, `ckpt/{build}/{tile}/{stage}.tar.zst`, `results/{build}/{tile}/attempt-n/` + `_SUCCESS`. Workers get presigned URLs per lease.
4. **Fallback data path:** home HTTP file server reached over Tailscale userspace networking with an ephemeral, tag-restricted auth key.
5. **Queue server on the home server** (SQLite/Postgres) with lease + heartbeat (60 s interval, 3 min expiry), attempt counters, dead-letter state, and token-checked idempotent completion.
6. **Stage checkpoints within each tile** (features → matches → mapping). Resume by skipping completed stages. Assume preemption gives zero warning.
7. **Autoscaler cron on the home server:** scales to queue depth under `MAX_INSTANCES`/`MAX_BID`. It destroys (never stops) paused/outbid, `exited`, `offline`, stuck-`loading` or silent instances, because stopped instances still bill storage.
8. **Workers self-destroy** via `CONTAINER_API_KEY` after N idle minutes. A nightly sweep destroys any `sfm-worker`-labelled instance older than a cap.
9. **Budget guardrails:** per-build $ cap tracked from instance cost plus bandwidth, kill-switch destroys all labelled instances, prepaid credit kept small.
10. **Reduce step runs at home** (merge tile models / global BA on the 3090), pulling only `_SUCCESS` results from R2. Then expire R2 prefixes with a lifecycle rule.

**Validate first with one cheap instance:**
- Image pull time on cached vs uncached hosts.
- Whether Tailscale userspace works on Vast.
- What `actual_status` an outbid instance reports.
- Whether any signal reaches the container on outbid.
- Real R2 throughput from a few hosts.
