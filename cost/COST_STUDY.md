# Cost study v2: building and serving maps for a 10,000 m² conference venue

**For:** the Placeframe team · **Measured:** 2026-11-17 on one RTX 3090 (24 GB) + i7-11700K (16 threads), with
an Insta360 X5 house capture and a 3-recording house map · **Code:** `cost/` (study, serving model, charts)

## Summary

- **Builds are 3–6x faster than in our first study, and now scale linearly.** A 9,072-image map builds in 37 min
  (was 108 min). Projected for a 10,000 m² venue (~19,700 images, ~28 min of walking): **~1.3 h on one RTX 3090**,
  about $0.20–0.40 of rented GPU time; the first study projected 677 h for one connected map (18 h in tiles).
- **Serving cost is set by how often phones query.** For 3,000 attendees walking in breaks, an adaptive query rate
  (every 5 s while fixes agree) needs ~102 RTX 3090s at peak and 211 GPU-hours per event day: **$32/day on
  marketplace GPUs, $150–213 on cloud on-demand GPUs**. A fixed 2 Hz rate needs 8–10x more.
- **Map storage is negligible (~3 GB per venue, cents per month); data transfer is not.** GPUs that autoscale pull
  the map (~1.15 GB) each time one starts: ~900 GB/day here. From S3/GCS/Azure to GPUs in another provider that is
  **$78–108/day in egress, more than the GPUs**; from Cloudflare R2 it is $0. With GPUs in the same cloud as the
  store, egress is ~$1/day.
- **Rented GPUs:** right-sized interruptible RTX 3090s were unreliable tonight (see below). For single builds of
  this size the home GPU is the better choice; renting pays off for parallel builds of different maps.

## 1. Building maps

![build time](1_build_time.png)

| Map | Images | First study | Now | Of which mapping |
|---|---|---|---|---|
| house, first 25% of the walk | 564 | 221 s | 216 s | 110 s |
| house, 50% | 1,134 | 1,567 s | 436 s | 236 s |
| house, full walk | 2,268 | 5,749 s | 1,018 s | 617 s |
| house x2, joined into one map | 4,536 | 2,831 s (separate copies) | 1,024 s | 236 s |
| house x4, joined into one map | 9,072 | 6,480 s (separate copies) | 2,194 s | 477 s |
| real merge: 3 recordings of one house | 8,370 | — | 2,075 s | — |

- What changed: GPU bundle adjustment (Ceres + cuDSS) for all builds, and the prior-pose mapper for multi-recording
  maps (frames start at their capture poses; pose-prior bundle adjustment instead of incremental registration).
- x2/x4 tile the full walk side by side (2 m apart) in one reconstruction with one bundle adjustment, so
  retrieval, pairing and mapping run over the whole joined area. A "virtual bridge" of synthetic matches between
  tiles would need synthetic 3D points projected into frames on both sides; with the prior-pose mapper the tiles
  are already placed in one model by their priors, and the bridge would add a few hundred matches, so it was not
  built. The real 3-recording merge is a genuinely connected large map and lands on the same line.
- **Where the time goes now:** in joined maps, feature extraction (~33%) and matching (~36%) dominate; mapping is
  ~22%. For merges into an existing map, reusing the base map's features and matches would remove most of the
  base images' share (estimate: 30–50% of a merge, depending on how much of the map is new). Not built yet.
- Preprocessing of each X5 recording (calibration, odometry, stitching) runs on the CPU, ~25 min per 14-min
  recording; odometry and stitching now run side by side (~5 min saved), and a second recording's preprocessing
  overlaps the first one's reconstruction.

**Venue projection** (open halls walked in ~6 m lanes at ~1 m/s; images per second of walking as in the house
capture; seconds per image as in the joined and merged maps):

| | Value |
|---|---|
| Walking to capture 10,000 m² | ~28 min (one or more X5 recordings) |
| Map images | ~19,700 |
| Server build, one RTX 3090 | ~1.3 h ($0.20–0.40 at $0.15–0.30/h rented) |
| Preprocessing (CPU) | ~50 min |
| Stored: map (~1.15 GB) + capture kept for rebuilds (~2 GB) | ~3.2 GB |

## 2. Rented, right-sized GPU

We tried to build the x4 map on one interruptible Vast.ai RTX 3090 with a host like ours (≥12 cores, ≥32 GB RAM),
not the cheapest large GPU. 31 matching offers at $0.13–0.16/h minimum bid; the broker bids 1.5x.

| Attempt | Result |
|---|---|
| RTX 3090, China, $0.245/h | 32 min still downloading the 4.3 GB worker environment and the 960 MB capture, then stopped by the host (interruptible: likely outbid) |
| RTX 3090, Czechia, $0.205/h | failed to start (Docker error on the host) |
| further offers | none launchable within the filter at ≤ $0.30/h for the next 20 min |

Spend ~$0.20. It also exposed a bug in our lease heartbeat (an unchanged progress update did not refresh the
lease, so the job was re-leased to the home GPU while the rented one set up), now fixed. Earlier measured runs
(yesterday): a rented worker is GPU-ready ~2 min 20 s after creation on a well-connected host, and an A100 built a
400-image map in ~5 min. **Conclusion:** at ~1.3 h per venue map, a home or reserved GPU is simpler and cheaper;
rent for parallel builds, choose hosts by download speed and reliability as much as by price, and expect
interruptible instances to be lost.

## 3. Serving: 3,000 attendees in a 10,000 m² venue

![load](2_serving_load.png)

**Load model.** An 8 h programme of 45-minute sessions and 15-minute breaks. 5% of attendees localize during
sessions; in breaks 75% walk with the app, ramping up over the first 3 min and down over the last 4. A phone that
starts localizing sends ~10 queries (at 2 Hz) until its place is confirmed, then tracks at the policy's rate. The
adaptive policy's rate (0.19 queries/s while walking) and its accuracy were measured by replaying a recorded
5-minute phone walk: 4x fewer queries than 1 Hz with the same accuracy.

**Throughput** (measured, one RTX 3090, tracked queries through the API): 4.9 q/s with one client (p50 198 ms),
7.4 q/s with four (p50 521 ms); peak GPU memory 2.7 GB, GPU ~60% busy (the CPU side is the limit). We plan with
7 q/s per GPU, autoscaled every 5 minutes with 30% headroom.

![cost](3_serving_cost.png)

| Query policy | Queries per day | Peak q/s | Peak GPUs | GPU-hours/day | Vast RTX 3090 ($0.15/h) | GCP L4 ($0.71/h) | AWS A10G ($1.01/h) |
|---|---|---|---|---|---|---|---|
| fixed 2 Hz | 32.1 M | 4,619 | 858 | 2,030 | $304 | $1,441 | $2,050 |
| fixed 1 Hz | 16.2 M | 2,369 | 440 | 1,026 | $154 | $728 | $1,036 |
| **adaptive** | **3.3 M** | **547** | **102** | **211** | **$32** | **$150** | **$213** |

(L4/A10G throughput is assumed equal to the RTX 3090's; not measured.)

**Storage and transfer** (adaptive policy; the map pulled by every GPU server as it starts, ~775 starts/day; the
app downloads a ~1 MB point cloud per session; query images go straight to the GPUs):

| Store | Storage | Egress/day, GPUs at a marketplace | Egress/day, GPUs in the same cloud |
|---|---|---|---|
| **Cloudflare R2** | $0.05/month | **$0** | $0 |
| AWS S3 Standard | $0.07/month | $81 | $0.79 |
| GCS Standard | $0.06/month | $108 | $1.05 |
| Azure Blob Hot | $0.06/month | $78 | $0.76 |

## 4. Recommendations

1. **Query adaptively** (done in our client): the largest single factor in serving cost.
2. **Keep the map next to the GPUs, or on R2.** Marketplace GPUs plus R2, or cloud GPUs plus the same cloud's store;
   avoid paying egress for every scale-up. Keeping a few GPUs warm through sessions also cuts map loads.
3. **Pack more per GPU:** a tracked query now peaks at 2.7 GB of GPU memory and leaves the GPU 40% idle, so several
   localizer processes per GPU (or smaller, cheaper GPUs) should raise throughput; not measured yet.
4. **Reuse base features and matches in merges** (30–50% of a merge, estimated).
5. **Rent for parallel builds, not single venue builds.**

## Assumptions and caveats

- One venue model; attendee behaviour (active fractions, ramps, sessions per attendee) is assumed, not measured.
- Localization throughput was measured on one map (8,370 images) with one recorded walk; larger maps retrieve over
  more images (linear, small) and match the same number of pairs.
- Prices: Vast.ai interruptible from live offers on 2026-11-17; cloud on-demand list prices; storage and egress from
  provider price pages (R2: no egress fees).
- The x4 build ran while a recording was preprocessed at low priority on the same machine; its extraction time may
  be slightly high.
