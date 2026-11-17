# Placeframe: findings, fixes and extensions from a self-hosted deployment

**To:** the Placeframe team · **Status:** for your review and feedback; nothing submitted yet

We run Placeframe self-hosted on one RTX 3090 workstation, mapping homes with an Insta360 X5 (360° video) and
localizing phones against those maps. Our fork is based on `main` (May 2026); we have since compared it with
`dev`, `dev_hh` and your open branches (Nov 17) so that we only propose what you don't already have.

## Already on your branches (we will drop ours)

| Our change | Yours |
|---|---|
| Stale-lease reap committed in its own transaction | `dev` lease server (`87790560`) |
| Reconstructor token refresh during long jobs | moot on `dev` (internal lease server, no auth) |
| Reconstructed frames written in capture order | `dev_hh` `c051b46c` (also writes frame ids) |
| Block `/internal` on the public endpoint | superseded by the internal lease server |
| Empty the localizer's CUDA cache; LightGlue in small batches | `oom-fix` `aca42dfd`, `dev_hh` `3f09dd9b` |
| Pass the localizer's 422 reason to clients | `dev_hh` `0b6a814e` |

## Fixes not on your branches

1. **Restart the reconstructor process after each job.** The real cause of our out-of-memory kills: a long-lived
   worker kept ~8 GB of host memory after each large build, so the next 7,000-image build swapped for an hour and
   was killed (the build itself peaks at 11 GB). `execv` after each job; opt out with an env var.
2. **Limit progress output.** One build printed ~209,000 log lines (170,000 from geometry verification); on `dev`
   the publisher still prints every tick. Print at the API flush rate and every 5%.

## Extensions: measurements and questions

1. **Fast prior-pose mapper.** Start frames at their capture poses and refine with pose-prior bundle adjustment
   instead of incremental mapping. Merging a recording into a ~2,000-image map: ~40 min instead of 4–6+ h; a full
   8,370-image, three-recording house map builds in 35 min. `dev_hh`'s merge flow still maps incrementally.
2. **Merge registration must follow heading drift.** A new recording registered to the map with one rigid
   transform (as `dev_hh`'s align step does with yaw/scale/translation) left part of our house as a second copy
   rotated 32°: the X5's odometry heading drifted 32° over a 14-minute walk. Phones then localized confidently on
   the wrong copy. We now fit the alignment piecewise along the recording (rotation and position, from
   localizations every ~5 s) and check the result: the share of the new recording's map points also seen by the
   rest of the map went from 15% to 70%, and confidently wrong phone fixes on a recorded walk from 31 to 6 of 286.
   A merge sharing under 30% is not allowed to replace the old map.
3. **GPU bundle adjustment** (Ceres + cuDSS). Your `AGENTS.md` measured no gain at 318 cameras. At our sizes it
   matters: one global BA on a ~1,250-frame map takes 29 s instead of 293 s; our merges are 7,000–8,400 images.
4. **Localization of a moving phone (a different design from yours, for discussion).** Your design docs put the
   multi-hypothesis filter on the client. We built it on the server (`POST /localize/track`): the client sends
   each frame with its ARCore/ARKit odometry pose and a session id; the localizer produces candidate poses per
   place (retrieved images grouped by location, plus images near the predicted pose, with no confidence gate) and a
   per-session tracker keeps every candidate alignment between the map and the odometry frame, merges duplicates
   (0.5 m / 10°, which addresses the duplicate-hypothesis problem in your false-lock diagnosis), and returns the most
   probable one with its probability. It also re-anchors after the odometry's own jumps (ARCore moved one walk
   5 m in a second). On a 286-frame recorded walk: the served pose is within 0.5 m on 274–278 frames and wrong by
   more than 1 m on 6, all in one stretch with no usable views. The client stays thin and needs no localization
   code. We would value your view on server versus client placement; our evaluation tools would apply to either.
5. **Evaluation without ground truth.** Recorded client sessions (image, camera, odometry) replay through the
   localizer; a reference alignment comes from the odometry's local consistency (correct matches in a short window
   imply the same map alignment), split at odometry jumps. This is how the numbers above were measured, and how
   we chose the settings below.
6. **Phone queries: smaller and fewer.** Phone frames are 480–1080 px; upscaling a 640×480 frame to 1,024 px
   (`LOCAL_FEATURE_RESIZE_SHORTER_SIDE`) cost accuracy: at 640 the right pose was found 6% more often in a third
   less time (we made the size a parameter; the default is unchanged). An adaptive query rate (every 5 s while
   fixes agree, every frame after a disagreement or an odometry jump) sent 4× fewer queries than 1 Hz with the same
   accuracy; for a 3,000-attendee venue it is the largest serving-cost factor.
7. **Portrait images.** Android phones held upright deliver sideways camera images; without the right
   `orientation` our matches fell from confidence 0.75 to 0.07. Your Unity provider hard-codes `LEFT_TOP` (a
   transpose) on Android: is that intended for ARFoundation's image layout, or worth checking for portrait use?
8. **Preemptible workers:** lease heartbeat, configurable lease timeout, and stage checkpoints (features,
   matches) in object storage so a re-leased job resumes. We would adopt your requeue-count cap.
9. **Map placement from capture GPS** (`location.json`, placed in ENU at the fix) and **server-side recording of
   localization requests** for replay (complements your client-side recorder).

A cost study (building and serving maps for a 10,000 m² venue with 3,000 attendees, measured on this pipeline) is in
our pipeline repository: `cost/COST_STUDY.md`. In short: a venue map builds in ~1.3 h on one RTX 3090, and serving
cost is dominated by how often phones query (adaptive querying: ~$28 per event day on marketplace GPUs) and by
map egress when GPUs and storage sit in different clouds.

Outside Placeframe (separate repository; happy to share): the Insta360 X5 pipeline (calibration, odometry,
merging, GPS), a credential-free GPU fleet on rented interruptible GPUs (pre-signed object-storage URLs), a
Flutter client for Android/iOS without Unity, a cost study (build compute, serving load, storage) and operations
tooling (dashboard, watchdog, backups, nightly localization regression checks).

## Where to look

Everything is on our fork's [`local/main`](https://github.com/jcl5m1/placeframe/tree/local/main) branch (based on your
`main`; [full comparison](https://github.com/outernet-foundation/placeframe/compare/main...jcl5m1:placeframe:local/main)),
one commit per change so each can be reviewed, picked or dropped on its own. Nothing will be opened as a pull
request until you tell us which parts you want, and against which branch.

| Item | Commits |
|---|---|
| Fix 1: restart the reconstructor after each job | [`64f0c372`](https://github.com/jcl5m1/placeframe/commit/64f0c372) |
| Fix 2: limit progress output | [`d173cc56`](https://github.com/jcl5m1/placeframe/commit/d173cc56), [`4f217cdd`](https://github.com/jcl5m1/placeframe/commit/4f217cdd) |
| Extension 1: prior-pose mapper | [`356939cc`](https://github.com/jcl5m1/placeframe/commit/356939cc) |
| Extension 2: merge registration and check | in our pipeline repository ([`a488978`](https://github.com/jcl5m1/placeframe-insta360/commit/a488978)) |
| Extension 3: GPU bundle adjustment | worker image build in our pipeline repository (`gpu/`), no Placeframe code change |
| Extension 4: tracked localization | [`549d0ef4`](https://github.com/jcl5m1/placeframe/commit/549d0ef4) (API, localizer, tracker, tests), worker threads [`f5251e25`](https://github.com/jcl5m1/placeframe/commit/f5251e25) |
| Extension 6: query size parameter | [`683136a4`](https://github.com/jcl5m1/placeframe/commit/683136a4); adaptive rate in the app [`46d0838f`](https://github.com/jcl5m1/placeframe/commit/46d0838f) |
| Extension 7: upright phone images | [`4b606387`](https://github.com/jcl5m1/placeframe/commit/4b606387) |
| Extension 8: preemptible workers | [`4d7a7673`](https://github.com/jcl5m1/placeframe/commit/4d7a7673), [`2ba41ebf`](https://github.com/jcl5m1/placeframe/commit/2ba41ebf), heartbeat fix [`f75eeac3`](https://github.com/jcl5m1/placeframe/commit/f75eeac3) |
| Extension 9: GPS placement; request recording | [`ddc88e58`](https://github.com/jcl5m1/placeframe/commit/ddc88e58), [`1fb9ed32`](https://github.com/jcl5m1/placeframe/commit/1fb9ed32); [`50621a68`](https://github.com/jcl5m1/placeframe/commit/50621a68) |
| Panoramic rig pairing | [`ba13a599`](https://github.com/jcl5m1/placeframe/commit/ba13a599) |
| Flutter client | [`5cb59995`](https://github.com/jcl5m1/placeframe/commit/5cb59995), [`b94ec1eb`](https://github.com/jcl5m1/placeframe/commit/b94ec1eb), [`4b606387`](https://github.com/jcl5m1/placeframe/commit/4b606387), [`46d0838f`](https://github.com/jcl5m1/placeframe/commit/46d0838f) |
| Already on your branches (for reference) | [`c229dbbe`](https://github.com/jcl5m1/placeframe/commit/c229dbbe), [`f0624a2b`](https://github.com/jcl5m1/placeframe/commit/f0624a2b), [`2bbdc88f`](https://github.com/jcl5m1/placeframe/commit/2bbdc88f), [`28af7704`](https://github.com/jcl5m1/placeframe/commit/28af7704), [`523429df`](https://github.com/jcl5m1/placeframe/commit/523429df) |

Our deployment overlay ([`476d9888`](https://github.com/jcl5m1/placeframe/commit/476d9888)) mounts these changes over your published images on one host; it is not
meant for upstream.

## Questions for you

1. Would you take the two fixes (process restart, log volume) as pull requests against `dev`?
2. Are the merge findings (piecewise registration and the shared-points check) useful for `dev_hh`'s merge flow?
3. Server-side or client-side tracking: would you like the tracker and its evaluation tools as a reference, or
   ported to your client filter's interface?
4. Should GPU bundle adjustment come back as an option for large captures, given the numbers above?
