# Placeframe: bug fixes and extensions from a self-hosted deployment

**To:** the Placeframe team · **Status:** proposal; nothing submitted yet

We have been running Placeframe self-hosted on a single RTX 3090 workstation to map homes with an Insta360 X5
(360° video) and localize phones against those maps. Along the way we fixed some bugs in Placeframe and built
extensions around it. We would like to contribute the bug fixes, and to ask whether any of the extensions are
of interest before we prepare them as pull requests.

## Bug fixes (ready as small, separate pull requests)

Branch `fixes/upstream`: five commits on top of `main` (6 files, +87/−22). Each is independent.

| Fix | Symptom we hit | Change |
|---|---|---|
| Commit the stale-lease reap in its own transaction | A crashed worker's job stayed "in progress" for good: `request_lease` raises `NotFoundException` on an empty queue, which rolls back the session, including the reap | Reap in a separate session, committed before the queue lookup |
| Refresh the reconstructor's access token during long jobs | A multi-hour build finished but could not report success (expired token) | Background refresh of the shared `Authorization` header every 30 s |
| Write reconstructed frames in capture order | Trajectory overlays drew a scrambled path (frames were in COLMAP registration order) | Sort frames by rig and timestamp before writing |
| Restart the reconstructor process after each job | A long-lived worker kept ~8 GB after each large build; a later 7,242-image build started short of memory, swapped for an hour and was OOM-killed (the build itself peaks at 11 GB) | After each job the worker `execv`s a fresh process; `RECONSTRUCTOR_RECYCLE_AFTER_JOB=0` opts out |
| Limit reconstructor progress output | ~209,000 log lines for one build (170,000 from geometry verification): the publisher printed every tick and encoding/matching printed per image/batch | Print at the API flush rate (~2 Hz) and every 5% |

## Extensions (are any of these of interest?)

Placeframe changes on branch `local/main` (each is its own commit; all opt-in or behind a setting):

1. **Fast prior-pose mapper.** Start every frame at its capture pose and refine with pose-prior bundle adjustment
   instead of incremental registration. Merging a new recording into an existing ~2,000-image map: about 40 min
   instead of 4–6+ h. Chosen automatically for merges and for captures with a tight position prior.
2. **GPU bundle adjustment.** pycolmap built with CUDA (Ceres + cuDSS): one global BA on a ~1,250-frame map takes
   29 s instead of 293 s; first builds about 4× faster end to end on our test capture.
3. **Panoramic rig pairing.** For rigs whose cameras share a centre (360° views split into pinhole images), pair
   frames by proximity and match only views whose fields of view overlap.
4. **Preemptible workers.** Lease heartbeat, configurable lease timeout, re-queueing of timed-out jobs, and stage
   checkpoints (features, matches) to object storage so a re-leased job resumes.
5. **Place maps on the globe from GPS.** A capture may include `location.json`; a map created without a placement
   is placed from it (ENU at the fix), so location-based map lookup finds it. The Unity capture tool records the
   device location with each capture.
6. **Localization request recording** (optional) for replaying real client sessions in tests.

Outside Placeframe (separate repository; happy to share):

- **Insta360 X5 pipeline:** `.insv` to captures (dual-fisheye calibration, OpenVIO odometry), merging new recordings
  into existing maps, GPS from the camera's metadata.
- **Credential-free GPU fleet:** a lease agent on the server plus a broker that runs jobs on rented interruptible
  GPUs (Vast.ai) using only pre-signed object-storage URLs, with an access log and checkpoint resume. Tested end to
  end; a rented worker is ready about 2 min 20 s after it is created.
- **Flutter client (iOS/Android):** a lightweight localization and capture app without Unity, sharing all code except
  a small ARKit/ARCore frame module per platform (Android builds; iOS untested).
- **Operations:** a web dashboard for uploads and builds, a watchdog that restarts stuck work and flags failures,
  and nightly backups.

## Questions for you

1. Would you accept the five bug fixes as pull requests? Any preferred format: one per fix, or one combined PR?
2. Which extensions, if any, would you like upstream? The prior-pose mapper and GPU bundle adjustment seemed the most
   broadly useful to us.
3. For the Flutter client and the fleet: would you prefer them in the main repository, or as separate repositories
   that link to Placeframe?

All changes follow the repository's `CLAUDE.md` conventions, and we will run `uv run --no-sync preflight` on each
pull request before submitting.
