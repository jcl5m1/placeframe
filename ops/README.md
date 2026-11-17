# Operations: watchdog, incidents, logging

## Tier 1 — mechanical watchdog (`ops/watchdog.py`, every 5 min)
Installed in the user crontab (runs without a login session; `flock` = one pass at a time, `timeout 600`):

    */5 * * * * cd ~/github/placeframe-insta360 && flock -n /tmp/pf_watchdog.lock timeout 600 .venv/bin/python ops/watchdog.py >> ~/pf_watchdog/watchdog.log 2>&1

Each pass (~1 s when healthy) checks, and fixes what it can:

| Check | Detects | Action |
|---|---|---|
| services | map updater not answering, sampler/viewer dead, public URL down | `mapper/restart.sh`, respawn (detached), incident |
| containers | any long-running Placeframe container not `running` | `uv run up`, incident with container logs |
| server_builds | build whose status/progress/updated_at hasn't changed for 30 min **and** worker idle (<25% CPU, no log line for 20 min); worker logged success but couldn't report (expired token) | restart worker + mark failed + retry; or mark succeeded when outputs are complete |
| merge_jobs | job log silent 45 min in a local stage with idle pipeline processes; failed jobs | kill that job's process; restart the service (the job resumes after its last finished stage) unless another job is running, then leave it failed for a retry |
| study | cost-study driver dead with variants pending; new variant errors | respawn (restart-safe) |
| disk | < 15 GB free | archive finished job workspaces to `/mnt/hdd12t/placeframe_scratch` (symlinked) |
| cleanup_published | merge jobs whose map was published > 72 h ago | delete large intermediates (keep logs/records), move the raw recording to the HDD (symlinked) |
| backups | last nightly backup older than 36 h, or its MinIO copy failed | incident (banner) |
| memory | < 3 GB available (or swap > 75% used while < 8 GB available); idle reconstructor holding > 8 GB | restart the idle reconstructor; report pressure with the biggest processes |
| logs | log > 300 MB; log growing > 100 MB/h; reconstructor > 200k lines / 5 min | truncate keeping the last 20 MB; flag the noisy source |

Long bundle adjustments are not stalls: a busy worker (CPU > 25%) that is still logging is left alone.
Problems it cannot fix appear in a banner at the top of the mapper dashboard until dismissed there.
While `~/pf_fleet/home_worker_paused` exists, a stopped home reconstructor is not restarted (fleet runs).
State: `~/pf_watchdog/state.json`. Dry run: `python ops/watchdog.py --dry-run`.

Incidents also come from outside the watchdog: a merge whose new recording shares under 30% of its map points with
the existing map (`merge-poorly-joined`, from `add_recording_to_map.py`; the old map stays active) and nightly
localization regressions (`ops/regression.py`).

## Storage policy

The 12 TB HDD (`/mnt/hdd12t`) is for backups and long-term archives only: nothing real-time or latency-sensitive may
depend on it. Live data stays on the SSD (Docker volumes: MinIO, Postgres; `~/pf_data`: localization recordings,
working copies of regression and replay data). `ops/backup.py` copies every recorded day to the HDD each night and
removes days older than 14 from the SSD once the copy is complete; the replay tools read archived days when the HDD
is mounted. When the HDD is offline the watchdog reports `storage-offline` and backups, archiving and old-job cleanup
wait; the services keep running. A stopped container is restarted in place (`docker start`); `uv run up` is only
the last resort, since it recreates containers from published image tags.

## Map updater concurrency

The map updater runs up to `MAPPER_PARALLEL_JOBS` (default 2) jobs at once, so one recording's preprocessing (CPU)
overlaps another's reconstruction (GPU). Recordings of the same upload group, and merges into the same map, still run
one after another: each merges into the map the previous one produced. Within a job, odometry and stitching run side
by side. A job re-run against a different target map redoes registration, merge and build (`target_map.txt`).

## Tier 2 — agent escalation (every 10 min while a Claude session is open)
Reads new `needs_attention` incidents from `~/pf_watchdog/incidents.jsonl`, diagnoses root cause from the recorded
evidence, fixes code/config, restarts/resumes, and logs a one-line resolution to `~/pf_watchdog/resolutions.log`.
If evidence is insufficient it adds targeted logging (below) instead of guessing.

## Logging policy
- One line per *event* with the state needed to debug it (ids, counts, durations) — never per-item spam.
- Progress counters (`k/N`) at a bounded rate (every ~2–5% or every few seconds), parsed by the dashboard.
- Chatty third-party output (COLMAP glog `I2026...` lines) is filtered from job logs; Docker container logs are
  capped at 3 × 50 MB per service (`compose.local.yml`, `x-bounded-logs`).
- When a noisy-log incident fires, reduce verbosity at the source or move it behind a debug flag.

## Diagnostic logging
(Additions made while chasing incidents are listed here with the incident they serve, so they can be removed later.)
- `reconstructor/main.py`: `[Auth] Background token refresh failed: …` — token refresher (incident 2026-11-15: expired
  token stalled a 4.5 h build).

## Recovering a build that finished but couldn't report

`ops/recover_map_metrics.py <reconstruction_id>` recomputes the map metrics the localizer needs
(`map_image_count`, `map_point_count`, `map_avg_track_length`, `map_bounding_volume_m3`, `map_viewpoint_diversity`).
It reads the stored `sfm_model` and uses the reconstructor's own `MetricsBuilder`, then writes the metrics into the
manifest. Without these metrics, `/localize` returns 502 for that map. The watchdog runs this before it marks an
unreported build as succeeded. It was used once by hand, on 2026-11-15, for the merged house map `cd548a59`.

## Changing the reconstructor's mounted code

The local patches (`pairs.py`, `colmap.py`, `main.py`, `prior_mapper.py`) are single-file bind mounts into the
reconstructor container. After editing any of them:

- **Restart the worker:** `sg docker -c "docker restart placeframe-reconstructor-cuda-1"`, and only while no build is
  running. The worker is a long-lived process that imported the modules when it started, so edits do nothing until
  then.
- **Formatters replace the file** (for example `ruff format`), which a running container's single-file mount doesn't
  follow. A restart re-resolves the path. Check with `docker exec … grep` that the container sees the new code.
- **Mapper mode** (`local/reconstructor-mapper.json`, mounted at `/etc/placeframe/mapper.json`) is read at the start
  of every build, so it needs no restart. Write it in place (`echo … > file`).

## Backups (`ops/backup.py`, nightly 03:30)
`/mnt/hdd12t/backups/placeframe/`: `postgres/pg_dumpall-*.sql.zst` (all databases incl. Keycloak; last 14 kept) and
`minio/<bucket>/` (incremental mirror of every bucket except Loki's logs, whose object names NTFS can't store).
`last_backup.json` records the run. `python ops/backup.py --verify` restores the newest dump into a throwaway Postgres
and compares row counts with the live database (passed 2026-11-16: 37 captures, 49 reconstructions, 23 maps).

Restore: `zstd -dc <dump> | docker exec -i placeframe-postgres-1 psql -U postgres -f -`, then copy `minio/<bucket>/`
back with `mc mirror` (stack stopped except Postgres and MinIO).
