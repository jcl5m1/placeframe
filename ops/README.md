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
| merge_jobs | job log silent 45 min in a local stage with idle pipeline processes; failed jobs | kill hung process, restart service (job resumes after its last finished stage) |
| study | cost-study driver dead with variants pending; new variant errors | respawn (restart-safe) |
| disk | < 15 GB free | archive finished job workspaces to `/mnt/hdd12t/placeframe_scratch` (symlinked) |
| logs | log > 300 MB; log growing > 100 MB/h; reconstructor > 200k lines / 5 min | truncate keeping the last 20 MB; flag the noisy source |

Long bundle adjustments are not stalls: a busy worker (CPU > 25%) that is still logging is left alone.
State: `~/pf_watchdog/state.json`. Dry run: `python ops/watchdog.py --dry-run`.

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
