#!/usr/bin/env python3
"""Nightly backup of the Placeframe server state to the HDD: everything needed to rebuild every map and account.

  * Postgres: pg_dumpall (all databases + roles; Keycloak users live there too), zstd-compressed, one file per run,
    the last KEEP_DUMPS kept.
  * MinIO: every bucket mirrored (incremental: only new/changed objects are copied) with the stack's MinIO client
    image, so captures, reconstructions and recordings are kept as plain files.
  * backups/placeframe/last_backup.json records what was saved; the watchdog warns when it is older than 36 h.

  python ops/backup.py              # run from cron at 03:30
  python ops/backup.py --verify     # restore the latest dump into a throwaway Postgres and compare row counts
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

DEST = Path("/mnt/hdd12t/backups/placeframe")
KEEP_DUMPS = 14
WORKER_ENV = Path.home() / "pf_fleet" / "local-worker.env"  # holds the MinIO credentials the workers use
MC_IMAGE = "cgr.dev/chainguard/minio-client@sha256:e1f3f37e8b13f34be49494f441ca01ff7be4e05d7cf04bf26681b0fe380852e1"
PG_IMAGE = "postgis/postgis:16-3.4-alpine"
TABLES = ("capture_sessions", "reconstructions", "localization_maps")
# Localization recordings are written to the SSD (the API must not depend on the HDD); every day is archived to the
# HDD here, and days older than KEEP_RECORDING_DAYS leave the SSD once their archive copy is complete.
RECORDINGS = Path.home() / "pf_data" / "localization_recordings"
RECORDINGS_ARCHIVE = Path("/mnt/hdd12t/placeframe_scratch/localization_recordings")
KEEP_RECORDING_DAYS = 14


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
        return
    if not os.path.ismount("/mnt/hdd12t"):
        raise SystemExit("/mnt/hdd12t is not mounted: nothing backed up or archived tonight")
    archived = archive_recordings()
    t0 = time.time()
    (DEST / "postgres").mkdir(parents=True, exist_ok=True)
    (DEST / "minio").mkdir(parents=True, exist_ok=True)
    dump = DEST / "postgres" / f"pg_dumpall-{time.strftime('%Y%m%d-%H%M%S')}.sql.zst"
    sh(f"docker exec placeframe-postgres-1 pg_dumpall -U postgres | zstd -q -T0 -10 -o {dump}")
    for old in sorted((DEST / "postgres").glob("pg_dumpall-*.sql.zst"))[:-KEEP_DUMPS]:
        old.unlink()
    env = dict(line.split("=", 1) for line in WORKER_ENV.read_text().splitlines() if "=" in line and not line.startswith("#"))
    creds = Path.home() / "pf_fleet" / ".mc_backup.env"  # credentials via a private file, not the command line
    creds.touch(mode=0o600)
    creds.write_text(f"MC_HOST_l=http://{env['MINIO_ACCESS_KEY']}:{env['MINIO_SECRET_KEY']}@minio:9000\n")
    try:
        # Loki's log chunks are skipped: disposable, and their object names contain ':' (not allowed on NTFS).
        listing = sh("docker exec placeframe-minio-1 mc ls l/")
        buckets = [line.split()[-1].rstrip("/") for line in listing.splitlines() if line.strip()]
        results = [sh(f"docker run --rm --network placeframe_default -v {DEST / 'minio'}:/backup --env-file {creds} "
                      f"--entrypoint mc {MC_IMAGE} mirror --overwrite --preserve --quiet l/{b} /backup/{b}", check=False)
                   for b in buckets if b != "loki"]
        mirror = (max(r[0] for r in results), "\n".join(r[1] for r in results if r[0]))
    finally:
        creds.unlink(missing_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0),
              "postgres_dump": dump.name, "postgres_bytes": dump.stat().st_size,
              "minio_bytes": int(sh(f"du -sb {DEST / 'minio'}").split()[0]), "minio_ok": mirror[0] == 0,
              "minio_errors": mirror[1][-2000:] if mirror[0] else "", "live_rows": row_counts("placeframe-postgres-1"),
              "recordings_archived": archived}
    (DEST / "last_backup.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "minio_errors"}))
    if mirror[0]:
        raise SystemExit(f"MinIO mirror failed: {mirror[1][-500:]}")


def archive_recordings() -> dict:
    """Copy every recorded day to the HDD; remove days older than KEEP_RECORDING_DAYS from the SSD once their
    archive copy has every file."""
    RECORDINGS_ARCHIVE.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{RECORDINGS}/", f"{RECORDINGS_ARCHIVE}/"], check=True)
    cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - KEEP_RECORDING_DAYS * 86400))
    removed = []
    for day in sorted(d for d in RECORDINGS.iterdir() if d.is_dir() and d.name < cutoff):
        here = {f.name for f in day.iterdir()}
        there = {f.name for f in (RECORDINGS_ARCHIVE / day.name).iterdir()} if (RECORDINGS_ARCHIVE / day.name).is_dir() else set()
        if here <= there:
            shutil.rmtree(day)
            removed.append(day.name)
    return {"days_on_ssd": sorted(d.name for d in RECORDINGS.iterdir() if d.is_dir()), "removed_from_ssd": removed}


def verify() -> None:
    """Restore the newest dump into a throwaway Postgres container and compare the main tables with the live DB."""
    dump = sorted((DEST / "postgres").glob("pg_dumpall-*.sql.zst"))[-1]
    name = "pf-backup-verify"
    sh(f"docker rm -f {name}", check=False)
    sh(f"docker run -d --name {name} -e POSTGRES_PASSWORD=verify {PG_IMAGE}")
    try:
        for _ in range(60):
            if sh(f"docker exec {name} pg_isready -U postgres", check=False)[0] == 0:
                break
            time.sleep(1)
        time.sleep(3)
        sh(f"zstd -dc {dump} | docker exec -i {name} psql -q -U postgres -f - > /dev/null 2>&1", check=False)
        restored, live = row_counts(name), row_counts("placeframe-postgres-1")
        print(json.dumps({"dump": dump.name, "restored": restored, "live": live, "match": restored == live}, indent=1))
        if restored != live:
            raise SystemExit("restored row counts differ from the live database")
    finally:
        sh(f"docker rm -f {name}", check=False)


def row_counts(container: str) -> dict[str, int]:
    q = " union all ".join(f"select '{t}', count(*) from {t}" for t in TABLES)
    out = sh(f'docker exec {container} psql -U postgres -d placeframe -At -F= -c "{q}"')
    return {k: int(v) for k, v in (line.split("=") for line in out.splitlines() if "=" in line)}


def sh(cmd: str, check: bool = True):
    p = subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True)
    if check:
        if p.returncode:
            raise SystemExit(f"failed ({p.returncode}): {cmd[:160]}\n{p.stderr[-1500:]}")
        return p.stdout
    return p.returncode, p.stdout + p.stderr


if __name__ == "__main__":
    main()
