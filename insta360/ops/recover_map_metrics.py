#!/usr/bin/env python3
"""Recompute a reconstruction's map metrics from its stored sfm_model and write them into the DB manifest.

Needed when a build finished but could not report (expired worker token) and was marked succeeded by hand
or by the watchdog: the localizer requires map_* metrics (map_image_count, map_point_count,
map_avg_track_length, map_bounding_volume_m3, map_viewpoint_diversity) and returns 502 without them.

Uses the reconstructor image's own MetricsBuilder.build_reconstruction_metrics, so values match a normal
build. Verified-match metrics (which need the feature database) stay as they are.

  python ops/recover_map_metrics.py <reconstruction_id> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

FILES = ["cameras.txt", "frames.txt", "images.txt", "points3D.txt", "rigs.txt"]
SCRIPT = """
import json, pycolmap
from reconstructor.metrics_builder import MetricsBuilder
rec = pycolmap.Reconstruction()
rec.read_text('/m')
b = MetricsBuilder()
b.build_reconstruction_metrics(rec)
print('METRICS' + json.dumps({k: v for k, v in b.metrics.model_dump().items() if v is not None}))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reconstruction_id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    metrics = recompute(a.reconstruction_id)
    print(json.dumps(metrics, indent=2))
    if not a.dry_run:
        apply(a.reconstruction_id, metrics)
        print("manifest metrics updated")


def recompute(rid: str) -> dict:
    image = sh("docker inspect placeframe-reconstructor-cuda-1 --format '{{.Config.Image}}'").strip()
    with tempfile.TemporaryDirectory(dir="/mnt/hdd12t/placeframe_scratch") as tmp:
        for f in FILES:
            data = subprocess.run(["sg", "docker", "-c", f"docker exec placeframe-minio-1 mc cat l/dev-reconstructions/{rid}/sfm_model/{f}"],
                                  capture_output=True, check=True).stdout
            if not data:
                raise SystemExit(f"sfm_model/{f} missing for {rid}")
            (Path(tmp) / f).write_bytes(data)
        script = Path(tmp) / "m.py"
        script.write_text(SCRIPT)
        out = sh(f"docker run --rm --network none --cpus 2 -v {tmp}:/m:ro --entrypoint /opt/uv/venv/bin/python {image} /m/m.py")
    line = next((x for x in out.splitlines() if x.startswith("METRICS")), None)
    if not line:
        raise SystemExit("metrics computation failed:\n" + out[-2000:])
    return json.loads(line[len("METRICS"):])


def apply(rid: str, metrics: dict) -> None:
    patch = json.dumps(metrics).replace("'", "''")
    sql = (f"update reconstructions set manifest = jsonb_set(manifest, '{{metrics}}', "
           f"coalesce(manifest->'metrics', '{{}}'::jsonb) || '{patch}'::jsonb) where id = '{rid}'")
    sh(f"docker exec -i placeframe-postgres-1 psql -U postgres -d placeframe -v ON_ERROR_STOP=1", stdin=sql)


def sh(cmd: str, stdin: str | None = None) -> str:
    p = subprocess.run(["sg", "docker", "-c", cmd], input=stdin, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"command failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    return p.stdout + p.stderr


if __name__ == "__main__":
    main()
