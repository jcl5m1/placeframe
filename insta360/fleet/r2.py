#!/usr/bin/env python3
"""Cloudflare R2 for the fleet: an S3 client from ~/pf_fleet/r2.env (or R2_* environment variables, as on rented
workers), plus put/get with parallel multipart transfers and a throughput test.

  python fleet/r2.py speedtest <file> [--key _speedtest/x] [--keep]
  python fleet/r2.py put <file> <key>
  python fleet/r2.py get <key> <file>
  python fleet/r2.py ls [prefix]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

ENV_FILE = Path.home() / "pf_fleet/r2.env"
# 64 MB parts, 16 in flight: saturates ~1-10 Gbps links without much memory.
TRANSFER = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speedtest")
    s.add_argument("file", type=Path)
    s.add_argument("--key", default="_speedtest/sample.bin")
    s.add_argument("--keep", action="store_true")
    p = sub.add_parser("put")
    p.add_argument("file", type=Path)
    p.add_argument("key")
    g = sub.add_parser("get")
    g.add_argument("key")
    g.add_argument("file", type=Path)
    ls = sub.add_parser("ls")
    ls.add_argument("prefix", nargs="?", default="")
    a = ap.parse_args()

    s3, bucket = client()
    if a.cmd == "put":
        print(f"{put(s3, bucket, a.file, a.key):.0f} Mbit/s")
    elif a.cmd == "get":
        print(f"{get(s3, bucket, a.key, a.file):.0f} Mbit/s")
    elif a.cmd == "ls":
        for o in s3.list_objects_v2(Bucket=bucket, Prefix=a.prefix).get("Contents", []):
            print(f"{o['Size'] / 1e6:10.1f} MB  {o['Key']}")
    else:
        size = a.file.stat().st_size
        up = put(s3, bucket, a.file, a.key)
        down = get(s3, bucket, a.key, a.file.with_suffix(".r2check"))
        a.file.with_suffix(".r2check").unlink()
        if not a.keep:
            s3.delete_object(Bucket=bucket, Key=a.key)
        print(f"{size / 1e9:.2f} GB: upload {up:.0f} Mbit/s, download {down:.0f} Mbit/s")


def client():
    env = dict(os.environ)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.removeprefix("export ").strip(), v.strip().strip("'\""))
    s3 = boto3.client("s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                      aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
                      region_name="auto", config=Config(retries={"max_attempts": 5, "mode": "adaptive"},
                                                        max_pool_connections=32))
    return s3, env["R2_BUCKET"]


def put(s3, bucket: str, path: Path, key: str) -> float:
    t = time.monotonic()
    s3.upload_file(str(path), bucket, key, Config=TRANSFER)
    return path.stat().st_size * 8 / 1e6 / (time.monotonic() - t)


def get(s3, bucket: str, key: str, path: Path) -> float:
    t = time.monotonic()
    s3.download_file(bucket, key, str(path), Config=TRANSFER)
    return path.stat().st_size * 8 / 1e6 / (time.monotonic() - t)


if __name__ == "__main__":
    main()
