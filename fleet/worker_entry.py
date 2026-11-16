#!/usr/bin/env python3
"""Entry point of the fleet worker image. On rented instances the worker's signing key arrives as an
environment variable (WORKER_PRIVATE_KEY_B64); it is written to the path the reconstructor expects, then the
standard reconstructor entrypoint takes over. Locally (key already mounted) it only execs the entrypoint."""
import base64
import os
from pathlib import Path

key = os.environ.pop("WORKER_PRIVATE_KEY_B64", None)
if key:
    path = Path(os.environ.setdefault("PRIVATE_KEY_PATH", "/run/secrets/worker-private-key.pem"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.write(fd, base64.b64decode(key))
    os.close(fd)
os.environ.setdefault("RECONSTRUCTOR_CHECKPOINTS", "1")
os.execv("/usr/local/bin/entrypoint.sh", ["/usr/local/bin/entrypoint.sh"])
