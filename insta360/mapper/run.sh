#!/bin/sh
# Start the map-updater service (needs docker group access for OpenVIO).
cd "$(dirname "$0")/.."
exec sg docker -c "uv run uvicorn mapper.server:app --host 127.0.0.1 --port 8150 --root-path /mapper"
