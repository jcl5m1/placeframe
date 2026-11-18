#!/usr/bin/env bash
# One-shot setup for Placeframe + the Insta360 converter on Ubuntu.
#
#   sudo ./install.sh              # installs system packages, then sets up the user-side bits
#
# What it does (idempotent — safe to re-run):
#   1. apt: ffmpeg, ca-certificates, curl, git
#   2. Docker Engine + Compose plugin (official repo), adds you to the `docker` group
#   3. NVIDIA Container Toolkit, registers the nvidia runtime with Docker
#   4. As your normal user: uv, the converter's Python env, Placeframe's workspace, .env check
#   5. Verifies a GPU container can see the card
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

TARGET_USER="${SUDO_USER:-}"
if [[ -z "$TARGET_USER" || "$TARGET_USER" == "root" ]]; then
  echo "Run via sudo from your normal account (SUDO_USER is not set)." >&2
  exit 1
fi
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLACEFRAME_DIR="${PLACEFRAME_DIR:-$(dirname "$HERE")/placeframe}"
as_user() { sudo -u "$TARGET_USER" -H env PATH="$TARGET_HOME/.local/bin:$PATH" bash -lc "$*"; }
step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" && "${ID_LIKE:-}" != *ubuntu* && "${ID:-}" != "debian" ]]; then
  echo "This script targets Ubuntu/Debian (found ${PRETTY_NAME:-unknown})." >&2
  exit 1
fi
DOCKER_DISTRO=ubuntu; [[ "${ID}" == "debian" ]] && DOCKER_DISTRO=debian
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"

step "Base packages (ffmpeg, curl, git)"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg git ffmpeg

step "Docker Engine + Compose plugin"
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${DOCKER_DISTRO}/gpg" -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  # Docker may not publish a repo for brand-new releases yet; fall back to the latest LTS codename.
  if ! curl -fsI "https://download.docker.com/linux/${DOCKER_DISTRO}/dists/${CODENAME}/Release" >/dev/null; then
    echo "No Docker repo for '${CODENAME}', falling back to 'noble'."
    CODENAME=noble
  fi
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${DOCKER_DISTRO} ${CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
  echo "Docker already installed: $(docker --version)"
fi
systemctl enable --now docker
usermod -aG docker "$TARGET_USER"

step "NVIDIA Container Toolkit"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  if ! command -v nvidia-ctk >/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
else
  echo "WARNING: nvidia-smi not working — skipping GPU runtime. Placeframe's reconstructor/localizer need a GPU."
fi

step "User-side setup for $TARGET_USER"
as_user 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'
as_user "cd '$HERE' && uv sync"
if [[ -d "$PLACEFRAME_DIR" ]]; then
  as_user "cd '$PLACEFRAME_DIR' && uv sync"
  [[ -f "$PLACEFRAME_DIR/.env" ]] || as_user "cp '$PLACEFRAME_DIR/.env.sample' '$PLACEFRAME_DIR/.env'"
else
  as_user "git clone https://github.com/outernet-foundation/placeframe.git '$PLACEFRAME_DIR' && cd '$PLACEFRAME_DIR' && uv sync && cp .env.sample .env"
fi

step "Verifying GPU access from a container"
if command -v nvidia-ctk >/dev/null; then
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L || echo "WARNING: GPU container test failed."
fi

step "Done"
ENV_FILE="$PLACEFRAME_DIR/.env"
if grep -q '<your_' "$ENV_FILE"; then
  cat <<EOF
Still needed: edit $ENV_FILE and set
  PUBLIC_DOMAIN=<your ngrok static domain, e.g. foo-bar.ngrok-free.app>
  NGROK_AUTHTOKEN=<token from https://dashboard.ngrok.com/get-started/your-authtoken>
(and change the default passwords before exposing it).
EOF
fi
cat <<EOF

Log out and back in (or run: newgrp docker) so '$TARGET_USER' can use Docker without sudo, then:

  cd $PLACEFRAME_DIR && uv run up --quiet-pull          # start the backend
  cd $HERE
  uv run insta360_to_placeframe.py ~/walk.mp4 --camera-height 1.6
  uv run upload_to_placeframe.py ~/walk_placeframe/capture.tar
EOF
