#!/usr/bin/env bash
set -euo pipefail

# A freshly created volume is root-owned — hand .venv to the remote user before
# anything tries to write into it.
sudo chown -R "$(whoami)" .venv 2>/dev/null || true

# Qt runtime libs. PySide6 ships its own Qt build but links against these system
# libraries, and the devcontainers base image is too slim to have them: without
# them even QT_QPA_PLATFORM=offscreen (what tests/test_gui.py sets) fails to load
# the platform plugin. The xcb-* set is what the on-screen X11 plugin needs for
# `mise run gui`; fonts-dejavu-core keeps rendered text from falling back to
# boxes.
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  libegl1 libgl1 libglib2.0-0 libdbus-1-3 libfontconfig1 fonts-dejavu-core \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
  libxcb-xinerama0 libxcb-xkb1
sudo rm -rf /var/lib/apt/lists/*

# Provision the mise toolchain (uv and node), then install both dependency sets.
# `mise` itself is on PATH here, but the tools it manages are NOT auto-activated
# in this non-interactive shell, so install steps must go through `mise exec --`.
# uv resolves the interpreter itself from .python-version, downloading it if the
# managed build isn't present yet. npm covers the browser app's dev tooling
# (typescript, vitest, eslint, prettier); nothing it installs ships to the page.
mise trust && mise install
mise exec -- uv sync
mise exec -- npm ci

# GitHub auth over SSH: use only ~/.ssh/id_git (bind-mounted read-only from the
# host). `IdentitiesOnly yes` makes ssh ignore every other key the forwarded
# agent offers (e.g. work deploy keys), so the correct key is always used for
# github.com — otherwise a read-only deploy key can win the handshake and block
# pushes.
sudo install -d -m 700 -o "$(whoami)" -g "$(whoami)" "$HOME/.ssh"
cat > "$HOME/.ssh/config" <<'EOF'
Host github.com
  User git
  IdentityFile ~/.ssh/id_git
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF
chmod 600 "$HOME/.ssh/config"
