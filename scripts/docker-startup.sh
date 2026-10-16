#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Docker entrypoint: clone the repository, run its install.sh, start kiss-web.
#
# Environment:
#   GH_TOKEN             GitHub token; with one the private kiss.git is
#                        cloned, without one (or when it is not reachable)
#                        the public kiss_ai.git
#   KISS_WEB_PASSWORD    password of the webapp; empty keeps the one already
#                        configured in this container's state, else a random
#                        one is generated.  The password in force is printed.
#   model API keys       ANTHROPIC_API_KEY & co. pass through to the agents
set -e

REPO_URL="https://github.com/ksenxx/kiss.git"
REPO_URL_FALLBACK="https://github.com/ksenxx/kiss_ai.git"
# One directory per repository name; kiss-web serves whichever was cloned.
REPO_DIR_PRIMARY="$HOME/kiss"
REPO_DIR_FALLBACK="$HOME/kiss_ai"
REPO_DIR=""
# The daemon's state (config, TLS certificates, task history) and the app
# install; install.sh and kiss-web both honour KISS_HOME.
export KISS_HOME="$HOME/.kiss"
APP_DIR="$KISS_HOME/kiss"

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
step() { printf '\033[0;34m[STEP]\033[0m  %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Git credentials from GH_TOKEN (optional: the public repo needs none)
# ---------------------------------------------------------------------------
if [ -n "${GH_TOKEN:-}" ]; then
    step "Configuring git credentials..."
    git config --global credential.helper store
    echo "https://x-access-token:${GH_TOKEN}@github.com" > "$HOME/.git-credentials"
    chmod 600 "$HOME/.git-credentials"
fi

# ---------------------------------------------------------------------------
# 2. Clone or pull the repository
# ---------------------------------------------------------------------------
if [ -d "$REPO_DIR_PRIMARY/.git" ]; then
    REPO_DIR="$REPO_DIR_PRIMARY"
elif [ -d "$REPO_DIR_FALLBACK/.git" ]; then
    REPO_DIR="$REPO_DIR_FALLBACK"
fi
if [ -n "$REPO_DIR" ]; then
    step "Repository exists at $REPO_DIR; pulling the latest..."
    git -C "$REPO_DIR" pull --ff-only || true
elif [ -n "${GH_TOKEN:-}" ] && git clone "$REPO_URL" "$REPO_DIR_PRIMARY"; then
    REPO_DIR="$REPO_DIR_PRIMARY"
else
    [ -n "${GH_TOKEN:-}" ] && info "$REPO_URL is not reachable with this token; using the public repository"
    step "Cloning $REPO_URL_FALLBACK to $REPO_DIR_FALLBACK..."
    git clone "$REPO_URL_FALLBACK" "$REPO_DIR_FALLBACK"
    REPO_DIR="$REPO_DIR_FALLBACK"
fi
info "REPO_DIR=$REPO_DIR"

# ---------------------------------------------------------------------------
# 3. Install: the app under $APP_DIR, Playwright's Chromium, cloudflared.
#    KISS_SKIP_LAUNCH: this script starts kiss-web itself, in the foreground,
#    so the container lives exactly as long as the daemon.  --non-interactive:
#    automation must never sit at a prompt.
# ---------------------------------------------------------------------------
step "Running $REPO_DIR/install.sh..."
export PATH="$HOME/.local/bin:$PATH"
KISS_SKIP_LAUNCH=1 bash "$REPO_DIR/install.sh" --non-interactive
info "install.sh completed"

# ---------------------------------------------------------------------------
# 4. Password and work directory.  The host's browser reaches the container
#    through Docker's bridge, never over the container's loopback, and
#    kiss-web refuses every non-loopback peer while no password is set.
# ---------------------------------------------------------------------------
step "Configuring the webapp password..."
PASSWORD_LINE=$(python3 "$REPO_DIR/src/kiss/scripts/remote_config.py" \
    "$KISS_HOME/config.json" "$REPO_DIR" "${KISS_WEB_PASSWORD:-}")
info "Webapp password: ${PASSWORD_LINE#SORCAR_PASSWORD=}"

# ---------------------------------------------------------------------------
# 5. Start kiss-web in the foreground (PID 1's child; signals stop it).
# ---------------------------------------------------------------------------
info "Starting kiss-web (work dir: $REPO_DIR) on port 8787..."
exec "$APP_DIR/.venv/bin/kiss-web" --workdir "$REPO_DIR"
