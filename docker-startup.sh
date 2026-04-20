#!/bin/bash
# Docker entrypoint: clone private repo, install KISS, launch code-server.
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

REPO_URL="https://github.com/ksenxx/kiss.git"
REPO_DIR="/home/kiss"

# ---------------------------------------------------------------------------
# 1. Configure git credentials from GH_TOKEN
# ---------------------------------------------------------------------------
if [ -n "$GH_TOKEN" ]; then
    step "Configuring git credentials..."
    git config --global credential.helper store
    echo "https://x-access-token:${GH_TOKEN}@github.com" > "$HOME/.git-credentials"
    chmod 600 "$HOME/.git-credentials"
    info "Git credentials configured"
else
    echo "WARNING: GH_TOKEN not set — git clone of private repo will fail"
fi

# ---------------------------------------------------------------------------
# 2. Clone the private repo
# ---------------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    step "Cloning $REPO_URL to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    info "Repository cloned"
else
    step "Repository already present at $REPO_DIR — pulling latest..."
    cd "$REPO_DIR" && git pull || true
fi

# ---------------------------------------------------------------------------
# 3. Run install.sh
# ---------------------------------------------------------------------------
step "Running install.sh..."
cd "$REPO_DIR"
bash "$REPO_DIR/install.sh"
info "install.sh completed"

# ---------------------------------------------------------------------------
# 4. Install Playwright system deps (requires sudo, installed by code-server image)
# ---------------------------------------------------------------------------
if [ -f "$REPO_DIR/.venv/bin/playwright" ]; then
    step "Installing Playwright system dependencies..."
    sudo "$REPO_DIR/.venv/bin/playwright" install-deps chromium 2>&1 || true
    info "Playwright system deps installed"
fi

# ---------------------------------------------------------------------------
# 5. Install VSIX into code-server (install.sh may not find code-server as 'code')
# ---------------------------------------------------------------------------
VSIX="$REPO_DIR/src/kiss/agents/vscode/kiss-sorcar.vsix"
if [ -f "$VSIX" ]; then
    step "Installing KISS Sorcar extension into code-server..."
    code-server --install-extension "$VSIX" --force 2>&1 || true
    info "Extension installed"
fi

# ---------------------------------------------------------------------------
# 6. Launch code-server via the base image entrypoint
# ---------------------------------------------------------------------------
info "Starting code-server..."
export KISS_PROJECT_PATH="$REPO_DIR"
exec /usr/bin/entrypoint.sh "$@"
