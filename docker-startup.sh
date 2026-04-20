#!/bin/bash
# Docker entrypoint: clone the private repo, run install.sh, launch code-server.
set -e

REPO_URL="https://github.com/ksenxx/kiss.git"
REPO_DIR="/home/kiss"

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
step() { printf '\033[0;34m[STEP]\033[0m  %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Configure git credentials from GH_TOKEN
# ---------------------------------------------------------------------------
if [ -n "$GH_TOKEN" ]; then
    step "Configuring git credentials..."
    git config --global credential.helper store
    echo "https://x-access-token:${GH_TOKEN}@github.com" > "$HOME/.git-credentials"
    chmod 600 "$HOME/.git-credentials"
else
    echo "ERROR: GH_TOKEN not set — cannot clone private repo" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Clone or pull the private repo to /home/kiss
# ---------------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    step "Cloning $REPO_URL to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    step "Repo exists at $REPO_DIR — pulling latest..."
    cd "$REPO_DIR" && git pull || true
fi

# ---------------------------------------------------------------------------
# 3. Run install.sh (Python env, Playwright, VS Code extension)
#    Add the venv bin to PATH so copy-kiss.sh (called during VSIX build) can
#    find python3 even though the system has no global python.
# ---------------------------------------------------------------------------
step "Running /home/kiss/install.sh..."
cd "$REPO_DIR"
export PATH="$REPO_DIR/.venv/bin:$HOME/.local/bin:$PATH"
bash "$REPO_DIR/install.sh"
info "install.sh completed"

# ---------------------------------------------------------------------------
# 4. Install Playwright system deps (requires sudo)
# ---------------------------------------------------------------------------
if [ -f "$REPO_DIR/.venv/bin/playwright" ]; then
    step "Installing Playwright system dependencies..."
    sudo "$REPO_DIR/.venv/bin/playwright" install-deps chromium 2>&1 || true
fi

# ---------------------------------------------------------------------------
# 5. Install VSIX into code-server
# ---------------------------------------------------------------------------
VSIX="$REPO_DIR/src/kiss/agents/vscode/kiss-sorcar.vsix"
if [ -f "$VSIX" ]; then
    step "Installing KISS Sorcar extension into code-server..."
    code-server --install-extension "$VSIX" --force 2>&1 || true
    info "Extension installed"
fi

# ---------------------------------------------------------------------------
# 6. Launch code-server
# ---------------------------------------------------------------------------
info "Starting code-server..."
export KISS_PROJECT_PATH="$REPO_DIR"
exec /usr/bin/entrypoint.sh "$@"
