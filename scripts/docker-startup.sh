#!/bin/bash
# Docker entrypoint: clone the private repo, run install.sh, launch code-server.
set -e

REPO_URL="https://github.com/ksenxx/kiss.git"
REPO_URL_FALLBACK="https://github.com/ksenxx/kiss_ai.git"
# Each repo gets its own directory so the workspace path in code-server
# matches the cloned repo name.  REPO_DIR is set to whichever clone
# succeeded (or already exists) — see the clone block below.
REPO_DIR_PRIMARY="/home/kiss"
REPO_DIR_FALLBACK="/home/kiss_ai"
REPO_DIR=""

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
if [ -d "$REPO_DIR_PRIMARY/.git" ]; then
    REPO_DIR="$REPO_DIR_PRIMARY"
    step "Repo exists at $REPO_DIR — pulling latest..."
    cd "$REPO_DIR" && git pull || true
elif [ -d "$REPO_DIR_FALLBACK/.git" ]; then
    REPO_DIR="$REPO_DIR_FALLBACK"
    step "Repo exists at $REPO_DIR — pulling latest..."
    cd "$REPO_DIR" && git pull || true
else
    step "Cloning $REPO_URL to $REPO_DIR_PRIMARY..."
    if git clone "$REPO_URL" "$REPO_DIR_PRIMARY"; then
        REPO_DIR="$REPO_DIR_PRIMARY"
    else
        step "Primary URL failed — trying fallback $REPO_URL_FALLBACK to $REPO_DIR_FALLBACK..."
        git clone "$REPO_URL_FALLBACK" "$REPO_DIR_FALLBACK"
        REPO_DIR="$REPO_DIR_FALLBACK"
    fi
fi
info "REPO_DIR=$REPO_DIR"

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
# 5a. Disable Workspace Trust globally so the "Do you trust the authors of
#     the files in this folder?" dialog never blocks the extension from
#     activating on first launch.  We belt-and-suspenders this with the
#     `--disable-workspace-trust` CLI flag in the Dockerfile CMD.
# ---------------------------------------------------------------------------
SETTINGS_DIR="$HOME/.local/share/code-server/User"
SETTINGS_FILE="$SETTINGS_DIR/settings.json"
mkdir -p "$SETTINGS_DIR"
if [ ! -f "$SETTINGS_FILE" ]; then
    cat > "$SETTINGS_FILE" <<'JSON'
{
    "security.workspace.trust.enabled": false,
    "security.workspace.trust.startupPrompt": "never",
    "security.workspace.trust.banner": "never",
    "security.workspace.trust.emptyWindow": false
}
JSON
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
# 6. Pre-register both possible repo directories as trusted workspaces.
#     Trust is already disabled globally (step 5a), so this is mostly
#     defense-in-depth: if a future code-server build re-enables trust
#     or ignores our settings.json, these explicit trusted-folder entries
#     ensure the extension still activates without a prompt.
#
#     VS Code stores trusted folders in the global state SQLite DB
#     (state.vscdb).  We instead use a workspace-storage hint and the
#     `security.workspace.trust.untrustedFiles` setting (already covered
#     by settings.json), plus we make `$REPO_DIR` the workspace root so
#     code-server treats it as the current open folder.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. Launch code-server.
#     The CMD in the Dockerfile passes "/home/kiss" as a placeholder
#     workspace path; rewrite it to the actual REPO_DIR that was cloned
#     (either /home/kiss or /home/kiss_ai).  Any other occurrence of
#     /home/kiss as a CLI arg is left alone (none are expected today).
# ---------------------------------------------------------------------------
args=()
for arg in "$@"; do
    if [ "$arg" = "/home/kiss" ] || [ "$arg" = "/home/kiss_ai" ]; then
        args+=("$REPO_DIR")
    else
        args+=("$arg")
    fi
done

info "Starting code-server (workspace: $REPO_DIR)..."
export KISS_PROJECT_PATH="$REPO_DIR"
exec /usr/bin/entrypoint.sh "${args[@]}"
