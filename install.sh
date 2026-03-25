#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Standard install locations
BIN_DIR="$HOME/.local/bin"
# Detect OS
OS="$(uname -s)"
case "$OS" in
    Darwin) OS="macos" ;;
    Linux)  OS="linux" ;;
    *)      echo "ERROR: Unsupported OS: $OS"; exit 1 ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
    arm64|aarch64) ARCH="arm64" ;;
esac

mkdir -p "$BIN_DIR"

# ---------------------------------------------------------------------------
# 1. Install or upgrade Homebrew  (https://brew.sh)
# ---------------------------------------------------------------------------
echo ">>> [1/3] Installing Homebrew..."
if command -v brew &> /dev/null; then
    echo "   Homebrew already installed, upgrading..."
    brew update
else
    echo "   Installing Homebrew from latest binaries..."
    if [ "$OS" = "macos" ]; then
        # Grab latest Homebrew.pkg from GitHub releases and install non-interactively
        BREW_PKG_URL="$(curl -sSf --max-time 10 \
            "https://api.github.com/repos/Homebrew/brew/releases/latest" \
            | grep '"browser_download_url".*Homebrew\.pkg' \
            | sed -E 's/.*"(https:[^"]+)".*/\1/')"
        BREW_TMP="$(mktemp -d)"
        curl -fSL -o "$BREW_TMP/Homebrew.pkg" "$BREW_PKG_URL"
        sudo installer -pkg "$BREW_TMP/Homebrew.pkg" -target /
        rm -rf "$BREW_TMP"
    fi
fi

# Add Homebrew to PATH for this session
if [ "$OS" = "macos" ] && [ "$ARCH" = "arm64" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [ "$OS" = "macos" ]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi

echo "   Homebrew $(brew --version | head -1) ready"

# ---------------------------------------------------------------------------
# 2. Install uv from binaries if not installed  (https://astral.sh/uv)
# ---------------------------------------------------------------------------
echo ">>> [2/3] Installing uv..."
if command -v uv &> /dev/null; then
    echo "   uv already installed"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$BIN_DIR:$HOME/.cargo/bin:$PATH"
fi
echo "   uv $(uv --version) ready"

# ---------------------------------------------------------------------------
# 3. Create virtual environment and sync Python dependencies
# ---------------------------------------------------------------------------
echo ">>> [3/3] Setting up Python environment..."
cd "$PROJECT_DIR"
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "   Creating virtual environment with Python 3.13..."
    uv venv --python 3.13
fi
uv sync

# Symlink entry-point scripts into bin
for script in sorcar check generate-api-docs; do
    if [ -f "$PROJECT_DIR/.venv/bin/$script" ]; then
        ln -sf "$PROJECT_DIR/.venv/bin/$script" "$BIN_DIR/$script"
    fi
done

# ---------------------------------------------------------------------------
# 4. Install Playwright Chromium  (https://playwright.dev)
# ---------------------------------------------------------------------------
echo ">>> [4] Installing Playwright Chromium..."
uv run playwright install chromium

# ---------------------------------------------------------------------------
# Write install_dir marker (used by env.py for offline-installer compat)
# ---------------------------------------------------------------------------
mkdir -p "$HOME/.kiss"
printf '%s\n' "$PROJECT_DIR" > "$HOME/.kiss/install_dir"

# ---------------------------------------------------------------------------
# Create env.sh
# ---------------------------------------------------------------------------
PROFILE_SNIPPET="$PROJECT_DIR/env.sh"
cat > "$PROFILE_SNIPPET" << EOF
# KISS Agent Framework - added by install.sh
export PATH="$BIN_DIR:\$PATH"
EOF

# ---------------------------------------------------------------------------
# Add source line to shell rc
# ---------------------------------------------------------------------------
_add_to_shell_rc() {
    local rc_file="$1"
    local source_line="source \"$PROFILE_SNIPPET\""
    if [ -f "$rc_file" ]; then
        if ! grep -qF "$source_line" "$rc_file"; then
            printf '\n%s\n' "$source_line" >> "$rc_file"
            echo "   Added to $rc_file"
        else
            echo "   Already in $rc_file"
        fi
    else
        echo "$source_line" > "$rc_file"
        echo "   Created $rc_file with source line"
    fi
}

echo ">>> Configuring shell profile..."
case "${SHELL:-/bin/zsh}" in
    */zsh)  _add_to_shell_rc "$HOME/.zshrc" ;;
    */bash) _add_to_shell_rc "$HOME/.bashrc" ;;
    *)      _add_to_shell_rc "$HOME/.zshrc"
            _add_to_shell_rc "$HOME/.bashrc" ;;
esac

echo ""
echo "=== Installation Complete ==="
echo "Project: $PROJECT_DIR"
echo ""
echo "Open a new terminal or run: source \"$PROFILE_SNIPPET\""
