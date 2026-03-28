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

# Require Apple Silicon — x86 is not supported
if [ "$(uname -m)" != "arm64" ] && [ "$(uname -m)" != "aarch64" ]; then
    echo "ERROR: Sorcar cannot be installed on x86 hardware. Apple Silicon (arm64) is required."
    exit 1
fi

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
if [ "$OS" = "macos" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
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
# Write install_dir marker (used by env.py for installer compat)
# ---------------------------------------------------------------------------
mkdir -p "$HOME/.kiss"
printf '%s\n' "$PROJECT_DIR" > "$HOME/.kiss/install_dir"

# ---------------------------------------------------------------------------
# Add PATH to shell rc file
# ---------------------------------------------------------------------------
_add_path_to_rc() {
    local rc_file="$1"
    local path_line="export PATH=\"$BIN_DIR:\$PATH\""
    if [ -f "$rc_file" ]; then
        if ! grep -qF "$BIN_DIR" "$rc_file"; then
            printf '\n# KISS Agent Framework\n%s\n' "$path_line" >> "$rc_file"
            echo "   Added $BIN_DIR to $rc_file"
        else
            echo "   $BIN_DIR already in $rc_file"
        fi
    else
        printf '# KISS Agent Framework\n%s\n' "$path_line" > "$rc_file"
        echo "   Created $rc_file with PATH"
    fi
}

echo ">>> Configuring shell profile..."
case "${SHELL:-/bin/zsh}" in
    */zsh)  _add_path_to_rc "$HOME/.zshrc" ;;
    */bash) _add_path_to_rc "$HOME/.bashrc" ;;
    */fish)
        _fish_config="$HOME/.config/fish/config.fish"
        mkdir -p "$(dirname "$_fish_config")"
        _fish_line="fish_add_path $BIN_DIR"
        if [ -f "$_fish_config" ] && grep -qF "$BIN_DIR" "$_fish_config"; then
            echo "   $BIN_DIR already in $_fish_config"
        else
            printf '\n# KISS Agent Framework\n%s\n' "$_fish_line" >> "$_fish_config"
            echo "   Added $BIN_DIR to $_fish_config"
        fi
        ;;
    *)      _add_path_to_rc "$HOME/.zshrc"
            _add_path_to_rc "$HOME/.bashrc" ;;
esac

echo ""
echo "=== Installation Complete ==="
echo "Project: $PROJECT_DIR"
echo ""
echo "Open a new terminal for PATH changes to take effect."
