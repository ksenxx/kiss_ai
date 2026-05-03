#!/bin/bash
# Install KISS Sorcar from source: downloads binary dependencies, creates a
# Python virtualenv, and installs Playwright Chromium.
# Log saved to ~/.kiss/install.log
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Standard install locations
BIN_DIR="$HOME/.local/bin"
LOG_DIR="$HOME/.kiss"
LOG_FILE="$LOG_DIR/install.log"
NODE_VERSION="v22.16.0"
UV_VERSION="0.11.2"

mkdir -p "$BIN_DIR" "$LOG_DIR"
export PATH="$BIN_DIR:$PATH"

# Detect OS / arch
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Darwin|Linux) ;;
    *)  echo "ERROR: Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|aarch64|arm64) ;;
    *)  echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Require curl
if ! command -v curl &>/dev/null; then
    echo "ERROR: curl is required but not found. Please install curl first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: ensure Xcode Command Line Tools are installed (macOS only)
#
# Tries a non-interactive `softwareupdate` install first. If that does not
# complete successfully, falls back to the GUI installer triggered by
# `xcode-select --install` and waits for the user to press a key once the
# install dialog finishes.
# ---------------------------------------------------------------------------
ensure_xcode_clt() {
    [ "$OS" = "Darwin" ] || return 0

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools already installed at $(xcode-select -p)"
        return 0
    fi

    echo "   Xcode Command Line Tools not found — attempting non-interactive install..."

    # softwareupdate trick: a sentinel file makes the CLT package appear in
    # the softwareupdate catalog, then install it by its label.
    local SENTINEL=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
    sudo touch "$SENTINEL" 2>/dev/null || true
    local PROD
    PROD="$(softwareupdate -l 2>/dev/null \
        | awk '/^[[:space:]]*\*.*Command Line Tools/ {
                 sub(/^[[:space:]]*\*[[:space:]]*(Label:[[:space:]]*)?/, "");
                 print
             }' \
        | tail -n1)"
    if [ -n "$PROD" ]; then
        echo "   Installing: $PROD"
        sudo softwareupdate -i "$PROD" --verbose 2>&1 || true
    else
        echo "   No Command Line Tools package found in softwareupdate catalog."
    fi
    sudo rm -f "$SENTINEL" 2>/dev/null || true

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools installed at $(xcode-select -p)"
        return 0
    fi

    # Fallback: trigger the GUI installer and wait for the user.
    echo "   Non-interactive install did not complete. Triggering GUI installer..."
    xcode-select --install 2>&1 || true
    echo ""
    echo "   A dialog has appeared to install the Xcode Command Line Tools."
    echo "   Complete the installation in that dialog, then return to this terminal."
    # Read from the controlling terminal so this works inside `{ ... } | tee`.
    if [ -r /dev/tty ]; then
        read -n 1 -s -r -p "   Press any key to continue with the rest of installation..." </dev/tty
    else
        read -n 1 -s -r -p "   Press any key to continue with the rest of installation..."
    fi
    echo ""

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools installed at $(xcode-select -p)"
    else
        echo "   ERROR: Xcode Command Line Tools still not detected. Aborting."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: install git
# ---------------------------------------------------------------------------
install_git() {
    case "$OS" in
        Darwin)
            if command -v brew &>/dev/null; then
                echo "   Installing git via Homebrew..."
                brew install git
            else
                echo "   Triggering Xcode Command Line Tools (provides git)..."
                xcode-select --install 2>&1 || true
                echo "   NOTE: Complete the Xcode CLT dialog, then re-run this script."
                exit 1
            fi
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -y && sudo apt-get install -y git
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y git
            elif command -v yum &>/dev/null; then
                sudo yum install -y git
            elif command -v pacman &>/dev/null; then
                sudo pacman -S --noconfirm git
            elif command -v apk &>/dev/null; then
                sudo apk add git
            else
                echo "   ERROR: No supported package manager found. Install git from https://git-scm.com"
                exit 1
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: install uv from binary tarball
# ---------------------------------------------------------------------------
install_uv() {
    echo "   Downloading uv $UV_VERSION ..."
    local TARGET
    case "$OS" in
        Darwin)
            case "$ARCH" in
                x86_64)         TARGET="x86_64-apple-darwin" ;;
                aarch64|arm64)  TARGET="aarch64-apple-darwin" ;;
            esac
            ;;
        Linux)
            case "$ARCH" in
                x86_64)         TARGET="x86_64-unknown-linux-gnu" ;;
                aarch64|arm64)  TARGET="aarch64-unknown-linux-gnu" ;;
            esac
            ;;
    esac
    local URL="https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-${TARGET}.tar.gz"
    if curl -fsSL "$URL" | tar xz -C "$BIN_DIR" --strip-components=1; then
        echo "   uv $UV_VERSION installed to $BIN_DIR"
    else
        echo "   ERROR: Failed to download uv from $URL"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: install Node.js from binary tarball
# ---------------------------------------------------------------------------
install_node() {
    echo "   Downloading Node.js $NODE_VERSION ..."
    local OS_NODE ARCH_NODE
    OS_NODE="$(echo "$OS" | tr '[:upper:]' '[:lower:]')"   # darwin / linux
    case "$ARCH" in
        x86_64)         ARCH_NODE="x64" ;;
        aarch64|arm64)  ARCH_NODE="arm64" ;;
    esac
    local URL="https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-${OS_NODE}-${ARCH_NODE}.tar.gz"
    mkdir -p "$HOME/.local"
    if curl -fsSL "$URL" | tar xz -C "$HOME/.local" --strip-components=1; then
        echo "   Node.js $NODE_VERSION installed to ~/.local/"
    else
        echo "   ERROR: Failed to download Node.js from $URL"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: install VS Code and ensure CLI is on PATH
# ---------------------------------------------------------------------------
install_code_cli() {
    case "$OS" in
        Darwin)
            local VSCODE_APP="/Applications/Visual Studio Code.app"
            if [ ! -d "$VSCODE_APP" ]; then
                echo "   Downloading VS Code for macOS..."
                local ARCH_VS
                case "$ARCH" in
                    aarch64|arm64) ARCH_VS="darwin-arm64" ;;
                    x86_64)        ARCH_VS="darwin" ;;
                esac
                local TMP_ZIP
                TMP_ZIP="$(mktemp /tmp/vscode-XXXXXX.zip)"
                if curl -fsSL "https://update.code.visualstudio.com/latest/${ARCH_VS}/stable" -o "$TMP_ZIP"; then
                    unzip -q "$TMP_ZIP" -d /Applications/
                    rm -f "$TMP_ZIP"
                    echo "   VS Code installed to /Applications/"
                else
                    rm -f "$TMP_ZIP"
                    echo "   ERROR: Failed to download VS Code"
                    return 1
                fi
            fi
            local CODE_BIN="$VSCODE_APP/Contents/Resources/app/bin/code"
            if [ -x "$CODE_BIN" ]; then
                ln -sf "$CODE_BIN" "$BIN_DIR/code"
                echo "   Linked VS Code CLI to $BIN_DIR/code"
            fi
            ;;
        Linux)
            if command -v snap &>/dev/null; then
                sudo snap install --classic code 2>&1 || true
            elif command -v apt-get &>/dev/null; then
                curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
                    | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg 2>&1
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
                    | sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null 2>&1
                sudo apt-get update -y && sudo apt-get install -y code 2>&1
            elif command -v dnf &>/dev/null; then
                sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc 2>&1
                sudo tee /etc/yum.repos.d/vscode.repo >/dev/null <<'REPO'
[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc
REPO
                sudo dnf install -y code 2>&1
            else
                echo "   Please install VS Code from https://code.visualstudio.com"
                return 1
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: locate VS Code CLI binary
# ---------------------------------------------------------------------------
find_code_cli() {
    CODE_CLI=""
    for candidate in \
        "$(command -v code 2>/dev/null || true)" \
        "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
        "$BIN_DIR/code" \
        "/usr/local/bin/code" \
        "/usr/bin/code" \
        "/snap/bin/code"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            CODE_CLI="$candidate"
            return 0
        fi
    done
    return 1
}

# === Main install (logged to ~/.kiss/install.log) =========================
{
    echo "=== KISS Sorcar Install ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Directory: $PROJECT_DIR"
    echo "OS: $OS ($ARCH)"
    echo ""

    # --- 0. Xcode Command Line Tools (macOS only) ----------------------------
    if [ "$OS" = "Darwin" ]; then
        echo ">>> Checking Xcode Command Line Tools..."
        ensure_xcode_clt
        echo ""
    fi

    # --- 1. git ---------------------------------------------------------------
    echo ">>> [1/11] Checking git..."
    if ! command -v git &>/dev/null; then
        install_git
    fi
    echo "   $(git --version) ready"
    echo ""

    # --- 2. uv ----------------------------------------------------------------
    echo ">>> [2/11] Installing uv..."
    if ! command -v uv &>/dev/null; then
        install_uv
    fi
    echo "   $(uv --version) ready"
    echo ""

    # --- 3. Node.js -----------------------------------------------------------
    echo ">>> [3/11] Installing Node.js..."
    if ! command -v node &>/dev/null; then
        install_node || true
    fi
    if command -v node &>/dev/null; then
        echo "   node $(node --version) ready"
    else
        echo "   WARNING: Node.js not installed — some agent tools may be unavailable"
    fi
    echo ""

    # --- 4. VS Code -----------------------------------------------------------
    echo ">>> [4/11] Installing VS Code..."
    if ! find_code_cli; then
        install_code_cli || true
        find_code_cli || true
    fi
    if [ -n "$CODE_CLI" ]; then
        echo "   code CLI ready: $CODE_CLI"
    else
        echo "   WARNING: VS Code not installed — extension install will be skipped"
    fi
    echo ""

    # --- 5. Python environment ------------------------------------------------
    echo ">>> [5/11] Setting up Python environment..."
    cd "$PROJECT_DIR"
    if [ ! -d "$PROJECT_DIR/.venv" ]; then
        echo "   Creating virtual environment with Python 3.13..."
        uv venv --python 3.13
    fi
    uv sync

    # Symlink entry-point scripts into bin
    for script in sorcar check generate-api-docs kiss-web; do
        if [ -f "$PROJECT_DIR/.venv/bin/$script" ]; then
            ln -sf "$PROJECT_DIR/.venv/bin/$script" "$BIN_DIR/$script"
        fi
    done

    echo "   Python environment ready"
    echo ""

    # --- 6. Playwright Chromium -----------------------------------------------
    echo ">>> [6/11] Installing Playwright Chromium..."
    uv run playwright install chromium
    echo ""

    # --- 7. cloudflared (for remote web server tunnel) -------------------------
    echo ">>> [7/11] Installing cloudflared..."
    if ! command -v cloudflared &>/dev/null; then
        case "$OS" in
            Darwin)
                if command -v brew &>/dev/null; then
                    brew install cloudflared
                else
                    CF_ARCH=""
                    case "$ARCH" in
                        x86_64)         CF_ARCH="amd64" ;;
                        aarch64|arm64)  CF_ARCH="arm64" ;;
                    esac
                    CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-${CF_ARCH}.tgz"
                    if curl -fsSL "$CF_URL" | tar xz -C "$BIN_DIR"; then
                        echo "   cloudflared installed to $BIN_DIR"
                    else
                        echo "   WARNING: Failed to download cloudflared from $CF_URL"
                    fi
                fi
                ;;
            Linux)
                CF_ARCH=""
                case "$ARCH" in
                    x86_64)         CF_ARCH="amd64" ;;
                    aarch64|arm64)  CF_ARCH="arm64" ;;
                esac
                CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}"
                if curl -fsSL -o "$BIN_DIR/cloudflared" "$CF_URL"; then
                    chmod +x "$BIN_DIR/cloudflared"
                    echo "   cloudflared installed to $BIN_DIR"
                else
                    echo "   WARNING: Failed to download cloudflared from $CF_URL"
                fi
                ;;
        esac
    fi
    if command -v cloudflared &>/dev/null; then
        echo "   $(cloudflared --version) ready"
    else
        echo "   WARNING: cloudflared not installed — kiss-web --tunnel will be unavailable"
    fi
    echo ""

    # --- 8. Download official Claude Code skills ------------------------------
    echo ">>> [8/11] Downloading official Claude Code skills..."
    CLAUDE_SKILLS_DIR="$PROJECT_DIR/src/kiss/agents/claude_skills"
    if [ -d "$CLAUDE_SKILLS_DIR" ] && [ "$(ls -d "$CLAUDE_SKILLS_DIR"/*/ 2>/dev/null)" ]; then
        echo "   Claude skills already present — skipping download"
    else
        mkdir -p "$CLAUDE_SKILLS_DIR"
        SKILLS_TMP="$(mktemp -d)"
        echo "   Cloning anthropics/claude-code plugins..."
        if git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/anthropics/claude-code.git "$SKILLS_TMP/claude-code" 2>&1; then
            cd "$SKILLS_TMP/claude-code"
            git sparse-checkout set plugins 2>&1
            # Copy each plugin directory into claude_skills
            for plugin_dir in plugins/*/; do
                if [ -d "$plugin_dir" ]; then
                    plugin_name="$(basename "$plugin_dir")"
                    cp -R "$plugin_dir" "$CLAUDE_SKILLS_DIR/$plugin_name"
                fi
            done
            cd "$PROJECT_DIR"
            SKILL_COUNT="$(ls -d "$CLAUDE_SKILLS_DIR"/*/ 2>/dev/null | wc -l | tr -d ' ')"
            echo "   Installed $SKILL_COUNT Claude skills to $CLAUDE_SKILLS_DIR"
        else
            echo "   WARNING: Failed to download Claude Code skills"
        fi
        rm -rf "$SKILLS_TMP"
    fi
    echo ""

    # --- 9. Build VS Code extension ------------------------------------------
    echo ">>> [9/11] Building VS Code extension..."
    VSCODE_EXT_DIR="$PROJECT_DIR/src/kiss/agents/vscode"
    VSIX="$VSCODE_EXT_DIR/kiss-sorcar.vsix"
    if [ -f "$VSIX" ]; then
        echo "   kiss-sorcar.vsix already exists — skipping build"
    else
        cd "$VSCODE_EXT_DIR"
        npm ci
        npm run package
        cd "$PROJECT_DIR"
        if [ -f "$VSIX" ]; then
            echo "   Built $VSIX"
        else
            echo "   WARNING: Failed to build VSIX"
        fi
    fi
    echo ""

    # --- 10. Install VS Code extension ----------------------------------------
    echo ">>> [10/11] Installing VS Code extension..."
    if [ -f "$VSIX" ]; then
        if find_code_cli && [ -n "$CODE_CLI" ]; then
            "$CODE_CLI" --install-extension "$VSIX" --force 2>&1
            echo "   Extension installed into VS Code"
        else
            echo "   WARNING: VS Code CLI not found — skipping extension install"
            echo "   To install manually: code --install-extension $VSIX --force"
        fi
    else
        echo "   WARNING: VSIX not found — skipping extension install"
    fi
    # Clean up source claude_skills now that they are bundled in the extension
    if [ -d "$CLAUDE_SKILLS_DIR" ]; then
        rm -rf "$CLAUDE_SKILLS_DIR"
        echo "   Cleaned up $CLAUDE_SKILLS_DIR (bundled in extension)"
    fi
    echo ""

    # --- 11. Start kiss-web daemon service ------------------------------------
    echo ">>> [11/11] Setting up kiss-web daemon service..."
    KISS_WEB_BIN="$PROJECT_DIR/.venv/bin/kiss-web"
    if [ -x "$KISS_WEB_BIN" ]; then
        case "$OS" in
            Darwin)
                PLIST_LABEL="com.kiss.web-server"
                PLIST_DIR="$HOME/Library/LaunchAgents"
                PLIST_FILE="$PLIST_DIR/${PLIST_LABEL}.plist"
                mkdir -p "$PLIST_DIR"
                # Unload existing service if present
                launchctl bootout "gui/$(id -u)/$PLIST_LABEL" 2>/dev/null || true
                cat > "$PLIST_FILE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${KISS_WEB_BIN}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/kiss-web-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/kiss-web-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${BIN_DIR}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
PLIST
                launchctl bootstrap "gui/$(id -u)" "$PLIST_FILE" 2>/dev/null || \
                    launchctl load -w "$PLIST_FILE" 2>/dev/null || true
                echo "   macOS LaunchAgent installed: $PLIST_FILE"
                echo "   kiss-web will start on login and restart if killed"
                echo "   Logs: ${LOG_DIR}/kiss-web-stdout.log, ${LOG_DIR}/kiss-web-stderr.log"
                ;;
            Linux)
                SYSTEMD_DIR="$HOME/.config/systemd/user"
                SERVICE_FILE="$SYSTEMD_DIR/kiss-web.service"
                mkdir -p "$SYSTEMD_DIR"
                cat > "$SERVICE_FILE" <<SERVICE
[Unit]
Description=KISS Sorcar Remote Web Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${KISS_WEB_BIN}
WorkingDirectory=${PROJECT_DIR}
Restart=always
RestartSec=5
Environment=PATH=${BIN_DIR}:/usr/local/bin:/usr/bin:/bin
StandardOutput=append:${LOG_DIR}/kiss-web-stdout.log
StandardError=append:${LOG_DIR}/kiss-web-stderr.log

[Install]
WantedBy=default.target
SERVICE
                systemctl --user daemon-reload
                systemctl --user enable --now kiss-web
                # Enable lingering so user services run without active login session
                loginctl enable-linger "$(whoami)" 2>/dev/null || true
                echo "   systemd user service installed: $SERVICE_FILE"
                echo "   kiss-web will start on boot and restart if killed"
                echo "   Logs: ${LOG_DIR}/kiss-web-stdout.log, ${LOG_DIR}/kiss-web-stderr.log"
                echo "   Status: systemctl --user status kiss-web"
                ;;
        esac
    else
        echo "   WARNING: kiss-web binary not found — skipping daemon setup"
    fi
    echo ""

    # --- Write install_dir marker (used by env.py) ----------------------------
    printf '%s\n' "$PROJECT_DIR" > "$HOME/.kiss/install_dir"

    # --- Persist PATH in shell rc file ----------------------------------------
    echo "--- Persist PATH in shell rc ---"
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
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Project: $PROJECT_DIR"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to $LOG_FILE"
echo "Open a new terminal for PATH changes to take effect."
