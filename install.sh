#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Install KISS Sorcar from source.
#
# This script's job is intentionally small: bootstrap only the tools needed to
# build and install the VS Code extension from a cloned checkout, then launch
# VS Code.  Runtime setup is owned by the extension's DependencyInstaller so
# users get the same installation path whether they run this script or install
# the VSIX directly.
#
# Log saved to ~/.kiss/install.log
#
# `pipefail` is required so the trailing `{ ... } 2>&1 | tee "$LOG_FILE"`
# pipeline propagates a non-zero exit from the body (e.g. a failed
# `npm run package`) instead of returning `tee`'s always-zero status.
# Without it, a broken VSIX build was silently masked and the container
# ended up shipping the stale committed VSIX.
set -eo pipefail

# Capture the user's working directory *before* any `cd` so that VS Code can
# later be launched with this directory as the workspace root.  The agents
# spawned inside VS Code default their PWD to the workspace root (see
# ``kiss.agents.vscode.server`` — ``os.getcwd()`` is the fallback when
# ``KISS_WORKDIR`` is unset), so opening the workspace here makes the
# agents' PWD match the user's original shell PWD.
USER_PWD="$PWD"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

BIN_DIR="$HOME/.local/bin"
LOG_DIR="$HOME/.kiss"
LOG_FILE="$LOG_DIR/install.log"
NODE_VERSION="v22.16.0"

# Required versions — extracted from the repo's source of truth so that
# install.sh stays in sync with DependencyInstaller.ts and package.json
# without hard-coding duplicates.
DEP_INSTALLER_TS="$PROJECT_DIR/src/kiss/agents/vscode/src/DependencyInstaller.ts"
VSCODE_PACKAGE_JSON="$PROJECT_DIR/src/kiss/agents/vscode/package.json"

# The trailing `|| true` matters: with `set -eo pipefail` an absent file or
# renamed constant makes `grep` exit non-zero, which would otherwise kill the
# whole script at these assignments before it printed anything.  An empty
# version simply skips the corresponding version check below.
REQUIRED_GIT_VERSION=$(grep "const GIT_VERSION" "$DEP_INSTALLER_TS" 2>/dev/null | head -1 | sed "s/.*= '//;s/'.*//" || true)
REQUIRED_UV_VERSION=$(grep "const UV_VERSION" "$DEP_INSTALLER_TS" 2>/dev/null | head -1 | sed "s/.*= '//;s/'.*//" || true)
REQUIRED_VSCODE_VERSION=$(grep '"vscode"' "$VSCODE_PACKAGE_JSON" 2>/dev/null | head -1 | sed 's/[^0-9.]//g' || true)
REQUIRED_NODE_VERSION="${NODE_VERSION#v}"

mkdir -p "$BIN_DIR" "$LOG_DIR"
export PATH="$BIN_DIR:$PATH"

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

if ! command -v curl &>/dev/null; then
    echo "ERROR: curl is required but not found. Please install curl first."
    exit 1
fi

# Make Homebrew visible even when this script runs detached from a login
# shell.  The webapp's update button spawns install.sh from the kiss-web
# daemon, whose launchd/systemd environment has a minimal PATH without
# /opt/homebrew/bin (or /usr/local/bin on Intel Macs).  Without this,
# `command -v brew` failed even though Homebrew was installed, so
# `ensure_homebrew` tried to re-install it and `upgrade_git` aborted the
# whole update with "Cannot upgrade git without Homebrew".
if [ "$OS" = "Darwin" ] && ! command -v brew &>/dev/null; then
    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

# Returns 0 only if /dev/tty can actually be opened for reading.  A plain
# `[ -r /dev/tty ]` test only inspects the permission bits, which pass even
# inside a detached Docker container where the controlling terminal does not
# exist and opening /dev/tty fails with ENXIO ("No such device or address").
# Probing with a real open avoids that crash and lets prompts fall back to
# their non-interactive default.
can_read_tty() {
    { : < /dev/tty; } 2>/dev/null
}

ensure_xcode_clt() {
    [ "$OS" = "Darwin" ] || return 0

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools already installed at $(xcode-select -p)"
        return 0
    fi

    echo "   Xcode Command Line Tools not found — attempting non-interactive install..."

    local SENTINEL=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
    sudo touch "$SENTINEL" 2>/dev/null || true
    local PROD
    # `|| true` keeps a failing `softwareupdate` (no network, managed Macs)
    # from killing the script via `set -eo pipefail`.
    PROD="$(softwareupdate -l 2>/dev/null \
        | awk '/^[[:space:]]*\*.*Command Line Tools/ {
                 sub(/^[[:space:]]*\*[[:space:]]*(Label:[[:space:]]*)?/, "");
                 print
             }' \
        | tail -n1 || true)"
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

    echo "   Non-interactive install did not complete. Triggering GUI installer..."
    xcode-select --install 2>&1 || true
    echo ""
    echo "   A dialog has appeared to install the Xcode Command Line Tools."
    echo "   Complete the installation in that dialog, then return to this terminal."
    if can_read_tty; then
        # `|| true`: `read` fails on EOF/EIO even when /dev/tty opened fine;
        # under `set -e` that would abort the install instead of continuing.
        read -n 1 -s -r -p "   Press any key to continue with the rest of installation..." </dev/tty || true
    else
        echo "   Non-interactive shell detected — continuing without waiting."
    fi
    echo ""

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools installed at $(xcode-select -p)"
    else
        echo "   ERROR: Xcode Command Line Tools still not detected. Aborting."
        exit 1
    fi
}

ensure_homebrew() {
    [ "$OS" = "Darwin" ] || return 0

    if command -v brew &>/dev/null; then
        echo "   Homebrew already installed at $(command -v brew)"
        return 0
    fi

    echo ""
    echo "   Homebrew (https://brew.sh) is not installed."
    echo "   Installing it will enable KISS Sorcar to install necessary tools on demand"
    echo "   (e.g. git, cloudflared, and other runtime dependencies)."
    echo ""

    local REPLY_BREW=""
    if can_read_tty; then
        # `read` can fail (EOF/EIO) even when /dev/tty opened fine; fall back
        # to the non-interactive default instead of dying under `set -e`.
        read -r -p "   Install the latest Homebrew now? [Y/n] " REPLY_BREW </dev/tty || REPLY_BREW=""
    else
        echo "   Non-interactive shell detected — defaulting to Yes."
    fi

    case "$REPLY_BREW" in
        ""|y|Y|yes|YES|Yes)
            echo "   Installing Homebrew..."
            # `|| true`: a failed Homebrew bootstrap (no sudo, no network)
            # must not abort the install — the check below prints a warning
            # and the script continues without brew.
            if can_read_tty; then
                NONINTERACTIVE=1 /bin/bash -c \
                    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh || true)" </dev/tty || true
            else
                NONINTERACTIVE=1 /bin/bash -c \
                    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh || true)" || true
            fi
            # Make brew available in the current shell session.
            if [ -x /opt/homebrew/bin/brew ]; then
                eval "$(/opt/homebrew/bin/brew shellenv)"
            elif [ -x /usr/local/bin/brew ]; then
                eval "$(/usr/local/bin/brew shellenv)"
            fi
            if command -v brew &>/dev/null; then
                echo "   Homebrew installed at $(command -v brew)"
            else
                echo "   WARNING: Homebrew install did not complete; continuing without it."
            fi
            ;;
        *)
            echo "   Skipping Homebrew install. KISS Sorcar may not be able to install"
            echo "   some tools on demand without it."
            ;;
    esac
}

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

install_node() {
    echo "   Downloading Node.js $NODE_VERSION ..."
    local OS_NODE ARCH_NODE
    OS_NODE="$(echo "$OS" | tr '[:upper:]' '[:lower:]')"
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

find_code_cli() {
    CODE_CLI=""
    # Honor an explicit override so callers running inside a specific editor
    # distribution can force its CLI.  The Docker/code-server entrypoint sets
    # KISS_CODE_CLI=code-server so the extension is installed into
    # code-server's extensions directory
    # (~/.local/share/code-server/extensions) — the one the browser IDE
    # actually reads — instead of a separately apt-installed Microsoft VS Code
    # (~/.vscode/extensions), which code-server never loads.
    if [ -n "${KISS_CODE_CLI:-}" ]; then
        local override
        override="$(command -v "$KISS_CODE_CLI" 2>/dev/null || true)"
        if [ -n "$override" ] && [ -x "$override" ]; then
            CODE_CLI="$override"
            return 0
        fi
    fi
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

launch_vscode() {
    # ``$USER_PWD`` is captured at the top of this script before any ``cd``.
    # Passing it to VS Code makes it the workspace root so that agents
    # spawned inside the editor inherit it as their PWD.
    case "$OS" in
        Darwin)
            if open -a "Visual Studio Code" "$USER_PWD" >/dev/null 2>&1; then
                echo "Launched VS Code via 'open -a' with workspace $USER_PWD."
                return 0
            fi
            if [ -d "/Applications/Visual Studio Code.app" ] && open -a "/Applications/Visual Studio Code.app" "$USER_PWD" >/dev/null 2>&1; then
                echo "Launched VS Code from /Applications with workspace $USER_PWD."
                return 0
            fi
            ;;
        Linux)
            for candidate in \
                "$(command -v code 2>/dev/null || true)" \
                "$BIN_DIR/code" \
                "/usr/local/bin/code" \
                "/usr/bin/code" \
                "/snap/bin/code" \
                "/usr/share/code/code"; do
                if [ -n "$candidate" ] && [ -x "$candidate" ]; then
                    (nohup "$candidate" "$USER_PWD" >/dev/null 2>&1 &)
                    echo "Launched VS Code from $candidate with workspace $USER_PWD."
                    return 0
                fi
            done
            ;;
    esac

    if find_code_cli && [ -n "$CODE_CLI" ]; then
        (nohup "$CODE_CLI" "$USER_PWD" >/dev/null 2>&1 &)
        echo "Launched VS Code from $CODE_CLI with workspace $USER_PWD."
        return 0
    fi

    echo "Could not launch VS Code automatically. Open VS Code manually to finish setup."
    return 1
}

# Return 0 (true) when a VS Code window is already running.  Used to skip the
# explicit ``launch_vscode`` at the end of the install: when the editor is
# already open, the extension's own file watchers detect the reinstall (the
# overwritten ``out/extension.js`` and the freshly written
# ``~/.kiss/.extension-updated`` marker) and fire
# ``workbench.action.reloadWindow``.  That reload already brings the user back
# into a working window, so a second ``open``/launch here would only spawn a
# redundant duplicate window.
vscode_is_running() {
    case "$OS" in
        Darwin)
            # AppleScript reliably reports whether the app is running.
            if command -v osascript &>/dev/null; then
                local running
                running="$(osascript -e 'application "Visual Studio Code" is running' 2>/dev/null)"
                [ "$running" = "true" ] && return 0
            fi
            # Fallback: match the app's main process by its bundle path.
            command -v pgrep &>/dev/null && pgrep -f "Visual Studio Code.app" &>/dev/null && return 0
            ;;
        Linux)
            command -v pgrep &>/dev/null || return 1
            # The Electron main process is named "code"; also match common
            # absolute-path invocations in case the name is shortened.
            pgrep -x code &>/dev/null && return 0
            pgrep -f "/usr/share/code/code" &>/dev/null && return 0
            pgrep -f "/snap/code/" &>/dev/null && return 0
            ;;
    esac
    return 1
}

# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

# Compare two dotted version strings.  Returns 0 (true) when $1 >= $2.
version_gte() {
    local IFS=.
    # shellcheck disable=SC2206
    local i a=($1) b=($2)
    for ((i = 0; i < ${#b[@]}; i++)); do
        # Force base-10 so components with leading zeros (e.g. "08") are not
        # parsed as invalid octal, which would error out the arithmetic.
        local va=$((10#${a[i]:-0}))
        local vb=$((10#${b[i]:-0}))
        if ((va > vb)); then return 0; fi
        if ((va < vb)); then return 1; fi
    done
    return 0
}

# Ask the user whether to upgrade; abort if they decline.
#   $1 – tool display name
#   $2 – installed version
#   $3 – required version
prompt_upgrade_or_abort() {
    local name="$1" current="$2" required="$3"
    echo ""
    echo "   $name $current is older than the required version $required."
    local reply=""
    if can_read_tty; then
        # `read` can still fail even when /dev/tty opens successfully — EOF
        # when the terminal feeding it closes, or EIO when the script runs
        # detached from a usable terminal (e.g. the update button spawning
        # install.sh).  Under `set -e` an unguarded failure killed the whole
        # update right at this question with no error message; fall back to
        # the non-interactive default (Yes) instead.
        if ! read -r -p "   Upgrade $name to $required or later? [Y/n] " reply </dev/tty; then
            reply=""
            echo ""
            echo "   No interactive input available — defaulting to Yes."
        fi
    else
        echo "   Non-interactive shell detected — defaulting to Yes."
    fi
    case "$reply" in
        ""|y|Y|yes|YES|Yes) return 0 ;;
        *)
            echo ""
            echo "   ERROR: $name >= $required is required by this repository."
            echo "   Please upgrade $name manually and re-run this script."
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Upgrade helpers — invoked only when the user accepts the upgrade prompt
# ---------------------------------------------------------------------------

# Upgrade failures are deliberately non-fatal: a missing package manager or
# a flaky network must not abort the whole update (the previous behaviour —
# `exit 1` / unguarded commands under `set -e` — made the update button fail
# whenever the git-upgrade question fired in an environment without brew).
# The caller re-checks the installed version afterwards and warns if it is
# still too old.
upgrade_git() {
    echo "   Upgrading git..."
    case "$OS" in
        Darwin)
            if command -v brew &>/dev/null; then
                brew install git 2>/dev/null || brew upgrade git \
                    || echo "   WARNING: Homebrew could not upgrade git; continuing with the installed git."
            else
                echo "   WARNING: Cannot upgrade git without Homebrew; continuing with the installed git."
            fi
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -y && sudo apt-get install -y --only-upgrade git || true
            elif command -v dnf &>/dev/null; then
                sudo dnf upgrade -y git || true
            elif command -v yum &>/dev/null; then
                sudo yum update -y git || true
            elif command -v pacman &>/dev/null; then
                sudo pacman -Syu --noconfirm git || true
            elif command -v apk &>/dev/null; then
                sudo apk upgrade git || true
            else
                echo "   WARNING: No supported package manager found to upgrade git; continuing."
            fi
            ;;
    esac
    # A freshly installed git may live at a new path (e.g. /opt/homebrew/bin)
    # that bash's command hash still shadows with the old binary.
    hash -r
}

upgrade_uv() {
    echo "   Upgrading uv to $REQUIRED_UV_VERSION..."
    curl -LsSf "https://astral.sh/uv/${REQUIRED_UV_VERSION}/install.sh" | sh \
        || echo "   WARNING: uv upgrade failed; the VS Code extension will retry during setup."
    export PATH="$HOME/.local/bin:$PATH"
    hash -r
}

upgrade_node() {
    echo "   Upgrading Node.js to $NODE_VERSION..."
    install_node || echo "   WARNING: Node.js upgrade failed; continuing with the installed version."
    hash -r
}

upgrade_vscode() {
    echo "   Upgrading VS Code..."
    case "$OS" in
        Darwin)
            local ARCH_VS
            case "$ARCH" in
                aarch64|arm64) ARCH_VS="darwin-arm64" ;;
                x86_64)        ARCH_VS="darwin" ;;
            esac
            local TMP_ZIP TMP_APP_DIR
            TMP_ZIP="$(mktemp /tmp/vscode-XXXXXX.zip)"
            osascript -e 'quit app "Visual Studio Code"' 2>/dev/null || true
            sleep 2
            # Unpack into a temp dir FIRST and only then swap the app: the
            # old code removed /Applications/Visual Studio Code.app before
            # unzip, so a corrupt download crashed the script (`set -e`)
            # AND left the user with no VS Code at all.
            if curl -fsSL "https://update.code.visualstudio.com/latest/${ARCH_VS}/stable" -o "$TMP_ZIP"; then
                TMP_APP_DIR="$(mktemp -d /tmp/vscode-app-XXXXXX)"
                if unzip -q "$TMP_ZIP" -d "$TMP_APP_DIR" \
                        && [ -d "$TMP_APP_DIR/Visual Studio Code.app" ]; then
                    rm -rf "/Applications/Visual Studio Code.app"
                    mv "$TMP_APP_DIR/Visual Studio Code.app" /Applications/
                    echo "   VS Code upgraded in /Applications/"
                    local CODE_BIN="/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"
                    if [ -x "$CODE_BIN" ]; then
                        ln -sf "$CODE_BIN" "$BIN_DIR/code"
                    fi
                else
                    echo "   WARNING: Failed to unpack VS Code; continuing with the installed version."
                fi
                rm -rf "$TMP_ZIP" "$TMP_APP_DIR"
            else
                rm -f "$TMP_ZIP"
                echo "   WARNING: Failed to download VS Code; continuing with the installed version."
            fi
            ;;
        Linux)
            if command -v snap &>/dev/null; then
                sudo snap refresh code 2>&1 || true
            elif command -v apt-get &>/dev/null; then
                sudo apt-get update -y && sudo apt-get install -y --only-upgrade code 2>&1 || true
            elif command -v dnf &>/dev/null; then
                sudo dnf upgrade -y code 2>&1 || true
            else
                echo "   WARNING: Cannot upgrade VS Code automatically."
                echo "   Please upgrade from https://code.visualstudio.com if problems occur."
            fi
            ;;
    esac
    find_code_cli || true
}

upgrade_brew() {
    echo "   Updating Homebrew..."
    brew update
}

# ---------------------------------------------------------------------------
# Repo update helpers — stash local changes, pull latest, then restore them.
# ---------------------------------------------------------------------------

# Set to 1 once ``update_repo`` has stashed the working tree so that
# ``restore_stashed_changes`` knows whether there is anything to pop.
STASHED_CHANGES=0

restore_stashed_changes() {
    # Pop the stash created by ``update_repo`` so the working tree is left
    # exactly as we found it.  Wired to the EXIT trap so the unstash runs
    # "finally" — even if the install aborts midway under ``set -e``.
    if [ "$STASHED_CHANGES" = "1" ]; then
        echo ">>> Restoring stashed local changes..."
        git -C "$PROJECT_DIR" stash pop || true
        STASHED_CHANGES=0
    fi
}

update_repo() {
    # Pull the latest kiss_ai sources before building.  If the working tree is
    # dirty, stash the changes first (so ``git pull`` applies cleanly), then
    # pop them back via the EXIT trap once the install finishes.
    if ! git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
        echo "   Not a git checkout — skipping pull."
        return 0
    fi
    if [ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]; then
        echo "   Repository is dirty — stashing local changes..."
        if git -C "$PROJECT_DIR" stash push --include-untracked -m "install.sh auto-stash"; then
            STASHED_CHANGES=1
            trap restore_stashed_changes EXIT
        else
            echo "   WARNING: git stash failed; continuing without pulling."
            return 0
        fi
    fi
    echo "   Pulling latest changes..."
    # Non-fatal: offline machines must still be able to rebuild/reinstall
    # from the current checkout instead of crashing under `set -e`.
    git -C "$PROJECT_DIR" pull --ff-only || git -C "$PROJECT_DIR" pull \
        || echo "   WARNING: git pull failed (offline or diverged); continuing with the current checkout."
}

{
    echo "=== KISS Sorcar Source Install ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Directory: $PROJECT_DIR"
    echo "OS: $OS ($ARCH)"
    echo ""

    if [ "$OS" = "Darwin" ]; then
        echo ">>> Checking Xcode Command Line Tools..."
        ensure_xcode_clt
        echo ""

        echo ">>> Checking Homebrew..."
        ensure_homebrew
        echo ""
    fi

    echo ">>> [1/6] Checking git..."
    if ! command -v git &>/dev/null; then
        install_git
        hash -r
    fi
    if ! command -v git &>/dev/null; then
        echo "   ERROR: git is still not available after the install attempt."
        exit 1
    fi
    # `|| true`: under `pipefail` a git that prints no parseable version
    # would otherwise abort the script at this assignment.
    INSTALLED_GIT=$(git --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
    if [ -n "$REQUIRED_GIT_VERSION" ] && [ -n "$INSTALLED_GIT" ] && ! version_gte "$INSTALLED_GIT" "$REQUIRED_GIT_VERSION"; then
        prompt_upgrade_or_abort "git" "$INSTALLED_GIT" "$REQUIRED_GIT_VERSION"
        upgrade_git
        INSTALLED_GIT=$(git --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
        if [ -n "$INSTALLED_GIT" ] && ! version_gte "$INSTALLED_GIT" "$REQUIRED_GIT_VERSION"; then
            echo "   WARNING: git is still $INSTALLED_GIT (< $REQUIRED_GIT_VERSION); some features may not work."
        fi
    fi
    echo "   git $INSTALLED_GIT ready"
    echo ""

    echo ">>> Updating kiss_ai repository..."
    update_repo
    echo ""

    echo ">>> Checking uv..."
    if command -v uv &>/dev/null; then
        INSTALLED_UV=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
        if [ -n "$REQUIRED_UV_VERSION" ] && [ -n "$INSTALLED_UV" ] && ! version_gte "$INSTALLED_UV" "$REQUIRED_UV_VERSION"; then
            prompt_upgrade_or_abort "uv" "$INSTALLED_UV" "$REQUIRED_UV_VERSION"
            upgrade_uv
            INSTALLED_UV=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
        fi
        echo "   uv $INSTALLED_UV ready"
    else
        echo "   uv not found — will be installed by the VS Code extension"
    fi
    echo ""

    echo ">>> [2/6] Checking Node.js..."
    if ! command -v node &>/dev/null || ! command -v npm &>/dev/null || ! command -v npx &>/dev/null; then
        install_node || true
    fi
    if command -v node &>/dev/null && command -v npm &>/dev/null && command -v npx &>/dev/null; then
        INSTALLED_NODE=$(node --version 2>/dev/null | sed 's/^v//' || true)
        if [ -n "$REQUIRED_NODE_VERSION" ] && [ -n "$INSTALLED_NODE" ] && ! version_gte "$INSTALLED_NODE" "$REQUIRED_NODE_VERSION"; then
            prompt_upgrade_or_abort "Node.js" "$INSTALLED_NODE" "$REQUIRED_NODE_VERSION"
            upgrade_node
            INSTALLED_NODE=$(node --version 2>/dev/null | sed 's/^v//' || true)
        fi
        echo "   node v$INSTALLED_NODE ready"
        echo "   npm $(npm --version) ready"
    else
        echo "   ERROR: Node.js, npm, and npx are required to build the extension."
        echo "   Install Node.js from https://nodejs.org and re-run this script."
        exit 1
    fi
    echo ""

    echo ">>> [3/6] Checking VS Code CLI..."
    if ! find_code_cli; then
        install_code_cli || true
        find_code_cli || true
    fi
    if [ -n "$CODE_CLI" ]; then
        INSTALLED_VSCODE=$("$CODE_CLI" --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1 || true)
        if [ -n "$REQUIRED_VSCODE_VERSION" ] && [ -n "$INSTALLED_VSCODE" ] && ! version_gte "$INSTALLED_VSCODE" "$REQUIRED_VSCODE_VERSION"; then
            prompt_upgrade_or_abort "VS Code" "$INSTALLED_VSCODE" "$REQUIRED_VSCODE_VERSION"
            upgrade_vscode
            INSTALLED_VSCODE=$("$CODE_CLI" --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1 || true)
        fi
        echo "   code CLI ready: $CODE_CLI (v$INSTALLED_VSCODE)"
    else
        echo "   ERROR: VS Code CLI not found — cannot install the extension."
        echo "   Install VS Code from https://code.visualstudio.com and re-run this script."
        exit 1
    fi
    echo ""

    echo ">>> [4/6] Downloading official Claude Code skills..."
    # Skills are optional content — a network hiccup must not abort the update.
    bash "$PROJECT_DIR/scripts/fetch-claude-skills.sh" \
        || echo "   WARNING: Claude skills download failed; continuing."
    echo ""

    echo ">>> [5/6] Building VS Code extension..."
    VSCODE_EXT_DIR="$PROJECT_DIR/src/kiss/agents/vscode"
    VSIX="$VSCODE_EXT_DIR/kiss-sorcar.vsix"
    cd "$VSCODE_EXT_DIR"
    # --ignore-scripts: the lockfile's only packages with install scripts are
    # `keytar` (an *optional*, lazily-imported dep of @vscode/vsce used solely
    # for publish credentials — never by `vsce package`) and
    # `@vscode/vsce-sign` (signing only).  keytar's install script runs
    # `prebuild-install || node-gyp rebuild`, which downloads from the
    # archived atom/node-keytar GitHub releases (or compiles natively) and can
    # block forever with no output — hanging the Update button's install at
    # "[5/6] Building VS Code extension..." right after npm's deprecation
    # warnings.  Neither script is needed to compile and package the VSIX.
    # --no-audit --no-fund skip more network round-trips and noise.
    npm ci --ignore-scripts --no-audit --no-fund
    npm run package
    cd "$PROJECT_DIR"
    if [ ! -f "$VSIX" ]; then
        echo "   ERROR: Failed to build VSIX"
        exit 1
    fi
    echo "   Built $VSIX"
    echo ""

    echo ">>> [6/6] Installing VS Code extension..."
    if ! "$CODE_CLI" --install-extension "$VSIX" --force 2>&1; then
        echo "   ERROR: '$CODE_CLI --install-extension' failed; the update was not applied."
        exit 1
    fi
    echo "   Extension installed into VS Code"
    # ``kiss-sorcar.vsix`` is a build artifact and MUST NOT be committed
    # to git — it is ~2 MB of binary that bloats the history and is
    # rebuilt deterministically by the ``npm run package`` step above.
    # The file is matched by ``*.vsix`` in the repo's ``.gitignore`` so
    # ``git status`` never lists it and the auto-commit / ``git add .``
    # flows cannot pick it up.  We deliberately do NOT delete the VSIX
    # here so that subsequent ``code --install-extension`` invocations
    # (e.g. a manual retry) can reuse the freshly built artifact without
    # rebuilding it.  As a defence-in-depth check, refuse to continue if
    # the VSIX has somehow become tracked again (e.g. a ``git add -f``
    # by mistake) — that would cause the worktree flow to commit it.
    if git -C "$PROJECT_DIR" ls-files --error-unmatch "$VSIX" &>/dev/null; then
        echo "   ERROR: $VSIX is tracked by git but must remain ignored." >&2
        echo "   Run: git -C \"$PROJECT_DIR\" rm --cached \"$VSIX\"" >&2
        echo "   and ensure ``*.vsix`` stays in .gitignore." >&2
        exit 1
    fi

    # Stop the old kiss-web daemon BEFORE the extension auto-reloads.  The
    # ``--install-extension --force`` above replaced the extension directory
    # tree that the running daemon's bundled kiss_project (.venv/bin/kiss-web)
    # was loaded from, so the live daemon is technically broken even while it
    # is still listening (subsequent UDS requests can hit stale / missing
    # module paths).  If we skip this, the marker write below triggers
    # ``workbench.action.reloadWindow`` (~2 s later) and the freshly reloaded
    # webview reconnects to the broken daemon — the chat view comes up blank.
    #
    # ``pkill -x kiss-web`` is unreliable on macOS — kiss-web is a Python
    # shebang script so the kernel's ``comm`` field is the (15-char-truncated)
    # interpreter path, NOT the literal name.  Kill by listening port instead.
    # The macOS LaunchAgent / Linux systemd unit's ``KeepAlive`` respawns a
    # clean daemon from the freshly-installed VSIX before the reload fires.
    # Refuse to clobber a kiss-web daemon that is mid-task.  Without this
    # guard the SIGTERM block below silently kills any in-flight agent run
    # — the regression that turned task_history rows 3233/3234 into
    # ``"Task interrupted by server restart/shutdown"``.  The check mirrors
    # the bash-side guard in ``scripts/build-extension.sh`` and the
    # ``daemonHasActiveTasks``-based guard in
    # ``src/kiss/agents/vscode/src/DependencyInstaller.ts``.  Override with
    # ``KISS_FORCE_RESTART=1`` for a knowingly destructive re-install.
    if command -v python3 &>/dev/null; then
        if ! python3 "$PROJECT_DIR/scripts/check-kiss-web-active-tasks.py"; then
            if [ "${KISS_FORCE_RESTART:-}" = "1" ]; then
                echo "   KISS_FORCE_RESTART=1 set; proceeding despite active tasks."
            else
                echo "   Aborting install.sh: kiss-web has in-flight tasks." >&2
                echo "   Wait for them to finish, or rerun with KISS_FORCE_RESTART=1." >&2
                exit 3
            fi
        fi
    fi
    if command -v lsof &>/dev/null; then
        OLD_KISS_WEB_PIDS=$(lsof -ti :8787 2>/dev/null || true)
        if [ -n "$OLD_KISS_WEB_PIDS" ]; then
            echo "   Stopping old kiss-web daemon (PIDs: $OLD_KISS_WEB_PIDS)..."
            echo "$OLD_KISS_WEB_PIDS" | xargs kill 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10; do
                sleep 0.3
                if ! lsof -i :8787 -t &>/dev/null; then break; fi
            done
            # Force-kill survivors.
            KISS_WEB_STRAGGLERS=$(lsof -ti :8787 2>/dev/null || true)
            if [ -n "$KISS_WEB_STRAGGLERS" ]; then
                echo "$KISS_WEB_STRAGGLERS" | xargs kill -9 2>/dev/null || true
            fi
        fi
    fi
    # Remove the stale Unix-domain socket left behind by the now-dead daemon.
    # The new daemon's ``_setup_server`` unlinks before binding, but pre-emptive
    # cleanup avoids ENOENT/ECONNREFUSED reconnect-loop noise from the extension
    # client that is mid-flight during the launchd/systemd respawn window.
    rm -f "$HOME/.kiss/sorcar.sock"

    # Copy the bundled MODEL_INFO.json into ~/.kiss/ so the freshly installed
    # extension serves the latest model pricing/context table on startup
    # without waiting for ``model_info.py``'s lazy auto-copy.  The Python
    # loader in ``kiss.core.models.model_info`` also auto-refreshes when the
    # package copy is newer, but doing it here makes the data the user sees
    # match the just-installed source immediately.
    MODEL_INFO_SRC="$PROJECT_DIR/src/kiss/core/models/MODEL_INFO.json"
    MODEL_INFO_DST="$HOME/.kiss/MODEL_INFO.json"
    if [ -f "$MODEL_INFO_SRC" ]; then
        cp "$MODEL_INFO_SRC" "$MODEL_INFO_DST"
        echo "   Installed MODEL_INFO.json at $MODEL_INFO_DST"
    else
        echo "   WARNING: $MODEL_INFO_SRC missing — model table not refreshed."
    fi

    date -u +%Y-%m-%dT%H:%M:%SZ > "$HOME/.kiss/.extension-updated"
    # Remove any stale source-install marker from older versions of this
    # installer.  The extension now always runs against the kiss_project
    # bundled inside the VSIX, so the marker is no longer consulted and
    # leaving it around would only mislead troubleshooting.
    rm -f "$HOME/.kiss/install_dir"
    echo ""

    echo "=== Source bootstrap complete ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Project: $PROJECT_DIR"
    echo ""
    echo "KISS Sorcar runtime setup will finish inside VS Code."
    echo "The extension will install/check uv, Python dependencies, Playwright,"
    echo "cloudflared, shell PATH entries, API keys, remote access auth, and kiss-web."
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to $LOG_FILE"
# Only explicitly launch VS Code when it is not already running.  If a window
# is already open, the extension's watchers on ``out/extension.js`` and
# ``~/.kiss/.extension-updated`` (both touched in step [6/6]) trigger
# ``workbench.action.reloadWindow`` to pick up the update — launching here too
# would open a redundant second window.
if [ -n "${KISS_SKIP_LAUNCH:-}" ]; then
    # The caller (e.g. scripts/docker-startup.sh) owns launching the editor —
    # typically because it will start code-server itself right after this
    # script returns.  Launching here too would bind the same port and make
    # the caller's code-server fail with EADDRINUSE, crashing the container.
    echo "KISS_SKIP_LAUNCH set; skipping VS Code launch (caller will start the editor)."
elif vscode_is_running; then
    echo "VS Code is already running; the extension will reload to finish setup."
    echo "Skipping explicit launch to avoid opening a duplicate window."
else
    echo "Launching VS Code to finish setup..."
    launch_vscode || true
fi
