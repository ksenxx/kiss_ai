#!/bin/bash
#
# Build a standalone macOS offline installer package (.pkg) for the KISS project.
# Bundles: uv, code-server (with node), Python 3.13, git, Playwright Chromium,
# all Python wheels, and the project source.
#
# Usage: ./scripts/build_offline_pkg.sh
# Output: dist/kiss-offline-installer.pkg
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="$PROJECT_ROOT/.kiss.artifacts/tmp/offline-pkg"
PAYLOAD="$STAGE/payload"
SCRIPTS="$STAGE/scripts"
PKG_ID="com.kiss.offline-installer"
PKG_VERSION="1.0.0"
OUTPUT="$PROJECT_ROOT/dist/kiss-offline-installer.pkg"
ARCH="$(uname -m)"  # arm64 or x86_64

echo "=== Building KISS Offline Installer Package ==="
echo "Architecture: $ARCH"
echo "Staging: $STAGE"

# Clean staging
rm -rf "$STAGE"
mkdir -p "$PAYLOAD/kiss-offline" "$SCRIPTS"

BUNDLE="$PAYLOAD/kiss-offline"

# ---------------------------------------------------------------------------
# 1. uv binary
# ---------------------------------------------------------------------------
echo ">>> Bundling uv..."
mkdir -p "$BUNDLE/bin"
UV_BIN="$(which uv)"
cp "$UV_BIN" "$BUNDLE/bin/uv"
chmod +x "$BUNDLE/bin/uv"
# Also copy uvx if it exists
if [ -f "$(dirname "$UV_BIN")/uvx" ]; then
    cp "$(dirname "$UV_BIN")/uvx" "$BUNDLE/bin/uvx"
    chmod +x "$BUNDLE/bin/uvx"
fi
echo "   uv: $(du -sh "$BUNDLE/bin/uv" | cut -f1)"

# ---------------------------------------------------------------------------
# 2. code-server (standalone release with bundled node)
# ---------------------------------------------------------------------------
echo ">>> Bundling code-server..."
CS_VERSION="4.111.0"
CS_TARBALL="code-server-${CS_VERSION}-macos-${ARCH}.tar.gz"
CS_URL="https://github.com/coder/code-server/releases/download/v${CS_VERSION}/${CS_TARBALL}"
CS_CACHE="$STAGE/cache/$CS_TARBALL"
mkdir -p "$STAGE/cache"

if [ ! -f "$CS_CACHE" ]; then
    echo "   Downloading code-server ${CS_VERSION} for ${ARCH}..."
    curl -fSL -o "$CS_CACHE" "$CS_URL"
fi
echo "   Extracting code-server..."
mkdir -p "$BUNDLE/code-server"
tar xzf "$CS_CACHE" -C "$BUNDLE/code-server" --strip-components=1
echo "   code-server: $(du -sh "$BUNDLE/code-server" | cut -f1)"

# ---------------------------------------------------------------------------
# 3. Python 3.13 standalone (from uv's cache)
# ---------------------------------------------------------------------------
echo ">>> Bundling Python 3.13 standalone..."
PYTHON_SRC="$HOME/.local/share/uv/python"
PYTHON_DIR=$(ls -d "$PYTHON_SRC"/cpython-3.13*-macos-aarch64-none 2>/dev/null | head -1)
if [ -z "$PYTHON_DIR" ]; then
    PYTHON_DIR=$(ls -d "$PYTHON_SRC"/cpython-3.13*-macos-x86_64-none 2>/dev/null | head -1)
fi
if [ -z "$PYTHON_DIR" ]; then
    echo "   Python 3.13 not found in uv cache, fetching..."
    uv python install 3.13
    PYTHON_DIR=$(ls -d "$PYTHON_SRC"/cpython-3.13*-macos-* 2>/dev/null | head -1)
fi
PYTHON_DIRNAME="$(basename "$PYTHON_DIR")"
echo "   Copying from $PYTHON_DIR ($PYTHON_DIRNAME)..."
cp -R "$PYTHON_DIR" "$BUNDLE/python"
# Save the original directory name so the installer can restore it
echo "$PYTHON_DIRNAME" > "$BUNDLE/python-dirname.txt"
echo "   Python: $(du -sh "$BUNDLE/python" | cut -f1)"

# ---------------------------------------------------------------------------
# 4. Git (from Xcode CLT - portable binary + git-core)
# ---------------------------------------------------------------------------
echo ">>> Bundling git..."
GIT_BIN="/Library/Developer/CommandLineTools/usr/bin/git"
GIT_CORE="/Library/Developer/CommandLineTools/usr/libexec/git-core"
if [ -f "$GIT_BIN" ]; then
    mkdir -p "$BUNDLE/git/bin" "$BUNDLE/git/libexec"
    cp "$GIT_BIN" "$BUNDLE/git/bin/git"
    chmod +x "$BUNDLE/git/bin/git"
    # Copy git-core helpers
    cp -R "$GIT_CORE" "$BUNDLE/git/libexec/git-core"
    # Also copy git-remote-https and other needed helpers from bin
    for helper in git-remote-https git-remote-http git-receive-pack git-upload-pack git-upload-archive; do
        if [ -f "/Library/Developer/CommandLineTools/usr/bin/$helper" ]; then
            cp "/Library/Developer/CommandLineTools/usr/bin/$helper" "$BUNDLE/git/bin/$helper"
        fi
    done
    echo "   git: $(du -sh "$BUNDLE/git" | cut -f1)"
else
    echo "   WARNING: Xcode CLT git not found at $GIT_BIN, git not bundled"
fi

# ---------------------------------------------------------------------------
# 5. Python wheels (offline pip cache)
# ---------------------------------------------------------------------------
echo ">>> Downloading Python wheels for offline install..."
mkdir -p "$BUNDLE/wheels"
cd "$PROJECT_ROOT"
# Ensure pip is available for downloading wheels
uv pip install pip 2>/dev/null || true
# Export requirements from uv lock
uv export --format requirements.txt --no-dev --no-hashes > "$STAGE/requirements.txt"
# Remove the -e . line (we'll install the project separately)
sed -i '' '/^-e \./d' "$STAGE/requirements.txt"
# Download all dependency wheels
uv run python -m pip download --dest "$BUNDLE/wheels" -r "$STAGE/requirements.txt"
# Build the project wheel so the installer doesn't need build tools
uv build --wheel --out-dir "$BUNDLE/wheels"
echo "   wheels: $(du -sh "$BUNDLE/wheels" | cut -f1) ($(ls "$BUNDLE/wheels" | wc -l | tr -d ' ') files)"

# ---------------------------------------------------------------------------
# 6. Playwright Chromium browser
# ---------------------------------------------------------------------------
echo ">>> Bundling Playwright Chromium..."
PW_CACHE="$HOME/Library/Caches/ms-playwright"
mkdir -p "$BUNDLE/playwright"
for browser_dir in "$PW_CACHE"/chromium-* "$PW_CACHE"/chromium_headless_shell-* "$PW_CACHE"/ffmpeg-*; do
    if [ -d "$browser_dir" ]; then
        bname="$(basename "$browser_dir")"
        echo "   Copying $bname..."
        cp -R "$browser_dir" "$BUNDLE/playwright/$bname"
    fi
done
echo "   playwright: $(du -sh "$BUNDLE/playwright" | cut -f1)"

# ---------------------------------------------------------------------------
# 7. Project source
# ---------------------------------------------------------------------------
echo ">>> Bundling project source..."
mkdir -p "$BUNDLE/project"
# Copy essential project files (excluding .git, venv, artifacts, etc.)
rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='.kiss.artifacts' --exclude='htmlcov*' --exclude='.coverage*' \
    --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.pytest_cache' \
    --exclude='node_modules' --exclude='dist' \
    "$PROJECT_ROOT/" "$BUNDLE/project/"
echo "   project: $(du -sh "$BUNDLE/project" | cut -f1)"

# ---------------------------------------------------------------------------
# 8. Create the offline install script (runs as postinstall in .pkg)
# ---------------------------------------------------------------------------
echo ">>> Creating install script..."
cat > "$BUNDLE/install-offline.sh" << 'INSTALL_SCRIPT'
#!/bin/bash
#
# KISS Offline Installer
# Installs all bundled dependencies without internet access.
#
set -euo pipefail

KISS_BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_BASE="${KISS_INSTALL_DIR:-$HOME/.kiss-install}"
PROJECT_DIR="${KISS_PROJECT_DIR:-$HOME/kiss_ai}"

echo "=== KISS Offline Installer ==="
echo "Bundle: $KISS_BUNDLE_DIR"
echo "Install base: $INSTALL_BASE"
echo "Project dir: $PROJECT_DIR"

mkdir -p "$INSTALL_BASE/bin"

# 1. Install uv
echo ">>> Installing uv..."
cp "$KISS_BUNDLE_DIR/bin/uv" "$INSTALL_BASE/bin/uv"
chmod +x "$INSTALL_BASE/bin/uv"
if [ -f "$KISS_BUNDLE_DIR/bin/uvx" ]; then
    cp "$KISS_BUNDLE_DIR/bin/uvx" "$INSTALL_BASE/bin/uvx"
    chmod +x "$INSTALL_BASE/bin/uvx"
fi
# Also install to ~/.local/bin for standard uv location
mkdir -p "$HOME/.local/bin"
cp "$INSTALL_BASE/bin/uv" "$HOME/.local/bin/uv"
[ -f "$INSTALL_BASE/bin/uvx" ] && cp "$INSTALL_BASE/bin/uvx" "$HOME/.local/bin/uvx"

# 2. Install code-server
echo ">>> Installing code-server..."
mkdir -p "$INSTALL_BASE/code-server"
cp -R "$KISS_BUNDLE_DIR/code-server/"* "$INSTALL_BASE/code-server/"
chmod +x "$INSTALL_BASE/code-server/bin/code-server"
# Symlink to bin
ln -sf "$INSTALL_BASE/code-server/bin/code-server" "$INSTALL_BASE/bin/code-server"

# 3. Install Python 3.13 standalone
echo ">>> Installing Python 3.13..."
PYTHON_DIRNAME="$(cat "$KISS_BUNDLE_DIR/python-dirname.txt")"
PYTHON_DEST="$HOME/.local/share/uv/python/$PYTHON_DIRNAME"
mkdir -p "$(dirname "$PYTHON_DEST")"
if [ ! -d "$PYTHON_DEST" ]; then
    cp -R "$KISS_BUNDLE_DIR/python" "$PYTHON_DEST"
fi

# 4. Install git
echo ">>> Installing git..."
if [ -d "$KISS_BUNDLE_DIR/git" ]; then
    mkdir -p "$INSTALL_BASE/git"
    cp -R "$KISS_BUNDLE_DIR/git/"* "$INSTALL_BASE/git/"
    chmod +x "$INSTALL_BASE/git/bin/git"
    ln -sf "$INSTALL_BASE/git/bin/git" "$INSTALL_BASE/bin/git"
    # Set GIT_EXEC_PATH for the installed git
    export GIT_EXEC_PATH="$INSTALL_BASE/git/libexec/git-core"
fi

# 5. Install Playwright browsers
echo ">>> Installing Playwright browsers..."
PW_DEST="$HOME/Library/Caches/ms-playwright"
mkdir -p "$PW_DEST"
for browser_dir in "$KISS_BUNDLE_DIR/playwright/"*/; do
    browser_name="$(basename "$browser_dir")"
    if [ ! -d "$PW_DEST/$browser_name" ]; then
        echo "   Installing $browser_name..."
        cp -R "$browser_dir" "$PW_DEST/$browser_name"
    fi
done

# 6. Set up the project
echo ">>> Setting up project..."
export PATH="$INSTALL_BASE/bin:$HOME/.local/bin:$PATH"
if [ -d "$KISS_BUNDLE_DIR/project" ]; then
    mkdir -p "$PROJECT_DIR"
    cp -R "$KISS_BUNDLE_DIR/project/"* "$PROJECT_DIR/"
    cd "$PROJECT_DIR"
    
    # Create venv with offline Python (--clear to handle re-installs)
    "$INSTALL_BASE/bin/uv" venv --python 3.13 --clear
    
    # Install from local wheels (fully offline, including pre-built project wheel)
    # Explicitly target the project venv to avoid uv resolving a different workspace
    "$INSTALL_BASE/bin/uv" pip install --python "$PROJECT_DIR/.venv" --no-index --find-links "$KISS_BUNDLE_DIR/wheels" kiss-agent-framework
fi

# 7. Create shell profile additions
PROFILE_SNIPPET="$INSTALL_BASE/env.sh"
cat > "$PROFILE_SNIPPET" << EOF
# KISS Agent Framework - added by offline installer
export PATH="$INSTALL_BASE/bin:\$HOME/.local/bin:\$PATH"
export GIT_EXEC_PATH="$INSTALL_BASE/git/libexec/git-core"
EOF

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To use KISS, add the following to your shell profile (~/.zshrc or ~/.bashrc):"
echo ""
echo "  source $PROFILE_SNIPPET"
echo ""
echo "Then set your API keys:"
echo "  export ANTHROPIC_API_KEY=your_key_here"
echo "  export GEMINI_API_KEY=your_key_here"
echo ""
echo "Project installed at: $PROJECT_DIR"
echo "Run: cd $PROJECT_DIR && source .venv/bin/activate && sorcar"
INSTALL_SCRIPT

chmod +x "$BUNDLE/install-offline.sh"

# ---------------------------------------------------------------------------
# 9. Create the .pkg postinstall script
# ---------------------------------------------------------------------------
echo ">>> Creating package scripts..."
cat > "$SCRIPTS/postinstall" << 'POSTINSTALL'
#!/bin/bash
# macOS .pkg postinstall script
# $2 = install location (e.g., /usr/local or /)
set -euo pipefail

# The payload is installed to /usr/local/kiss-offline by the pkg
BUNDLE="/usr/local/kiss-offline"

# Detect the real user: prefer SUDO_USER, then console owner, then USER
if [ "$(id -u)" = "0" ]; then
    TARGET_USER="${SUDO_USER:-$(stat -f '%Su' /dev/console 2>/dev/null || echo root)}"
else
    TARGET_USER="${USER:-$(whoami)}"
fi
TARGET_HOME=$(eval echo "~$TARGET_USER")

echo "Installing KISS for user: $TARGET_USER (home: $TARGET_HOME)"

# Run the install script as the target user
export KISS_INSTALL_DIR="$TARGET_HOME/.kiss-install"
export KISS_PROJECT_DIR="$TARGET_HOME/kiss_ai"
export HOME="$TARGET_HOME"

if [ "$(id -u)" = "0" ]; then
    sudo -u "$TARGET_USER" \
        KISS_INSTALL_DIR="$KISS_INSTALL_DIR" \
        KISS_PROJECT_DIR="$KISS_PROJECT_DIR" \
        HOME="$TARGET_HOME" \
        bash "$BUNDLE/install-offline.sh"
else
    bash "$BUNDLE/install-offline.sh"
fi

echo "KISS offline installation complete!"
POSTINSTALL
chmod +x "$SCRIPTS/postinstall"

# ---------------------------------------------------------------------------
# 10. Build the .pkg
# ---------------------------------------------------------------------------
echo ">>> Building .pkg..."
mkdir -p "$(dirname "$OUTPUT")"

# Build component package
COMPONENT_PKG="$STAGE/kiss-component.pkg"
pkgbuild \
    --root "$PAYLOAD" \
    --identifier "$PKG_ID" \
    --version "$PKG_VERSION" \
    --install-location "/usr/local" \
    --scripts "$SCRIPTS" \
    "$COMPONENT_PKG"

# Create distribution XML for productbuild
cat > "$STAGE/distribution.xml" << DIST_XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>KISS Agent Framework (Offline)</title>
    <organization>com.kiss</organization>
    <domains enable_localSystem="true" enable_currentUserHome="true"/>
    <options customize="never" require-scripts="true" rootVolumeOnly="false"/>
    <welcome language="en" mime-type="text/plain"><![CDATA[
KISS Agent Framework - Offline Installer

This package installs all dependencies needed to run KISS Agent Framework
without an internet connection:

  • uv (Python package manager)
  • code-server (VS Code in the browser)
  • Python 3.13
  • Git
  • Playwright Chromium browser
  • All Python dependencies
  • KISS project source

After installation, add to your shell profile:
  source ~/.kiss-install/env.sh

Then set your API keys:
  export ANTHROPIC_API_KEY=your_key
  export GEMINI_API_KEY=your_key
]]></welcome>
    <choices-outline>
        <line choice="default">
            <line choice="com.kiss.offline-installer"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="com.kiss.offline-installer" visible="false">
        <pkg-ref id="com.kiss.offline-installer"/>
    </choice>
    <pkg-ref id="com.kiss.offline-installer" version="${PKG_VERSION}" onConclusion="none">kiss-component.pkg</pkg-ref>
</installer-gui-script>
DIST_XML

# Build product archive
productbuild \
    --distribution "$STAGE/distribution.xml" \
    --package-path "$STAGE" \
    "$OUTPUT"

echo ""
echo "=== Package Built Successfully ==="
echo "Output: $OUTPUT"
echo "Size: $(du -sh "$OUTPUT" | cut -f1)"
echo ""
echo "To install: open $OUTPUT"
echo "Or: sudo installer -pkg $OUTPUT -target /"
