#!/bin/bash
#
# Build a standalone macOS offline installer package (.pkg) for the KISS project.
# Bundles: uv, code-server (with node), Python 3.13,
# Playwright Chromium, all Python wheels, and the project source.
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
PKG_VERSION=$(grep '__version__' "$PROJECT_ROOT/src/kiss/_version.py" | sed 's/.*"\(.*\)".*/\1/')
OUTPUT="$PROJECT_ROOT/dist/kiss-offline-installer.pkg"
# Detect actual hardware architecture (immune to Rosetta translation)
if sysctl -n hw.optional.arm64 2>/dev/null | grep -q '1'; then
    ARCH="arm64"
else
    ARCH="x86_64"
fi

echo "=== Building KISS Offline Installer Package ==="
echo "Architecture: $ARCH"
echo "Staging: $STAGE"

# ---------------------------------------------------------------------------
# Helper: clear macOS quarantine/provenance attributes and re-sign binaries.
# macOS Sequoia kills ad-hoc signed binaries that carry provenance/quarantine
# attributes.  Removing the xattrs alone is insufficient because the OS caches
# the original quarantine decision per-path; re-signing forces re-evaluation.
# ---------------------------------------------------------------------------
strip_quarantine() {
    local target="$1"
    xattr -d com.apple.quarantine "$target" 2>/dev/null || true
    xattr -d com.apple.provenance "$target" 2>/dev/null || true
    codesign --force --sign - "$target" 2>/dev/null || true
}

# Ensure the local uv binary can run during this build
strip_quarantine "$(which uv)"

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
strip_quarantine "$BUNDLE/bin/uv"
# Also copy uvx if it exists
if [ -f "$(dirname "$UV_BIN")/uvx" ]; then
    cp "$(dirname "$UV_BIN")/uvx" "$BUNDLE/bin/uvx"
    chmod +x "$BUNDLE/bin/uvx"
    strip_quarantine "$BUNDLE/bin/uvx"
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
# Strip quarantine from code-server executables
find "$BUNDLE/code-server" -type f -perm +111 -exec sh -c 'xattr -d com.apple.quarantine "$1" 2>/dev/null; xattr -d com.apple.provenance "$1" 2>/dev/null; codesign --force --sign - "$1" 2>/dev/null; true' _ {} \;
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
# Strip quarantine from Python executables
find "$BUNDLE/python" -type f -perm +111 -exec sh -c 'xattr -d com.apple.quarantine "$1" 2>/dev/null; xattr -d com.apple.provenance "$1" 2>/dev/null; codesign --force --sign - "$1" 2>/dev/null; true' _ {} \;
echo "   Python: $(du -sh "$BUNDLE/python" | cut -f1)"

# ---------------------------------------------------------------------------
# 4. Python wheels (offline pip cache)
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
# 5. Playwright Chromium browser
# ---------------------------------------------------------------------------
echo ">>> Bundling Playwright Chromium..."
PW_BROWSERS="$HOME/Library/Caches/ms-playwright"
if [ -d "$PW_BROWSERS" ]; then
    mkdir -p "$BUNDLE/playwright-browsers"
    # Copy chromium, chromium_headless_shell, ffmpeg, and .links
    for item in "$PW_BROWSERS"/chromium-* "$PW_BROWSERS"/chromium_headless_shell-* "$PW_BROWSERS"/ffmpeg-*; do
        if [ -d "$item" ]; then
            cp -R "$item" "$BUNDLE/playwright-browsers/"
            echo "   $(basename "$item"): $(du -sh "$item" | cut -f1)"
        fi
    done
    # Copy the .links directory
    if [ -d "$PW_BROWSERS/.links" ]; then
        cp -R "$PW_BROWSERS/.links" "$BUNDLE/playwright-browsers/.links"
    fi
    # Create marker files (INSTALLATION_COMPLETE, DEPENDENCIES_VALIDATED already inside dirs)
    echo "   playwright-browsers: $(du -sh "$BUNDLE/playwright-browsers" | cut -f1)"
else
    echo "   WARNING: Playwright browsers not found at $PW_BROWSERS"
    echo "   Run 'playwright install chromium' first, then re-run this script."
fi

# ---------------------------------------------------------------------------
# 6. Project source + build architecture stamp
# ---------------------------------------------------------------------------
echo "$ARCH" > "$BUNDLE/build-arch.txt"
echo ">>> Bundling project source..."
# Note: Section numbering continues: 8=install script, 9=pkg postinstall, 10=build .pkg
mkdir -p "$BUNDLE/project"
# Copy essential project files (excluding .git, venv, artifacts, etc.)
rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='.kiss.artifacts' --exclude='htmlcov*' --exclude='.coverage*' \
    --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.pytest_cache' \
    --exclude='node_modules' --exclude='dist' --exclude='nohup.out' \
    "$PROJECT_ROOT/" "$BUNDLE/project/"
# Safety net: remove nohup.out if it slipped through (e.g. created by a concurrent process)
find "$BUNDLE/project" -name 'nohup.out' -delete 2>/dev/null || true
echo "   project: $(du -sh "$BUNDLE/project" | cut -f1)"

# ---------------------------------------------------------------------------
# 7. Create the offline install script (runs as postinstall in .pkg)
# ---------------------------------------------------------------------------
echo ">>> Creating install script..."
cat > "$BUNDLE/install-offline.sh" << 'INSTALL_SCRIPT'
#!/bin/bash
#
# KISS Offline Installer
# Installs all bundled dependencies without internet access.
#
set -euo pipefail

# macOS Sequoia kills ad-hoc signed binaries that carry quarantine/provenance attrs.
# Removing xattrs alone is insufficient; re-signing forces macOS to re-evaluate.
_strip_quarantine() {
    xattr -d com.apple.quarantine "$1" 2>/dev/null || true
    xattr -d com.apple.provenance "$1" 2>/dev/null || true
    codesign --force --sign - "$1" 2>/dev/null || true
}

KISS_BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

# Check architecture matches
BUILD_ARCH="$(cat "$KISS_BUNDLE_DIR/build-arch.txt" 2>/dev/null || echo unknown)"
# Detect actual hardware architecture (immune to Rosetta translation)
if sysctl -n hw.optional.arm64 2>/dev/null | grep -q '1'; then
    HOST_ARCH="arm64"
else
    HOST_ARCH="x86_64"
fi
if [ "$BUILD_ARCH" != "$HOST_ARCH" ]; then
    echo "ERROR: This package was built for $BUILD_ARCH but this machine is $HOST_ARCH."
    echo "       Please use a package built for $HOST_ARCH."
    exit 1
fi

# Check available disk space (need ~3 GB for installation)
_REQUIRED_MB=3000
_AVAIL_MB=$(df -m "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "$_AVAIL_MB" ] && [ "$_AVAIL_MB" -lt "$_REQUIRED_MB" ] 2>/dev/null; then
    echo "ERROR: Insufficient disk space. Need ${_REQUIRED_MB}MB, only ${_AVAIL_MB}MB available."
    exit 1
fi

# Kill any existing KISS processes that might interfere with reinstallation
_kill_existing() {
    local _pids
    # Kill code-server instances started by KISS (match the cs- data dir pattern)
    _pids=$(pgrep -f 'code-server.*\.kiss/cs-' 2>/dev/null || true)
    if [ -n "$_pids" ]; then
        echo "   Stopping existing code-server instances..."
        echo "$_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi
    # Kill sorcar/kiss processes from the install directory
    _pids=$(pgrep -f 'kiss_ai.*sorcar\|kiss_ai.*kiss' 2>/dev/null || true)
    if [ -n "$_pids" ]; then
        echo "   Stopping existing KISS processes..."
        echo "$_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi
}
_kill_existing

# Determine install location: env var > interactive prompt > default
_DEFAULT_DIR="$HOME/kiss_ai"
if [ -n "${KISS_INSTALL_DIR:-}" ]; then
    INSTALL_BASE="$KISS_INSTALL_DIR"
elif [ -t 0 ]; then
    printf "Install location [%s]: " "$_DEFAULT_DIR"
    read -r _USER_DIR
    INSTALL_BASE="${_USER_DIR:-$_DEFAULT_DIR}"
else
    INSTALL_BASE="$_DEFAULT_DIR"
fi
PROJECT_DIR="${KISS_PROJECT_DIR:-$INSTALL_BASE}"

echo "=== KISS Offline Installer ==="
echo "Bundle: $KISS_BUNDLE_DIR"
echo "Install base: $INSTALL_BASE"
echo "Project dir: $PROJECT_DIR"

mkdir -p "$INSTALL_BASE/bin"

# Write install location marker so the Python env module can find it
mkdir -p "$HOME/.kiss"
printf '%s\n' "$INSTALL_BASE" > "$HOME/.kiss/install_dir"

# 1. Install uv
echo ">>> Installing uv..."
cp "$KISS_BUNDLE_DIR/bin/uv" "$INSTALL_BASE/bin/uv"
chmod +x "$INSTALL_BASE/bin/uv"
_strip_quarantine "$INSTALL_BASE/bin/uv"
if [ -f "$KISS_BUNDLE_DIR/bin/uvx" ]; then
    cp "$KISS_BUNDLE_DIR/bin/uvx" "$INSTALL_BASE/bin/uvx"
    chmod +x "$INSTALL_BASE/bin/uvx"
    _strip_quarantine "$INSTALL_BASE/bin/uvx"
fi
# 2. Install code-server
echo ">>> Installing code-server..."
rm -rf "$INSTALL_BASE/code-server"
mkdir -p "$INSTALL_BASE/code-server"
cp -R "$KISS_BUNDLE_DIR/code-server/"* "$INSTALL_BASE/code-server/"
chmod +x "$INSTALL_BASE/code-server/bin/code-server"
find "$INSTALL_BASE/code-server" -type f -perm +111 -exec sh -c 'xattr -d com.apple.quarantine "$1" 2>/dev/null; xattr -d com.apple.provenance "$1" 2>/dev/null; codesign --force --sign - "$1" 2>/dev/null; true' _ {} \;
# Symlink to bin
ln -sf "$INSTALL_BASE/code-server/bin/code-server" "$INSTALL_BASE/bin/code-server"

# 3. Install Python 3.13 standalone (inside $INSTALL_BASE/python/)
echo ">>> Installing Python 3.13..."
PYTHON_DIRNAME="$(cat "$KISS_BUNDLE_DIR/python-dirname.txt")"
PYTHON_DEST="$INSTALL_BASE/python/$PYTHON_DIRNAME"
rm -rf "$PYTHON_DEST"
mkdir -p "$(dirname "$PYTHON_DEST")"
cp -R "$KISS_BUNDLE_DIR/python" "$PYTHON_DEST"
find "$PYTHON_DEST" -type f -perm +111 -exec sh -c 'xattr -d com.apple.quarantine "$1" 2>/dev/null; xattr -d com.apple.provenance "$1" 2>/dev/null; codesign --force --sign - "$1" 2>/dev/null; true' _ {} \;

# 4. Install Playwright Chromium browsers (inside $INSTALL_BASE/playwright-browsers/)
echo ">>> Installing Playwright Chromium..."
PW_DEST="$INSTALL_BASE/playwright-browsers"
if [ -d "$KISS_BUNDLE_DIR/playwright-browsers" ]; then
    rm -rf "$PW_DEST"
    mkdir -p "$PW_DEST"
    for item in "$KISS_BUNDLE_DIR/playwright-browsers"/chromium-* \
                "$KISS_BUNDLE_DIR/playwright-browsers"/chromium_headless_shell-* \
                "$KISS_BUNDLE_DIR/playwright-browsers"/ffmpeg-*; do
        if [ -d "$item" ]; then
            dirname="$(basename "$item")"
            cp -R "$item" "$PW_DEST/$dirname"
            echo "   Installed $dirname"
        fi
    done
    # Restore .links (update path to point to the new project venv)
    if [ -d "$KISS_BUNDLE_DIR/playwright-browsers/.links" ]; then
        mkdir -p "$PW_DEST/.links"
        for lf in "$KISS_BUNDLE_DIR/playwright-browsers/.links/"*; do
            [ -f "$lf" ] && cp "$lf" "$PW_DEST/.links/$(basename "$lf")"
        done
    fi
    # Strip quarantine from Chromium binaries
    find "$PW_DEST" -type f -perm +111 -exec sh -c 'xattr -d com.apple.quarantine "$1" 2>/dev/null; xattr -d com.apple.provenance "$1" 2>/dev/null; codesign --force --sign - "$1" 2>/dev/null; true' _ {} \;
fi

# 6. Set up the project
echo ">>> Setting up project..."
export PATH="$INSTALL_BASE/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$INSTALL_BASE/python"
export PLAYWRIGHT_BROWSERS_PATH="$INSTALL_BASE/playwright-browsers"
if [ -d "$KISS_BUNDLE_DIR/project" ]; then
    mkdir -p "$PROJECT_DIR"
    rsync -a --exclude='nohup.out' "$KISS_BUNDLE_DIR/project/" "$PROJECT_DIR/"
    cd "$PROJECT_DIR"
    
    # Create venv with offline Python (--clear to handle re-installs)
    "$INSTALL_BASE/bin/uv" venv --python 3.13 --clear
    
    # Install from local wheels (fully offline, including pre-built project wheel)
    # Explicitly target the project venv to avoid uv resolving a different workspace
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 "$INSTALL_BASE/bin/uv" pip install --python "$PROJECT_DIR/.venv" --no-index --find-links "$KISS_BUNDLE_DIR/wheels" kiss-agent-framework

    # Symlink project entry-point scripts into $INSTALL_BASE/bin so they are on PATH
    for script in sorcar check generate-api-docs; do
        if [ -f "$PROJECT_DIR/.venv/bin/$script" ]; then
            ln -sf "$PROJECT_DIR/.venv/bin/$script" "$INSTALL_BASE/bin/$script"
        fi
    done
fi

# 7. Clean up unnecessary files to save disk space (~200MB)
echo ">>> Cleaning up unnecessary files..."
_SAVED=0

# code-server: remove .map source maps (~174MB), .d.ts type declarations (~16MB), metadata
if [ -d "$INSTALL_BASE/code-server" ]; then
    for pattern in '*.map' '*.d.ts'; do
        _bytes=$(find "$INSTALL_BASE/code-server" -name "$pattern" -type f -exec stat -f%z {} + 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
        find "$INSTALL_BASE/code-server" -name "$pattern" -type f -delete
        _SAVED=$((_SAVED + _bytes))
    done
    for f in ThirdPartyNotices.txt README.md npm-shrinkwrap.json postinstall.sh; do
        find "$INSTALL_BASE/code-server" -name "$f" -type f -delete
    done
fi

# Python standalone: remove headers, Tk/Tcl, IDLE, ensurepip, pydoc, turtle, man pages
if [ -d "$PYTHON_DEST" ]; then
    for d in include lib/tcl8.6 lib/tk8.6 share; do
        rm -rf "$PYTHON_DEST/$d"
    done
    PYLIB="$PYTHON_DEST/lib/python3.13"
    if [ -d "$PYLIB" ]; then
        for d in idlelib ensurepip pydoc_data turtledemo tkinter; do
            rm -rf "$PYLIB/$d"
        done
        rm -f "$PYLIB/turtle.py"
    fi
    rm -f "$PYTHON_DEST/BUILD"
fi

echo "   Cleaned up ~$((_SAVED / 1048576))MB of unnecessary files"

# 8. Create shell profile additions
PROFILE_SNIPPET="$INSTALL_BASE/env.sh"
cat > "$PROFILE_SNIPPET" << EOF
# KISS Agent Framework - added by offline installer
export PATH="$INSTALL_BASE/bin:\$PATH"
export UV_PYTHON_INSTALL_DIR="$INSTALL_BASE/python"
export PLAYWRIGHT_BROWSERS_PATH="$INSTALL_BASE/playwright-browsers"
EOF

# Add source line to user's shell rc file
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
echo ""
echo "Project installed at: $PROJECT_DIR"

# 9. Prompt for API keys and launch sorcar (only in interactive terminal)
if [ -t 0 ]; then
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo ""
        echo "ANTHROPIC_API_KEY is not set."
        printf "Enter your Anthropic API key: "
        read -r ANTHROPIC_API_KEY
        export ANTHROPIC_API_KEY
    fi

    if [ -z "${GEMINI_API_KEY:-}" ]; then
        echo ""
        echo "GEMINI_API_KEY is not set."
        printf "Enter your Gemini API key: "
        read -r GEMINI_API_KEY
        export GEMINI_API_KEY
    fi

    # Persist the keys in env.sh so future shells have them
    cat >> "$PROFILE_SNIPPET" << EOF
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export GEMINI_API_KEY="$GEMINI_API_KEY"
EOF

    # 10. Launch sorcar
    echo ""
    echo "Launching sorcar..."
    cd "$PROJECT_DIR"
    exec ./sorcar "$PROJECT_DIR"
else
    echo ""
    echo "Set your API keys in a new terminal:"
    echo "  export ANTHROPIC_API_KEY=your_key"
    echo "  export GEMINI_API_KEY=your_key"

    # Launch sorcar directly
    echo ""
    echo "To launch sorcar, open a new terminal and run:"
    echo "  cd $PROJECT_DIR && ./sorcar $PROJECT_DIR"
fi
INSTALL_SCRIPT

chmod +x "$BUNDLE/install-offline.sh"

# ---------------------------------------------------------------------------
# 8. Create the .pkg postinstall script
# ---------------------------------------------------------------------------
echo ">>> Creating package scripts..."
cat > "$SCRIPTS/postinstall" << 'POSTINSTALL'
#!/bin/bash
# macOS .pkg postinstall script (runs as current user — no root required)
set -uo pipefail

# The payload is installed to ~/.kiss-staging/kiss-offline by the pkg
BUNDLE="$HOME/.kiss-staging/kiss-offline"
LOG_FILE="$HOME/.kiss-staging/kiss-install.log"

echo "Installing KISS for user: $USER (home: $HOME)"

# Resolve install dir: marker file > default
_MARKER="$HOME/.kiss/install_dir"
if [ -f "$_MARKER" ]; then
    _SAVED_DIR="$(cat "$_MARKER" 2>/dev/null)"
fi
_INSTALL_DIR="${_SAVED_DIR:-$HOME/kiss_ai}"

export KISS_INSTALL_DIR="$_INSTALL_DIR"
export KISS_PROJECT_DIR="$_INSTALL_DIR"

# ---------------------------------------------------------------------------
# Show a native macOS alert with the installation log on failure.
# ---------------------------------------------------------------------------
_show_error_window() {
    local jxa_file="$HOME/.kiss-staging/kiss-error-dialog.js"
    cat > "$jxa_file" << JXAEOF
ObjC.import('Cocoa');

var app = $.NSApplication.sharedApplication;
app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);

var errorText = 'Could not read installation log.';
var fm = $.NSFileManager.defaultManager;
var logPath = '$LOG_FILE';
if (fm.fileExistsAtPath(logPath)) {
    var data = fm.contentsAtPath(logPath);
    if (data && data.length > 0) {
        var s = $.NSString.alloc.initWithDataEncoding(data, $.NSUTF8StringEncoding);
        if (s) errorText = ObjC.unwrap(s);
    }
}
if (errorText.length > 50000) {
    errorText = '... (earlier output truncated) ...\\n\\n' + errorText.slice(-50000);
}

var alert = $.NSAlert.alloc.init;
alert.messageText = \$('KISS Installation Failed');
alert.informativeText = \$('The installation encountered an error. See details below:');
alert.alertStyle = 2;

var scrollView = $.NSScrollView.alloc.initWithFrame($.NSMakeRect(0, 0, 560, 300));
scrollView.hasVerticalScroller = true;
scrollView.borderType = 2;

var cs = scrollView.contentSize;
var textView = $.NSTextView.alloc.initWithFrame(
    $.NSMakeRect(0, 0, cs.width, cs.height)
);
textView.editable = false;
textView.selectable = true;
textView.string = \$(errorText);
textView.font = $.NSFont.fontWithNameSize('Menlo', 11);
textView.verticallyResizable = true;
textView.textContainer.containerSize = $.NSMakeSize(cs.width, 1e7);
textView.textContainer.widthTracksTextView = true;

scrollView.documentView = textView;
alert.accessoryView = scrollView;
alert.addButtonWithTitle(\$('OK'));

app.activateIgnoringOtherApps(true);
alert.runModal;
JXAEOF

    osascript -l JavaScript "$jxa_file" 2>/dev/null || true
    rm -f "$jxa_file"
}

if ! bash "$BUNDLE/install-offline.sh" 2>&1 | tee "$LOG_FILE"; then
    echo "Installation failed. Showing error details..."
    _show_error_window
    rm -f "$LOG_FILE"
    exit 1
fi

# Clean up the staging payload — everything has been copied to user directories
rm -rf "$HOME/.kiss-staging"
echo "Cleaned up staging directory"

echo "KISS offline installation complete!"
POSTINSTALL
chmod +x "$SCRIPTS/postinstall"

# ---------------------------------------------------------------------------
# 9. Build the .pkg
# ---------------------------------------------------------------------------
echo ">>> Building .pkg..."
mkdir -p "$(dirname "$OUTPUT")"

# Build component package
# Create an empty component plist so pkgbuild does NOT auto-detect .app bundles
# in the payload (e.g. Playwright's "Google Chrome for Testing.app").  Without
# this, macOS Installer treats the embedded Chrome as a separate app to install,
# opening a second installation wizard.
cat > "$STAGE/component.plist" << 'CPLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array/>
</plist>
CPLIST

COMPONENT_PKG="$STAGE/kiss-component.pkg"
pkgbuild \
    --root "$PAYLOAD" \
    --identifier "$PKG_ID" \
    --version "$PKG_VERSION" \
    --install-location ".kiss-staging" \
    --scripts "$SCRIPTS" \
    --component-plist "$STAGE/component.plist" \
    "$COMPONENT_PKG"

# Create distribution XML for productbuild
cat > "$STAGE/distribution.xml" << DIST_XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>KISS Agent Framework (Offline)</title>
    <organization>berkeley.edu</organization>
    <domains enable_currentUserHome="true"/>
    <options customize="never" require-scripts="true" rootVolumeOnly="false"/>
    <welcome language="en" mime-type="text/plain"><![CDATA[
KISS Agent Framework - Offline Installer

This package installs all dependencies needed to run KISS Agent Framework
without an internet connection:

  • uv (Python package manager)
  • code-server (VS Code in the browser)
  • Python 3.13
  • Playwright Chromium (browser automation)
  • All Python dependencies
  • KISS project source

Your shell profile (~/.zshrc or ~/.bashrc) will be
automatically configured. Open a new terminal after
installation to use KISS.

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
    <pkg-ref id="com.kiss.offline-installer" version="${PKG_VERSION}" auth="none" onConclusion="none">kiss-component.pkg</pkg-ref>
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
echo "Or: installer -pkg $OUTPUT -target CurrentUserHomeDirectory"
