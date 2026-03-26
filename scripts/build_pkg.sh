#!/bin/bash
#
# Build a macOS installer package (.pkg) for the KISS project.
# Bundles: VS Code, uv, the KISS VS Code extension, and the project source.
#
# At install time, places VS Code and uv into ~/.kiss/thirdparty/, creates
# symlinks in ~/.kiss/bin/, adds ~/.kiss/bin to the user's shell rc, creates
# a "Sorcar" app in /Applications, and installs the extension.
#
# Python, pip dependencies, Playwright Chromium, and API keys are handled
# automatically by the extension's ensureDependencies() on first activation.
#
# Usage: ./scripts/build_pkg.sh
# Output: dist/kiss-installer.pkg
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="$PROJECT_ROOT/.kiss.artifacts/tmp/pkg"
PAYLOAD="$STAGE/payload"
SCRIPTS="$STAGE/scripts"
PKG_ID="kiss.sorcar.installer"
PKG_VERSION=$(grep '__version__' "$PROJECT_ROOT/src/kiss/_version.py" | sed 's/.*"\(.*\)".*/\1/')
OUTPUT="$PROJECT_ROOT/dist/kiss-installer.pkg"
# Require Apple Silicon — x86 is not supported
if ! sysctl -n hw.optional.arm64 2>/dev/null | grep -q '1'; then
    echo "ERROR: Sorcar cannot be installed on x86 hardware. Apple Silicon (arm64) is required."
    exit 1
fi

echo "=== Building KISS Installer Package ==="
echo "Architecture: arm64"
echo "Staging: $STAGE"

# ---------------------------------------------------------------------------
# Helper: clear macOS quarantine/provenance attributes and re-sign binaries.
# ---------------------------------------------------------------------------
strip_quarantine() {
    local target="$1"
    xattr -d com.apple.quarantine "$target" 2>/dev/null || true
    codesign --force --sign - "$target" 2>/dev/null || true
}

# Clean staging
rm -rf "$STAGE"
mkdir -p "$PAYLOAD/kiss" "$SCRIPTS" "$STAGE/cache"

BUNDLE="$PAYLOAD/kiss"

# ---------------------------------------------------------------------------
# 1. Download latest uv binary
# ---------------------------------------------------------------------------
echo ">>> Downloading latest uv..."
mkdir -p "$BUNDLE/bin"
UV_TARBALL="uv-aarch64-apple-darwin.tar.gz"
UV_URL="https://github.com/astral-sh/uv/releases/latest/download/$UV_TARBALL"
UV_CACHE="$STAGE/cache/$UV_TARBALL"
if [ ! -f "$UV_CACHE" ]; then
    echo "   Downloading uv from $UV_URL..."
    curl -fSL -o "$UV_CACHE" "$UV_URL"
fi
echo "   Extracting uv..."
tar xzf "$UV_CACHE" -C "$BUNDLE/bin" --strip-components=1
chmod +x "$BUNDLE/bin/uv"
strip_quarantine "$BUNDLE/bin/uv"
if [ -f "$BUNDLE/bin/uvx" ]; then
    chmod +x "$BUNDLE/bin/uvx"
    strip_quarantine "$BUNDLE/bin/uvx"
fi
echo "   uv: $("$BUNDLE/bin/uv" --version 2>/dev/null || echo 'unknown')"

# ---------------------------------------------------------------------------
# 2. Download latest VS Code for macOS
# ---------------------------------------------------------------------------
echo ">>> Downloading latest VS Code..."
VSCODE_OS="darwin-arm64"
VSCODE_ZIP="$STAGE/cache/vscode-${VSCODE_OS}.zip"
VSCODE_URL="https://update.code.visualstudio.com/latest/${VSCODE_OS}/stable"
if [ ! -f "$VSCODE_ZIP" ]; then
    echo "   Downloading VS Code for $VSCODE_OS..."
    curl -fSL -o "$VSCODE_ZIP" "$VSCODE_URL"
fi
echo "   Extracting VS Code..."
mkdir -p "$BUNDLE/vscode-app"
# VS Code zip extracts to "Visual Studio Code.app"
ditto -xk "$VSCODE_ZIP" "$BUNDLE/vscode-app"
# Rename to a simpler name
if [ -d "$BUNDLE/vscode-app/Visual Studio Code.app" ]; then
    mv "$BUNDLE/vscode-app/Visual Studio Code.app" "$BUNDLE/vscode-app/Code.app"
fi
# Strip quarantine from all executables
find "$BUNDLE/vscode-app" -type f -perm +111 -exec sh -c \
    'xattr -d com.apple.quarantine "$1" 2>/dev/null; xattr -d com.apple.provenance "$1" 2>/dev/null; true' _ {} \;
echo "   VS Code: $(du -sh "$BUNDLE/vscode-app" | cut -f1)"

# ---------------------------------------------------------------------------
# 3. Build the VS Code extension (.vsix)
# ---------------------------------------------------------------------------
echo ">>> Building VS Code extension..."
VSCODE_EXT_DIR="$PROJECT_ROOT/src/kiss/agents/vscode"
cd "$VSCODE_EXT_DIR"
npm run package 2>&1 | tail -3
VSIX_FILE=$(ls -t "$VSCODE_EXT_DIR"/*.vsix 2>/dev/null | head -1)
if [ -z "$VSIX_FILE" ]; then
    echo "   ERROR: Failed to build .vsix extension"
    exit 1
fi
cp "$VSIX_FILE" "$BUNDLE/kiss-extension.vsix"
echo "   Extension: $(basename "$VSIX_FILE") ($(du -sh "$VSIX_FILE" | cut -f1))"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 4. Bundle the logo for the Sorcar app
# ---------------------------------------------------------------------------
echo ">>> Bundling logo..."
cp "$PROJECT_ROOT/assets/kiss_logo.svg" "$BUNDLE/kiss_logo.svg"
# Also copy PNG version if available (easier to convert to .icns)
if [ -f "$PROJECT_ROOT/assets/kiss_logo.png" ]; then
    cp "$PROJECT_ROOT/assets/kiss_logo.png" "$BUNDLE/kiss_logo.png"
fi

# ---------------------------------------------------------------------------
# 5. Project source
# ---------------------------------------------------------------------------
echo ">>> Bundling project source..."

mkdir -p "$BUNDLE/project"
rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='.kiss.artifacts' --exclude='htmlcov*' --exclude='.coverage*' \
    --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.pytest_cache' \
    --exclude='node_modules' --exclude='dist' --exclude='nohup.out' \
    "$PROJECT_ROOT/" "$BUNDLE/project/"
find "$BUNDLE/project" -name 'nohup.out' -delete 2>/dev/null || true
echo "   project: $(du -sh "$BUNDLE/project" | cut -f1)"

# ---------------------------------------------------------------------------
# 6. Create the install script (runs as postinstall in .pkg)
# ---------------------------------------------------------------------------
echo ">>> Creating install script..."
cat > "$BUNDLE/install.sh" << 'INSTALL_SCRIPT'
#!/bin/bash
#
# KISS Installer
# Installs VS Code and uv to ~/.kiss/thirdparty/, creates symlinks in
# ~/.kiss/bin/, adds to shell rc, creates Sorcar.app, installs extension.
# Python, pip deps, Playwright, and API keys are handled by the extension.
#
set -euo pipefail

_strip_quarantine() {
    xattr -d com.apple.quarantine "$1" 2>/dev/null || true
    xattr -d com.apple.provenance "$1" 2>/dev/null || true
    codesign --force --sign - "$1" 2>/dev/null || true
}

KISS_BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="$HOME"
THIRDPARTY="$HOME_DIR/.kiss/thirdparty"
BIN_DIR="$HOME_DIR/.kiss/bin"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! sysctl -n hw.optional.arm64 2>/dev/null | grep -q '1'; then
    echo "ERROR: Sorcar cannot be installed on x86 hardware. Apple Silicon (arm64) is required."
    exit 1
fi

_REQUIRED_MB=3000
_AVAIL_MB=$(df -m "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "$_AVAIL_MB" ] && [ "$_AVAIL_MB" -lt "$_REQUIRED_MB" ] 2>/dev/null; then
    echo "ERROR: Insufficient disk space. Need ${_REQUIRED_MB}MB, only ${_AVAIL_MB}MB available."
    exit 1
fi

# Determine project install location
_DEFAULT_DIR="$HOME/.kiss/project"
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

echo "=== KISS Installer ==="
echo "Bundle: $KISS_BUNDLE_DIR"
echo "Thirdparty: $THIRDPARTY"
echo "Bin: $BIN_DIR"
echo "Project: $PROJECT_DIR"

mkdir -p "$THIRDPARTY" "$BIN_DIR" "$HOME_DIR/.kiss"
printf '%s\n' "$INSTALL_BASE" > "$HOME_DIR/.kiss/install_dir"

# ---------------------------------------------------------------------------
# 1. Install uv to ~/.kiss/thirdparty/uv/ and symlink to ~/.kiss/bin/
# ---------------------------------------------------------------------------
echo ">>> Installing uv..."
mkdir -p "$THIRDPARTY/uv"
cp "$KISS_BUNDLE_DIR/bin/uv" "$THIRDPARTY/uv/uv"
chmod +x "$THIRDPARTY/uv/uv"
_strip_quarantine "$THIRDPARTY/uv/uv"
ln -sf "$THIRDPARTY/uv/uv" "$BIN_DIR/uv"
if [ -f "$KISS_BUNDLE_DIR/bin/uvx" ]; then
    cp "$KISS_BUNDLE_DIR/bin/uvx" "$THIRDPARTY/uv/uvx"
    chmod +x "$THIRDPARTY/uv/uvx"
    _strip_quarantine "$THIRDPARTY/uv/uvx"
    ln -sf "$THIRDPARTY/uv/uvx" "$BIN_DIR/uvx"
fi
echo "   uv installed at $THIRDPARTY/uv/, symlinked to $BIN_DIR/uv"

# ---------------------------------------------------------------------------
# 2. Install VS Code to ~/.kiss/thirdparty/vscode/ and symlink to ~/.kiss/bin/
# ---------------------------------------------------------------------------
echo ">>> Installing VS Code..."
rm -rf "$THIRDPARTY/vscode"
mkdir -p "$THIRDPARTY/vscode"
cp -R "$KISS_BUNDLE_DIR/vscode-app/Code.app" "$THIRDPARTY/vscode/Code.app"
# Strip quarantine from all VS Code executables
find "$THIRDPARTY/vscode" -type f -perm +111 -exec sh -c \
    'xattr -d com.apple.quarantine "$1" 2>/dev/null; xattr -d com.apple.provenance "$1" 2>/dev/null; true' _ {} \;
# Symlink the 'code' CLI to ~/.kiss/bin/
VSCODE_CLI="$THIRDPARTY/vscode/Code.app/Contents/Resources/app/bin/code"
if [ -x "$VSCODE_CLI" ]; then
    ln -sf "$VSCODE_CLI" "$BIN_DIR/code"
    echo "   VS Code installed at $THIRDPARTY/vscode/, CLI symlinked to $BIN_DIR/code"
else
    echo "   WARNING: VS Code CLI not found at expected location"
fi

# ---------------------------------------------------------------------------
# 3. Add ~/.kiss/bin to shell rc file
# ---------------------------------------------------------------------------
echo ">>> Configuring shell PATH..."
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
    */zsh)  _add_path_to_rc "$HOME_DIR/.zshrc" ;;
    */bash) _add_path_to_rc "$HOME_DIR/.bashrc" ;;
    */fish)
        # fish uses a different syntax
        _fish_config="$HOME_DIR/.config/fish/config.fish"
        mkdir -p "$(dirname "$_fish_config")"
        _fish_line="fish_add_path $BIN_DIR"
        if [ -f "$_fish_config" ] && grep -qF "$BIN_DIR" "$_fish_config"; then
            echo "   $BIN_DIR already in $_fish_config"
        else
            printf '\n# KISS Agent Framework\n%s\n' "$_fish_line" >> "$_fish_config"
            echo "   Added $BIN_DIR to $_fish_config"
        fi
        ;;
    *)      _add_path_to_rc "$HOME_DIR/.zshrc"
            _add_path_to_rc "$HOME_DIR/.bashrc" ;;
esac

# ---------------------------------------------------------------------------
# 4. Create Sorcar.app in /Applications
# ---------------------------------------------------------------------------
echo ">>> Creating Sorcar application..."
SORCAR_APP="/Applications/Sorcar.app"
rm -rf "$SORCAR_APP"
mkdir -p "$SORCAR_APP/Contents/MacOS"
mkdir -p "$SORCAR_APP/Contents/Resources"

# Convert SVG logo to .icns icon
_create_icns() {
    local svg_src="$1"
    local png_src="$KISS_BUNDLE_DIR/kiss_logo.png"
    local iconset_dir="$KISS_BUNDLE_DIR/Sorcar.iconset"
    local icns_out="$2"

    mkdir -p "$iconset_dir"

    # Use PNG if available, otherwise try to convert SVG with sips
    local src_png=""
    if [ -f "$png_src" ]; then
        src_png="$png_src"
    else
        # Try qlmanage to render SVG to PNG
        src_png="$KISS_BUNDLE_DIR/kiss_logo_rendered.png"
        qlmanage -t -s 1024 -o "$KISS_BUNDLE_DIR" "$svg_src" 2>/dev/null || true
        if [ -f "$KISS_BUNDLE_DIR/kiss_logo.svg.png" ]; then
            mv "$KISS_BUNDLE_DIR/kiss_logo.svg.png" "$src_png"
        fi
    fi

    if [ -f "$src_png" ]; then
        # Generate all required icon sizes
        for size in 16 32 64 128 256 512 1024; do
            sips -z $size $size "$src_png" --out "$iconset_dir/icon_${size}x${size}.png" 2>/dev/null || true
        done
        # Create @2x variants
        for size in 16 32 128 256 512; do
            double=$((size * 2))
            if [ -f "$iconset_dir/icon_${double}x${double}.png" ]; then
                cp "$iconset_dir/icon_${double}x${double}.png" "$iconset_dir/icon_${size}x${size}@2x.png"
            fi
        done
        iconutil -c icns -o "$icns_out" "$iconset_dir" 2>/dev/null || true
    fi
    rm -rf "$iconset_dir"
}

ICNS_FILE="$SORCAR_APP/Contents/Resources/Sorcar.icns"
_create_icns "$KISS_BUNDLE_DIR/kiss_logo.svg" "$ICNS_FILE"

# If icns conversion failed, copy the SVG as a fallback reference
if [ ! -f "$ICNS_FILE" ]; then
    cp "$KISS_BUNDLE_DIR/kiss_logo.svg" "$SORCAR_APP/Contents/Resources/kiss_logo.svg"
    echo "   WARNING: Could not create .icns icon, using SVG fallback"
fi

# Create the launcher script
cat > "$SORCAR_APP/Contents/MacOS/Sorcar" << LAUNCHER
#!/bin/bash
# Sorcar launcher — opens the bundled VS Code
VSCODE_APP="$THIRDPARTY/vscode/Code.app"
if [ -d "\$VSCODE_APP" ]; then
    open "\$VSCODE_APP" "\$@"
else
    osascript -e 'display alert "Sorcar" message "VS Code not found at $THIRDPARTY/vscode/Code.app. Please reinstall." as critical'
    exit 1
fi
LAUNCHER
chmod +x "$SORCAR_APP/Contents/MacOS/Sorcar"

# Create Info.plist
cat > "$SORCAR_APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Sorcar</string>
    <key>CFBundleDisplayName</key>
    <string>Sorcar</string>
    <key>CFBundleIdentifier</key>
    <string>com.kiss.sorcar</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>Sorcar</string>
    <key>CFBundleIconFile</key>
    <string>Sorcar</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

echo "   Sorcar.app created at /Applications/Sorcar.app"

# ---------------------------------------------------------------------------
# 5. Copy project source and install the VS Code extension
# ---------------------------------------------------------------------------
echo ">>> Installing project source..."
export PATH="$BIN_DIR:$PATH"
if [ -d "$KISS_BUNDLE_DIR/project" ]; then
    mkdir -p "$PROJECT_DIR"
    rsync -a --exclude='nohup.out' "$KISS_BUNDLE_DIR/project/" "$PROJECT_DIR/"
    echo "   Project installed at $PROJECT_DIR"
fi

echo ">>> Installing KISS VS Code extension..."
VSCODE_CLI="$THIRDPARTY/vscode/Code.app/Contents/Resources/app/bin/code"
if [ -x "$VSCODE_CLI" ] && [ -f "$KISS_BUNDLE_DIR/kiss-extension.vsix" ]; then
    "$VSCODE_CLI" --install-extension "$KISS_BUNDLE_DIR/kiss-extension.vsix" --force 2>&1 || true
    echo "   Extension installed into bundled VS Code"
else
    echo "   WARNING: Could not install extension (CLI or .vsix not found)"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "VS Code at: $THIRDPARTY/vscode/Code.app"
echo "uv at: $THIRDPARTY/uv/uv"
echo "Binaries symlinked in: $BIN_DIR"
echo "Sorcar app: /Applications/Sorcar.app"
echo "Project: $PROJECT_DIR"
echo ""
echo "Open Sorcar from /Applications or run: $BIN_DIR/code"
echo "The extension will auto-install Python, dependencies, and Chromium on first launch."
INSTALL_SCRIPT

chmod +x "$BUNDLE/install.sh"

# ---------------------------------------------------------------------------
# 7. Create the .pkg postinstall script
# ---------------------------------------------------------------------------
echo ">>> Creating package scripts..."
cat > "$SCRIPTS/postinstall" << 'POSTINSTALL'
#!/bin/bash
# macOS .pkg postinstall script
set -uo pipefail

BUNDLE="$HOME/.kiss-staging/kiss"
LOG_FILE="$HOME/.kiss-staging/kiss-install.log"

echo "Installing KISS for user: $USER (home: $HOME)"

_MARKER="$HOME/.kiss/install_dir"
if [ -f "$_MARKER" ]; then
    _SAVED_DIR="$(cat "$_MARKER" 2>/dev/null)"
fi
_INSTALL_DIR="${_SAVED_DIR:-$HOME/.kiss/project}"

export KISS_INSTALL_DIR="$_INSTALL_DIR"
export KISS_PROJECT_DIR="$_INSTALL_DIR"

# Show a native macOS alert with the installation log on failure.
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

if ! bash "$BUNDLE/install.sh" 2>&1 | tee "$LOG_FILE"; then
    echo "Installation failed. Showing error details..."
    _show_error_window
    rm -f "$LOG_FILE"
    exit 1
fi

rm -rf "$HOME/.kiss-staging"
echo "Cleaned up staging directory"
echo "KISS installation complete!"
POSTINSTALL
chmod +x "$SCRIPTS/postinstall"

# ---------------------------------------------------------------------------
# 8. Build the .pkg
# ---------------------------------------------------------------------------
echo ">>> Building .pkg..."
mkdir -p "$(dirname "$OUTPUT")"

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

cat > "$STAGE/distribution.xml" << DIST_XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>KISS Agent Framework</title>
    <organization>berkeley.edu</organization>
    <domains enable_currentUserHome="true"/>
    <options customize="never" require-scripts="true" rootVolumeOnly="false"/>
    <welcome language="en" mime-type="text/plain"><![CDATA[
KISS Agent Framework Installer

This package installs the KISS Agent Framework:

  • VS Code (full desktop editor)
  • uv (Python package manager)
  • KISS VS Code Extension
  • KISS project source

VS Code and uv are installed to ~/.kiss/thirdparty/ with CLI
symlinks in ~/.kiss/bin/. A "Sorcar" application is created in
/Applications for easy access.

On first launch, the extension will automatically install Python,
all pip dependencies, Playwright Chromium, and prompt for API keys.
]]></welcome>
    <choices-outline>
        <line choice="default">
            <line choice="com.kiss.installer"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="com.kiss.installer" visible="false">
        <pkg-ref id="com.kiss.installer"/>
    </choice>
    <pkg-ref id="com.kiss.installer" version="${PKG_VERSION}" auth="none" onConclusion="none">kiss-component.pkg</pkg-ref>
</installer-gui-script>
DIST_XML

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
