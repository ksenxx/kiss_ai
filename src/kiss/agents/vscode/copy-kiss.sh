#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DEST="$SCRIPT_DIR/kiss_project"

if [ -n "${KISS_EXP_VERSION:-}" ]; then
    VERSION="$KISS_EXP_VERSION"
else
    VERSION=$(python3 -c "exec(open('$PROJECT_ROOT/src/kiss/core/_version.py').read()); print(__version__)")
fi
if [ -n "$VERSION" ]; then
    python3 -c "
import json, pathlib
p = pathlib.Path('$SCRIPT_DIR/package.json')
d = json.loads(p.read_text())
d['version'] = '$VERSION'
p.write_text(json.dumps(d, indent=2) + '\n')
"
    echo "Synced extension version to $VERSION"
fi

echo "Preparing kiss_project directory..."
rm -rf "$DEST"
mkdir -p "$DEST"

python3 -c "
import re, pathlib
text = pathlib.Path('$PROJECT_ROOT/pyproject.toml').read_text()
text = re.sub(
    r'\n# Include all git-managed files outside src/kiss/ in the wheel\n'
    r'\[tool\.hatch\.build\.targets\.wheel\.force-include\]\n'
    r'(?:.*\n)*?(?=\n\[)',
    '',
    text,
)
pathlib.Path('$DEST/pyproject.toml').write_text(text)
"
cp "$PROJECT_ROOT/uv.lock" "$DEST/"
cp "$PROJECT_ROOT/README.md" "$DEST/"

cp "$PROJECT_ROOT/LICENSE" "$SCRIPT_DIR/LICENSE"

echo "Copying source files..."
cd "$PROJECT_ROOT"
git ls-files src/kiss/ | while IFS= read -r f; do
    [ -f "$f" ] || continue
    case "$f" in
        src/kiss/agents/vscode/*.py) ;;
        src/kiss/agents/vscode/media/*) ;;
        src/kiss/agents/vscode/*)  continue ;;
    esac
    mkdir -p "$DEST/$(dirname "$f")"
    cp "$f" "$DEST/$f"
done

# Extra directories (space-separated, relative to PROJECT_ROOT) listed in
# KISS_BUNDLE_EXTRA_DIRS are bundled into the extension.  release.sh uses
# this to ship additional content in release VSIXes; install.sh leaves the
# variable unset, so a source install bundles nothing extra.  Note that
# `npm run package` re-runs this script via `vscode:prepublish`, so the
# variable must be set for the package step too, not just `copy-kiss`.
# BEGIN: kiss-extra-bundle  (tests extract this block verbatim)
for extra_dir in ${KISS_BUNDLE_EXTRA_DIRS:-}; do
    if [ -d "$PROJECT_ROOT/$extra_dir" ]; then
        mkdir -p "$DEST/$(dirname "$extra_dir")"
        cp -R "$PROJECT_ROOT/$extra_dir" "$DEST/$extra_dir"
        echo "Bundled $extra_dir into $DEST/$extra_dir"
    fi
done
# END: kiss-extra-bundle

echo "Copied KISS project files to $DEST"
