#!/bin/bash
# Copy KISS project files into kiss_project/ for standalone extension packaging.
# This makes the extension self-contained so it doesn't need the source tree.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DEST="$SCRIPT_DIR/kiss_project"

# Sync version from _version.py into package.json
VERSION=$(python3 -c "exec(open('$PROJECT_ROOT/src/kiss/_version.py').read()); print(__version__)")
if [ -n "$VERSION" ]; then
    # Use python for portable JSON editing
    python3 -c "
import json, pathlib
p = pathlib.Path('$SCRIPT_DIR/package.json')
d = json.loads(p.read_text())
d['version'] = '$VERSION'
p.write_text(json.dumps(d, indent=2) + '\n')
"
    echo "Synced extension version to $VERSION"
fi

rm -rf "$DEST"
mkdir -p "$DEST"

# Copy root project files needed for uv run
cp "$PROJECT_ROOT/pyproject.toml" "$DEST/"
cp "$PROJECT_ROOT/uv.lock" "$DEST/"
cp "$PROJECT_ROOT/README.md" "$DEST/"
cp "$PROJECT_ROOT/SYSTEM.md" "$DEST/"

# Copy all git-tracked src/kiss/ files
cd "$PROJECT_ROOT"
git ls-files src/kiss/ | while IFS= read -r f; do
    mkdir -p "$DEST/$(dirname "$f")"
    cp "$f" "$DEST/$f"
done

echo "Copied KISS project files to $DEST"
