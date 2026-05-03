#!/bin/bash
# Copy KISS project files into kiss_project/ for standalone extension packaging.
# This makes the extension self-contained so it doesn't need the source tree.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DEST="$SCRIPT_DIR/kiss_project"

# Sync version into package.json (use KISS_EXP_VERSION override if set)
if [ -n "${KISS_EXP_VERSION:-}" ]; then
    VERSION="$KISS_EXP_VERSION"
else
    VERSION=$(python3 -c "exec(open('$PROJECT_ROOT/src/kiss/_version.py').read()); print(__version__)")
fi
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
# Copy pyproject.toml but strip [tool.hatch.build.targets.wheel.force-include]
# section — those paths reference root-level files (.dockerignore, .github, etc.)
# that aren't copied to kiss_project/, causing hatchling editable builds to fail.
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

# Copy LICENSE to the extension directory so vsce package can find it
cp "$PROJECT_ROOT/LICENSE" "$SCRIPT_DIR/LICENSE"

# Copy all git-tracked src/kiss/ files, excluding VS Code extension build
# artifacts (tsconfig, TS sources, node configs) that would confuse the
# TypeScript language server when nested inside kiss_project/.
cd "$PROJECT_ROOT"
git ls-files src/kiss/ | while IFS= read -r f; do
    [ -f "$f" ] || continue
    case "$f" in
        src/kiss/agents/vscode/*.py) ;;  # keep Python files
        # Runtime assets used by web_server.py (kiss-web daemon):
        #   /media/* is served as static HTTPS assets
        #   SAMPLE_TASKS.json populates welcome-screen suggestions
        # Both are loaded relative to web_server.py via
        # ``Path(__file__).parent``, so they must live alongside it
        # inside kiss_project/ as well.
        src/kiss/agents/vscode/media/*) ;;
        src/kiss/agents/vscode/SAMPLE_TASKS.json) ;;
        src/kiss/agents/vscode/*)  continue ;;  # skip everything else
    esac
    mkdir -p "$DEST/$(dirname "$f")"
    cp "$f" "$DEST/$f"
done

# Copy Claude Code skills if present (downloaded by install.sh before build)
CLAUDE_SKILLS_SRC="$PROJECT_ROOT/src/kiss/agents/claude_skills"
if [ -d "$CLAUDE_SKILLS_SRC" ]; then
    cp -R "$CLAUDE_SKILLS_SRC" "$DEST/src/kiss/agents/claude_skills"
    echo "Copied Claude Code skills to $DEST/src/kiss/agents/claude_skills"
fi

echo "Copied KISS project files to $DEST"
