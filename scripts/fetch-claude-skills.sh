#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Download official Claude Code skills into src/kiss/agents/claude_skills.
#
# Usage:
#   scripts/fetch-claude-skills.sh
#
# Behaviour:
#   - Resolves PROJECT_ROOT as the parent of this script's directory.
#   - Creates $PROJECT_ROOT/src/kiss/agents/claude_skills if missing.
#   - If the directory already contains at least one plugin subdirectory,
#     exits 0 without re-downloading (idempotent).
#   - Otherwise sparse-clones github.com/anthropics/claude-code and copies
#     each plugins/* directory into claude_skills/.
#   - Requires git in PATH. Prints a warning and exits 0 if git is missing
#     or the clone fails, so callers (install.sh, build-extension.sh) can
#     continue with whatever state is on disk.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLAUDE_SKILLS_DIR="$PROJECT_ROOT/src/kiss/agents/claude_skills"

if [ -d "$CLAUDE_SKILLS_DIR" ] && [ -n "$(ls -d "$CLAUDE_SKILLS_DIR"/*/ 2>/dev/null)" ]; then
    echo "   Claude skills already present at $CLAUDE_SKILLS_DIR — skipping download"
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    echo "   WARNING: git not found in PATH — cannot download Claude Code skills"
    exit 0
fi

mkdir -p "$CLAUDE_SKILLS_DIR"
SKILLS_TMP="$(mktemp -d)"
trap 'rm -rf "$SKILLS_TMP"' EXIT

echo "   Cloning anthropics/claude-code plugins into $CLAUDE_SKILLS_DIR ..."
# Wrap ``git clone`` with a timeout so a slow or blocked network cannot
# hang the whole ``build-extension.sh`` / ``install.sh`` invocation
# indefinitely — the historical failure mode with no timeout on this
# clone.  Prefer ``timeout`` (GNU coreutils) then ``gtimeout``
# (Homebrew's ``brew install coreutils``); fall back to no wrapper on
# systems that have neither so behaviour is unchanged for users who
# never had the hang.  On timeout the outer ``if`` reaches the else
# branch, we warn, and callers continue with whatever is on disk.
if command -v timeout >/dev/null 2>&1; then
    GIT_CLONE_TIMEOUT="timeout 60"
elif command -v gtimeout >/dev/null 2>&1; then
    GIT_CLONE_TIMEOUT="gtimeout 60"
else
    GIT_CLONE_TIMEOUT=""
fi
if $GIT_CLONE_TIMEOUT git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/anthropics/claude-code.git "$SKILLS_TMP/claude-code" 2>&1; then
    (
        cd "$SKILLS_TMP/claude-code"
        git sparse-checkout set plugins 2>&1
        for plugin_dir in plugins/*/; do
            if [ -d "$plugin_dir" ]; then
                plugin_name="$(basename "$plugin_dir")"
                cp -R "$plugin_dir" "$CLAUDE_SKILLS_DIR/$plugin_name"
            fi
        done
    )
    SKILL_COUNT="$(ls -d "$CLAUDE_SKILLS_DIR"/*/ 2>/dev/null | wc -l | tr -d ' ')"
    echo "   Installed $SKILL_COUNT Claude skills to $CLAUDE_SKILLS_DIR"
else
    echo "   WARNING: Failed to download Claude Code skills"
fi
