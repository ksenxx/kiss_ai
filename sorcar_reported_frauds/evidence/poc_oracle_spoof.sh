#!/usr/bin/env bash
# Harmless POC: prove program-controlled output is accepted as an ASan crash.
set -u
ARTIFACT="${1:-./tmp/cleverest}"
cd "$ARTIFACT" || exit 2
# On macOS, artifact's grep -P requires Homebrew GNU grep; Linux needs no override.
if command -v ggrep >/dev/null 2>&1; then
  d=$(mktemp -d); ln -s "$(command -v ggrep)" "$d/grep"; PATH="$d:$PATH"
fi
# shellcheck source=/dev/null
source ./utils.sh
before='normal output'
after='ERROR: AddressSanitizer: heap-buffer-overflow (printed by target, not ASan)'
bug_before="$(check_output_bug "$before")"
bug_after="$(check_output_bug "$after")"
final=N
if [[ "$bug_before" || "$bug_after" ]]; then
  if [[ "$bug_before" && "$bug_after" ]]; then final=X
  elif [[ "$bug_after" ]]; then final=B
  fi
fi
printf 'bug_before=%q\nbug_after=%q\nclassified_BIC_final=%s\n' "$bug_before" "$bug_after" "$final"
test "$final" = B
