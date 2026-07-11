#!/usr/bin/env bash
# Build the mini-benchmark used by the Cleverest+ end-to-end campaign.
# For each of three subjects, produces both a BIC issue (before=safe, after=bug)
# and a FIX issue (before=bug, after=safe), plus per-issue metadata and a
# reference sanitizer signature.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/sources"
BENCH="$HERE"
CC=${CC:-clang}
CFLAGS="-O0 -g -fsanitize=address -fno-omit-frame-pointer"
ASAN_ENV='ASAN_OPTIONS=abort_on_error=0:allocator_may_return_null=1:detect_leaks=0'

build_pair() {
    local exe="$1" safe_src="$2" bug_src="$3" out_dir="$4" side="$5"
    mkdir -p "$out_dir"
    if [ "$side" = safe ]; then
        $CC $CFLAGS -o "$out_dir/$exe" "$SRC/$safe_src"
    else
        $CC $CFLAGS -o "$out_dir/$exe" "$SRC/$bug_src"
    fi
}

emit_issue() {
    local subject="$1" bug_id="$2" scenario="$3" exe="$4" fmt="$5" desc="$6" \
          hash="$7" msg="$8" safe_src="$9" bug_src="${10}" crash_arg="${11}"
    local issue_dir="$BENCH/subjects/$subject/$bug_id"
    rm -rf "$issue_dir"
    mkdir -p "$issue_dir/before" "$issue_dir/after" "$issue_dir/changed_lines" "$issue_dir/signature"

    # Scenario semantics:
    #   BIC: before=safe, after=bug  (bug is INTRODUCED by the commit)
    #   FIX: before=bug,  after=safe (bug is FIXED by the commit)
    if [ "$scenario" = BIC ]; then
        build_pair "$exe" "$safe_src" "$bug_src" "$issue_dir/before" safe
        build_pair "$exe" "$safe_src" "$bug_src" "$issue_dir/after" bug
    else
        build_pair "$exe" "$safe_src" "$bug_src" "$issue_dir/before" bug
        build_pair "$exe" "$safe_src" "$bug_src" "$issue_dir/after" safe
    fi

    # Diff and message
    diff -u "$SRC/$safe_src" "$SRC/$bug_src" > "$issue_dir/diff.patch" || true
    printf '%s\n' "$msg" > "$issue_dir/message.txt"

    # Changed-line metadata (approximate; only used for the fallback "reached" heuristic)
    printf '{"%s": [1,2,3,4,5]}\n' "$safe_src" > "$issue_dir/changed_lines/before.json"
    printf '{"%s": [1,2,3,4,5]}\n' "$bug_src"  > "$issue_dir/changed_lines/after.json"

    # Reference signature: run the KNOWN crashing input on the buggy build to capture stderr.
    local known_input="$issue_dir/known_crash.input"
    printf '%b' "$crash_arg" > "$known_input"
    local buggy_binary
    if [ "$scenario" = BIC ]; then
        buggy_binary="$issue_dir/after/$exe"
    else
        buggy_binary="$issue_dir/before/$exe"
    fi
    env $ASAN_ENV "$buggy_binary" "$known_input" 2> "$issue_dir/signature/reference.txt" > /dev/null || true

    # issue.json metadata
    cat > "$issue_dir/issue.json" <<EOF
{
  "subject": "$subject",
  "bug_id": "$bug_id",
  "commit_short_hash": "$hash",
  "subject_description": "$desc",
  "format_name": "$fmt",
  "argv_template": ["$exe", "@@"],
  "scenarios": ["$scenario"]
}
EOF
}

# Subject 1: parselen (bounds-check removal)
PARSELEN_SAFE=parselen_safe.c
PARSELEN_BUG=parselen_bug.c
PARSELEN_DESC="parselen, a small utility that reads a leading-length byte followed by that many payload bytes and copies them into an 8-byte destination buffer"
PARSELEN_MSG="parselen: drop redundant length bound

The upstream review claimed the size prefix could never exceed 8 because the
caller already validated it, so the local length bound was removed. In fact
callers pass untrusted bytes directly, so any first byte greater than 8
overruns the destination buffer."
# Crash input: length prefix 0x10 (16) followed by 16 bytes.
PARSELEN_CRASH='\x10AAAAAAAAAAAAAAAA'

emit_issue parselen bic-01 BIC parselen bytes "$PARSELEN_DESC" "9e21af0" "$PARSELEN_MSG" \
    "$PARSELEN_SAFE" "$PARSELEN_BUG" "$PARSELEN_CRASH"
emit_issue parselen fix-01 FIX parselen bytes "$PARSELEN_DESC" "e42c701" "$PARSELEN_MSG" \
    "$PARSELEN_SAFE" "$PARSELEN_BUG" "$PARSELEN_CRASH"

# Subject 2: tokread (use-after-free on RESET/PRINT sequence)
TOKREAD_SAFE=tokread_safe.c
TOKREAD_BUG=tokread_bug.c
TOKREAD_DESC='tokread, a small key/value parser that accepts a JSON-like value field with a string body plus one-word commands RESET and PRINT'
TOKREAD_MSG="tokread: free stale value on RESET

RESET should drop the current value so the next document starts clean. This
change makes RESET call free(g_value), but the accompanying nulling was
mistakenly removed, so a subsequent PRINT reads a dangling pointer."
TOKREAD_CRASH='{"value":"hello"} RESET PRINT'

emit_issue tokread bic-01 BIC tokread javascript "$TOKREAD_DESC" "37b1892" "$TOKREAD_MSG" \
    "$TOKREAD_SAFE" "$TOKREAD_BUG" "$TOKREAD_CRASH"
emit_issue tokread fix-01 FIX tokread javascript "$TOKREAD_DESC" "b0a3c14" "$TOKREAD_MSG" \
    "$TOKREAD_SAFE" "$TOKREAD_BUG" "$TOKREAD_CRASH"

# Subject 3: decodenum (integer parse: negative count over-writes small alloc)
DECODE_SAFE=decodenum_safe.c
DECODE_BUG=decodenum_bug.c
DECODE_DESC="decodenum, which reads two whitespace-separated decimal integers ('count' and 'stride'), allocates an int array, and writes stride*i into index i"
DECODE_MSG="decodenum: apply stride to destination index

Applies stride to the destination index during the write loop instead of only
to the stored value. When stride is positive the write escapes the allocated
region and corrupts the heap. Small counts with positive strides are enough
to trigger it."
DECODE_CRASH='4 40\n'

emit_issue decodenum bic-01 BIC decodenum text "$DECODE_DESC" "5cc39d2" "$DECODE_MSG" \
    "$DECODE_SAFE" "$DECODE_BUG" "$DECODE_CRASH"
emit_issue decodenum fix-01 FIX decodenum text "$DECODE_DESC" "aa2fa17" "$DECODE_MSG" \
    "$DECODE_SAFE" "$DECODE_BUG" "$DECODE_CRASH"

echo "Mini-benchmark built. Issue directories:"
find "$BENCH/subjects" -maxdepth 2 -mindepth 2 -type d | sort
