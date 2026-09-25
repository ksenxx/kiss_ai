#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Sync the agent's persistent memory (~/.kiss/memories, or the ``memory_dir``
# set in ~/.kiss/config.json) between this machine and a remote host in *both*
# directions, so the agent on either machine remembers what the agent on the
# other learned.
#
# Usage:  scripts/sync-memory.sh user@ip-address
#
# Called by ./rsorcar on every deploy (step 4c); also useful on its own to
# bring two memories together without redeploying.
#
# The memory is a flat directory of Markdown pages (kiss.core.memoryfield), each
# with a frontmatter ``updated`` timestamp, plus a ``*.sqlite3`` vector index
# that is a cache keyed by page content.  Two passes, in this order:
#
#   1. remote -> here.  The remote's pages travel as a tar stream over the ssh
#      connection itself (a fresh cloud image has no rsync) into a scratch
#      directory here, and src/kiss/scripts/merge_memory_pages.py folds them
#      into this machine's memory: a page this machine lacks is added, a page
#      both have keeps whichever copy was updated last, nothing is deleted.
#   2. here -> remote.  The mirror image: this machine's pages -- by now the
#      union of both -- go to a scratch directory under the remote's ~/.kiss,
#      and the same script, copied there first, folds them into the remote's
#      memory with the remote's system python3 (the project's environment is
#      not built yet when ./rsorcar runs this).
#
# What this never does:
#
#   * delete a page anywhere -- a page that exists on one machine ends up on
#     both, and the only way a page changes is to become the newer of the two
#     copies;
#   * copy the index: it is rebuilt incrementally by the next agent that opens
#     the memory, from the pages themselves;
#   * write a page in place: every page lands by an atomic rename, so an agent
#     reading the memory meanwhile sees a whole page, old or new;
#   * pick between two copies of a page that were written in the very same
#     second with different content: that is a conflict, each machine keeps
#     its own copy, and the page is named so a person can decide.
#
# A pass that could not finish leaves both memories as they were, says so, and
# makes the script exit non-zero.  Needs python3 and tar here and on the remote.
set -euo pipefail

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
step() { printf '\033[0;34m[STEP]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()  { printf '\033[0;31m[ERR]\033[0m  %s\n' "$*" >&2; exit 1; }

# Quote a value for the remote shell: a path may hold a space or an
# apostrophe, which would otherwise break -- or extend -- the command line.
# Everything is wrapped in single quotes, and a single quote inside ends the
# quoting, escapes itself and starts it again.
shquote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

TARGET="${1:-}"
[[ -n "$TARGET" ]] || die "Usage: $0 user@ip-address"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE="$(dirname "$SCRIPT_DIR")/src/kiss/scripts/merge_memory_pages.py"
[[ -f "$MERGE" ]] || die "$MERGE is missing — nothing can be merged."

KISS_DIR="${KISS_HOME:-$HOME/.kiss}"

# The three directories of one machine, one per line: its kiss directory, the
# scratch directory pass 2 unpacks into (``memories.incoming`` beside the
# memory's default place), and its memory -- the ``memory_dir`` of its
# config.json when set, else ``memories`` in the kiss directory
# (kiss.agents.sorcar.sorcar_agent decides the same way).  Every path is
# canonical (symlinks and ``.`` components resolved), so that the check below
# -- the memory must not be the scratch directory or lie inside it, since the
# scratch directory is removed wholesale -- cannot be fooled by another
# spelling of the same directory.  A relative memory_dir is printed as it is.
# Run here with this machine's kiss directory, and on the remote with its own.
MEMORY_DIRS_PY='import json, os, sys
kiss_dir = os.path.realpath(sys.argv[1])
memory_dir = ""
try:
    with open(os.path.join(kiss_dir, "config.json")) as f:
        memory_dir = str(json.load(f).get("memory_dir", "") or "").strip()
except Exception:
    pass
memory_dir = os.path.expanduser(memory_dir) if memory_dir else os.path.join(kiss_dir, "memories")
print(kiss_dir)
print(os.path.join(kiss_dir, "memories.incoming"))
print(os.path.realpath(memory_dir) if os.path.isabs(memory_dir) else memory_dir)'

LOCAL_MEM="$(python3 -c "$MEMORY_DIRS_PY" "$KISS_DIR" | sed -n 3p)"
REMOTE_ANSWER="$(ssh "$TARGET" "python3 -c $(shquote "$MEMORY_DIRS_PY") \"\$HOME/.kiss\"")" \
    || die "Could not ask $TARGET where it keeps its memory."
REMOTE_KISS_DIR="${REMOTE_ANSWER%%$'\n'*}"
REMOTE_REST="${REMOTE_ANSWER#*$'\n'}"
REMOTE_STAGING="${REMOTE_REST%%$'\n'*}"
REMOTE_MEM="${REMOTE_REST#*$'\n'}"
[[ "$REMOTE_ANSWER" == *$'\n'*$'\n'* && "$REMOTE_MEM" != *$'\n'* \
   && "$REMOTE_KISS_DIR" == /* && "$REMOTE_STAGING" == /* ]] \
    || die "Could not tell where $TARGET keeps its memory (answer: '$REMOTE_ANSWER')."
# A relative memory_dir is relative to the working directory of whichever
# process opens the memory, which nothing here can know.
[[ "$REMOTE_MEM" == /* ]] \
    || die "$TARGET keeps its memory in '$REMOTE_MEM', which is not an absolute path; set memory_dir in its ~/.kiss/config.json to one."
case "$REMOTE_MEM" in
    "$REMOTE_STAGING"|"$REMOTE_STAGING"/*)
        die "$TARGET keeps its memory in $REMOTE_MEM, which this sync uses as its scratch directory; move it." ;;
esac

TMP_DIR="$(mktemp -d)"
# Set once the remote may hold the scratch directory of pass 2: a transfer cut
# short must not leave it behind.
REMOTE_STAGING_MADE=0
# Set by a pass that did not do what it set out to do.  Such a run has not
# synced the two machines, and must not report that it has.
INCOMPLETE=0
# Pages that were written in the same second on both machines with different
# content; pass 1 names and counts them, for the closing message.
CONFLICTS=0

incomplete() {
    warn "$@"
    INCOMPLETE=1
}

cleanup() {
    if (( REMOTE_STAGING_MADE )); then
        ssh "$TARGET" "rm -rf $(shquote "$REMOTE_STAGING")" >/dev/null 2>&1 \
            || warn "Could not remove $TARGET:$REMOTE_STAGING — delete it by hand."
    fi
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Print the merge script's report: its first line is the count summary, every
# further line the name of a page each machine keeps its own copy of.
# $1: the report (or only its first line), $2: which memory received the pages.
report_merge() {
    local summary="${1%%$'\n'*}" conflicts="" name
    [[ "$1" == *$'\n'* ]] && conflicts="${1#*$'\n'}"
    info "$2: $summary."
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        CONFLICTS=$((CONFLICTS + 1))
        printf '         both machines changed %s in the same second; each keeps its own copy.\n' "$name"
    done <<< "$conflicts"
}

# ---------------------------------------------------------------------------
# Pass 1: remote -> here
# ---------------------------------------------------------------------------
pull_from_remote() {
    local rc=0 report
    ssh "$TARGET" "test -d $(shquote "$REMOTE_MEM")" || rc=$?
    case $rc in
        0) ;;
        1) info "No memory on $TARGET yet ($REMOTE_MEM) — nothing to bring back."; return 0 ;;
        *) incomplete "Could not look for $TARGET's memory (ssh exited $rc) — nothing brought back."
           return 0 ;;
    esac
    step "Bringing $TARGET's memory pages here ..."
    mkdir -p "$TMP_DIR/incoming"
    if ! ssh "$TARGET" "tar -cf - --exclude='*.sqlite3*' -C $(shquote "$REMOTE_MEM") ." \
            | tar -xf - -C "$TMP_DIR/incoming"; then
        incomplete "Could not fetch the memory pages from $TARGET — nothing brought back."
        return 0
    fi
    if report="$(python3 "$MERGE" "$TMP_DIR/incoming" "$LOCAL_MEM")"; then
        report_merge "$report" "This machine's memory"
    else
        incomplete "Could not merge $TARGET's pages into $LOCAL_MEM —" \
                   "no page here was lost; the next sync finishes the job."
    fi
}

# ---------------------------------------------------------------------------
# Pass 2: here -> remote
# ---------------------------------------------------------------------------
push_to_remote() {
    local report
    if [[ ! -d "$LOCAL_MEM" ]]; then
        info "No memory on this machine ($LOCAL_MEM) — nothing to send to $TARGET."
        return 0
    fi
    step "Sending this machine's memory pages to $TARGET ..."
    if ! ssh "$TARGET" 'mkdir -p "$HOME/.kiss" && chmod 700 "$HOME/.kiss"' \
            || ! scp -q "$MERGE" "$TARGET:.kiss/"; then
        incomplete "Could not copy $(basename "$MERGE") to $TARGET — nothing sent."
        return 0
    fi
    REMOTE_STAGING_MADE=1
    # COPYFILE_DISABLE keeps macOS's tar from adding a ._page of extended
    # attributes for every page, --no-xattrs from putting the attributes into
    # headers GNU tar warns about on the remote.
    if ! report="$(COPYFILE_DISABLE=1 tar -cf - --no-xattrs --exclude='*.sqlite3*' -C "$LOCAL_MEM" . \
            | ssh "$TARGET" "rm -rf $(shquote "$REMOTE_STAGING") && mkdir -p $(shquote "$REMOTE_STAGING") \
                && tar -xf - -C $(shquote "$REMOTE_STAGING") \
                && python3 $(shquote "$REMOTE_KISS_DIR/merge_memory_pages.py") $(shquote "$REMOTE_STAGING") $(shquote "$REMOTE_MEM")")"; then
        incomplete "Could not merge this machine's pages into $TARGET:$REMOTE_MEM —" \
                   "no page there was lost; the next sync finishes the job."
        return 0
    fi
    # Only the count line: a page this pass finds in conflict was in conflict
    # in pass 1 as well (neither copy has changed since), and was named there.
    report_merge "${report%%$'\n'*}" "$TARGET's memory"
}

pull_from_remote
push_to_remote

if (( INCOMPLETE )); then
    die "The memories here and on $TARGET are not in sync yet (see the warnings above)."
fi
if (( CONFLICTS )); then
    warn "$CONFLICTS page(s) differ between the two machines (named above); everything else is in sync."
else
    info "The memories here and on $TARGET hold the same pages."
fi
