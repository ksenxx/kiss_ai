#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Make the API keys a deploy shipped visible to every bash session of this
# machine, without damaging the shell configuration that is already here.
#
# Usage:  ssh user@host 'bash -s' < scripts/install-api-keys.sh
#         scripts/install-api-keys.sh            (on the machine itself)
#
# ~/.kiss/api_keys.env — the canonical key store every install shares (the
# kiss-web daemon parses it itself at startup) — is put in place by the deploy
# before this runs; this script only locks its permissions down and adds one
# line to ~/.bashrc that sources it, so interactive shells see the keys too.
# The line lives inside a delimited block, so a second deploy replaces the
# block instead of appending another copy.
#
# ~/.bashrc belongs to whoever uses the machine, so it is edited the careful
# way:
#
#   * the file as it was before the block was ever added is kept once, as
#     ~/.kiss/bashrc-before-sorcar-<time>.  A later deploy does not take
#     another copy -- that would overwrite the original with a file this
#     script has already edited;
#   * the new content is written beside it and renamed over it, so a deploy
#     that is interrupted mid-write leaves the previous ~/.bashrc rather than
#     a truncated one.  The temporary file inherits the real file's
#     permissions, and lives in the same directory so the rename is atomic;
#   * everything outside the block is copied through untouched.
set -euo pipefail

BEGIN='# >>> sorcar-cloud API keys >>>'
END='# <<< sorcar-cloud API keys <<<'
RC="$HOME/.bashrc"
# Where ./sorcar-cloud puts the two files, and what the line added to ~/.bashrc
# reads: this is not configurable, so that the three always agree.
KISS_DIR="$HOME/.kiss"

mkdir -p "$KISS_DIR"
chmod 700 "$KISS_DIR"
if [ -f "$KISS_DIR/api_keys.env" ]; then chmod 600 "$KISS_DIR/api_keys.env"; fi

touch "$RC"
# A ~/.bashrc that is a link to a dotfiles repository is edited where it really
# lives: replacing the link with a regular file would quietly disconnect the
# file somebody maintains from the shell that reads it.
if [ -L "$RC" ]; then
    TARGET="$(readlink -f "$RC" 2>/dev/null || true)"
    [ -n "$TARGET" ] && [ -f "$TARGET" ] && RC="$TARGET"
fi

# The file counts as one this script has already edited only when its markers
# come in pairs.  An unpaired BEGIN -- an interrupted earlier deploy -- means
# the file is *not* in a state this script wrote, and it is exactly the case
# where a copy of it matters most.  (It stays unpaired: the block below is
# removed only where it is complete, so the stray marker is carried through as
# an ordinary line and every deploy from now on takes a copy first.)
BEGINS="$(grep -cxF "$BEGIN" "$RC" || true)"
ENDS="$(grep -cxF "$END" "$RC" || true)"
if [ "$BEGINS" != "$ENDS" ] || [ "$BEGINS" = "0" ]; then
    backup="$KISS_DIR/bashrc-before-sorcar-$(date -u +%Y%m%dT%H%M%SZ)"
    while [ -e "$backup" ]; do backup="$backup+"; done
    cp -p "$RC" "$backup"
    echo "kept $RC as it was in $backup"
    [ "$BEGINS" = "$ENDS" ] \
        || echo "warning: $RC has $BEGINS begin and $ENDS end marker(s); the" \
                "unterminated one is left in place as an ordinary line."
fi

# cp -p first, so the temporary file starts out with the real file's owner
# and mode; the redirection then empties it without changing either.
NEW="$RC.sorcar-new"
rm -f "$NEW"
cp -p "$RC" "$NEW"
{
    # A block is dropped only where it is complete: from a BEGIN to the END
    # that closes it, with no second BEGIN in between.  So the lines after a
    # BEGIN are held back rather than discarded, and handed through unchanged
    # if what follows turns out not to be that block's END -- a file whose
    # block was never closed (an interrupted earlier deploy) keeps its whole
    # tail, this run and every run after it.
    awk -v b="$BEGIN" -v e="$END" '
        function flush() { for (i = 1; i <= n; i++) print held[i]; n = 0 }
        $0 == b        { flush(); held[n = 1] = $0; inblock = 1; next }
        inblock && $0 == e { n = 0; inblock = 0; next }
        inblock        { held[++n] = $0; next }
                       { print }
        END            { flush() }
    ' "$RC"
    echo "$BEGIN"
    echo '[ -f "$HOME/.kiss/api_keys.env" ] && . "$HOME/.kiss/api_keys.env"'
    echo "$END"
} > "$NEW"
mv -f "$NEW" "$RC"
echo "$RC sources $KISS_DIR/api_keys.env"
