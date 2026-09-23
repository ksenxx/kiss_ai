#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# How many API keys does this machine hold?  Prints one number.
#
# Usage:  { printf 'KEY_RE=%q\nFISH_RE=%q\n' "$KEY_LINE_RE" "$FISH_KEY_LINE_RE"
#           cat scripts/count-api-keys.sh; } | ssh user@host 'bash -s'
#         KEY_RE=<regex> FISH_RE=<regex> scripts/count-api-keys.sh   (on the machine itself)
#
#   KEY_RE   what a key line looks like in a bash/zsh file (export FOO_API_KEY=value)
#   FISH_RE  what a key line looks like in fish's config.fish (set -gx FOO_API_KEY value)
#
# ./rsorcar runs this on the remote before deciding whether to copy this
# machine's keys there (step 4): a remote that already has keys — in
# ~/.kiss/api_keys.env, the key store every install shares, or in a shell rc
# file on an install older than that store — keeps them.  The two patterns
# are rsorcar's, so the two machines agree on what counts as a key; they hold
# both kinds of quote, so rsorcar sends them ahead of this script on standard
# input, quoted by printf %q, rather than on the ssh command line where the
# remote's login shell would get to interpret them.
#
# The exit status matters more than the number: a file that exists but cannot
# be read exits 2, and rsorcar then stops the deploy rather than take the
# unreadable store for an empty one — the one case a copy must never happen
# in is the one where the remote's keys cannot be seen.
#
# This is a separate file rather than a here-document inside rsorcar because
# macOS ships bash 3.2, whose command substitution expands the parameters of
# a quoted here-document (``$( ... <<'EOF' ... EOF )``) on the local machine
# — the very thing the quotes are meant to prevent.

: "${KEY_RE:?the bash/zsh key-line pattern is required}" \
  "${FISH_RE:?the fish key-line pattern is required}"

count=0
count_keys() {  # $1: file, $2: what a key line looks like in it
    [ -e "$1" ] || return 0
    n="$(grep -cE "$2" "$1" 2>/dev/null)"
    case $? in
        0|1) count=$((count + n)) ;;
        *) echo "cannot read $1" >&2; exit 2 ;;
    esac
}
count_keys "$HOME/.kiss/api_keys.env" "$KEY_RE"
count_keys "$HOME/.bashrc" "$KEY_RE"
count_keys "$HOME/.zshrc" "$KEY_RE"
count_keys "$HOME/.config/fish/config.fish" "$FISH_RE"
echo "$count"
