#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Give this machine the GitHub.com credentials of the machine that deployed
# it, so the agent running here can use ``gh`` and push over https as the
# same person.
#
# Usage:  scripts/collect-github-auth.sh | ssh user@host 'bash ~/.kiss/install-github-auth.sh'
#         bash install-github-auth.sh < payload          (on the machine itself)
#
# The payload arrives on standard input, one line per account:
#
#     account <login> <token>
#
# It comes in on standard input and not as arguments because the arguments of
# a process are world-readable (/proc/<pid>/cmdline has no ptrace gate, which
# is why ``ps`` shows everybody's), and the remote command of an ssh call
# becomes the arguments of a shell here.  Nothing in this script echoes a
# token: ``gh auth status`` masks them, and the fallback writes them to a file
# only this account can read.
#
# What it does with them:
#
#   * ``gh auth login --with-token`` for each account, the last one ending up
#     active, exactly as it is on the machine the credentials came from.  gh
#     validates the token against the API as it goes, so a token that no
#     longer works is reported rather than stored.  ``--insecure-storage``
#     writes the token into ~/.config/gh/hosts.yml: a server has no keyring
#     to put it in, and asking for one can hang a session that has no
#     terminal to answer with.
#   * ``gh auth setup-git``, which points git at gh for github.com URLs.  It
#     has to run after the login -- gh refuses to configure git for a host it
#     is not logged in to -- and it is idempotent, so repeated deploys leave
#     one helper and not a pile of them.  It gets there by emptying the list
#     of credential helpers for github.com first, which is why ~/.gitconfig
#     is copied before it runs: a helper this machine was configured with
#     stops being asked about GitHub, and the copy is how it comes back.
#   * gh is installed first if it is missing, from the release tarball into
#     ~/.local/bin, because installing it the recommended way needs root.
#
# When gh cannot be had at all, the credentials are still installed the way
# git alone understands them: a line per account in ~/.git-credentials and
# ``store`` *added* to the helpers for github.com.  Added, not set: a helper
# somebody configured here is theirs, and git simply asks the helpers in turn.
#
# Nobody else on the machine gets to read a token.  Everything written here
# is written under umask 077, and a ~/.config/gh/hosts.yml that other accounts
# can already read is made unreadable to them *before* the first token goes
# into it -- a secret written where somebody else can read it cannot be taken
# back.  A machine where that cannot be done is given the credentials the
# git-only way instead, in a file this script creates itself.
#
# Nothing is overwritten without a copy.  Each of the three files a deploy
# can change -- ~/.config/gh/hosts.yml (the accounts this machine was already
# logged in to), ~/.gitconfig and ~/.git-credentials -- is copied into
# ~/.kiss/ as it was before the first deploy touched it, and only then: a
# second copy would be of a file this script has already written, and it
# would land on top of the one that is worth having.  On top of that,
# ~/.git-credentials keeps every line belonging to an account or a host this
# payload does not mention.
set -euo pipefail
# Files written here hold credentials; nobody else on the machine gets to
# read them, whatever the login shell's umask happens to be.
umask 077

HOST="github.com"
KISS_DIR="$HOME/.kiss"
GH_CONFIG_HOME="${GH_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/gh}"
HOSTS_YML="$GH_CONFIG_HOME/hosts.yml"
CREDENTIALS="$HOME/.git-credentials"

say() { printf '[%s] %s\n' "$(hostname -s 2>/dev/null || echo remote)" "$*"; }

# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------
LOGINS=()
TOKENS=()
while read -r kind login token _rest; do
    [ "${kind:-}" = "account" ] || continue
    [ -n "${login:-}" ] && [ -n "${token:-}" ] || continue
    LOGINS+=("$login")
    TOKENS+=("$token")
done
if [ "${#LOGINS[@]}" -eq 0 ]; then
    echo "ERROR: no github.com credentials arrived on standard input." >&2
    exit 1
fi

mkdir -p "$KISS_DIR"
chmod 700 "$KISS_DIR"

# ---------------------------------------------------------------------------
# Keep what was here before a deploy changes it
#
# The first deploy keeps the file the machine had of its own; a later deploy
# keeps it again only if what is there now is not already kept, which is
# exactly the case that matters: somebody added a credential helper of their
# own after the last deploy, and ``gh auth setup-git`` is about to clear it.
# Copying a file whose content is already kept would add nothing but noise --
# so re-deploying an untouched machine costs no backup at all -- and the
# earliest copy, the one somebody reaches for when they want back what the
# machine had before any of this, stays where it is, under the earliest name.
#
# A copy that could not be made is a failure, and it is returned as one: the
# caller is about to change the file, and "nothing is overwritten without a
# copy" is only true if the change does not happen when the copy does not.
# ``set -e`` cannot be relied on here, because a function called on the left
# of ``||`` runs with it switched off.
# ---------------------------------------------------------------------------
keep_before_change() {
    local file="$1" name="$2" backup
    [ -f "$file" ] || return 0
    for backup in "$KISS_DIR/$name"-*; do
        [ -e "$backup" ] && cmp -s "$file" "$backup" && return 0
    done
    backup="$KISS_DIR/$name-$(date -u +%Y%m%dT%H%M%SZ)"
    # The name carries the time to the second, and two deploys can happen
    # inside one second: writing over the copy that is already there would
    # destroy the record this whole function exists to keep.
    while [ -e "$backup" ]; do backup="$backup+"; done
    if ! { cp -p "$file" "$backup" && chmod go-rwx "$backup"; }; then
        rm -f "$backup"
        say "ERROR: could not keep a copy of $file in $KISS_DIR."
        return 1
    fi
    say "kept $(basename "$file") as this machine had it in ${backup/#$HOME/~}"
}

# ---------------------------------------------------------------------------
# gh
# ---------------------------------------------------------------------------
install_gh() {
    local arch version dir
    case "$(uname -m)" in
        x86_64|amd64)  arch=amd64 ;;
        aarch64|arm64) arch=arm64 ;;
        armv6l|armv7l) arch=armv6 ;;
        i386|i686)     arch=386 ;;
        *) say "no gh release is published for $(uname -m)."; return 1 ;;
    esac
    version="$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest 2>/dev/null \
               | python3 -c 'import json, sys
try:
    print(json.load(sys.stdin)["tag_name"].lstrip("v"))
except Exception:
    pass' 2>/dev/null)"
    [ -n "$version" ] || { say "could not find out which gh release is current."; return 1; }
    dir="$(mktemp -d)"
    if curl -fsSL "https://github.com/cli/cli/releases/download/v$version/gh_${version}_linux_${arch}.tar.gz" \
            | tar -xz -C "$dir" 2>/dev/null \
       && mkdir -p "$HOME/.local/bin" \
       && install -m 755 "$dir/gh_${version}_linux_${arch}/bin/gh" "$HOME/.local/bin/gh"; then
        rm -rf "$dir"
        hash -r
        say "installed gh $version in ~/.local/bin."
        return 0
    fi
    rm -rf "$dir"
    say "could not install gh."
    return 1
}

# gh reads $GH_TOKEN and $GITHUB_TOKEN before anything else, and refuses to
# store credentials while one of them is set -- and the deploy puts the API
# keys of the machine it came from in this shell's environment.  Every gh
# call here therefore runs without them.
gh_() { env -u GH_TOKEN -u GITHUB_TOKEN gh "$@"; }

login_with_gh() {
    local flags=(--hostname "$HOST" --git-protocol https --with-token) i failed=0 message
    # A gh from before April 2023 has no --insecure-storage; on such a version
    # plain text is what it does anyway.
    if gh auth login --help 2>/dev/null | grep -q -- '--insecure-storage'; then
        flags+=(--insecure-storage)
    fi
    # gh writes the tokens into hosts.yml in plain text, and rewriting a file
    # does not change the mode it already has.  So a hosts.yml that other
    # accounts on this machine can read is made unreadable to them *before* a
    # token goes into it, and a machine where that cannot be done does not get
    # the tokens at all -- there is no way to take back a secret that has
    # already been written where somebody else can read it.  A hosts.yml that
    # does not exist yet is created by gh at mode 600.
    if [ -f "$HOSTS_YML" ] && ! chmod 600 "$HOSTS_YML"; then
        say "ERROR: other accounts on this machine can read $HOSTS_YML and that" \
            "cannot be changed, so the tokens are not stored in it."
        return 1
    fi
    # In the order they arrived, so that the account that is active on the
    # machine the credentials came from is the last one logged in, and so the
    # active one here as well.
    for (( i = 0; i < ${#LOGINS[@]}; i++ )); do
        if message="$(printf '%s\n' "${TOKENS[$i]}" | gh_ auth login "${flags[@]}" 2>&1)"; then
            say "logged ${LOGINS[$i]} in to $HOST."
        else
            say "WARNING: could not log ${LOGINS[$i]} in to $HOST: $message"
            failed=1
        fi
    done
    [ "$failed" = "0" ] || return 1
    # setup-git empties the list of credential helpers for github.com before
    # it adds its own -- that is how it makes sure gh is the one that answers
    # -- so the file it does that in is kept first, and it does not run if
    # that copy could not be made.
    keep_before_change "$HOME/.gitconfig" "gitconfig-before-sorcar" || return 1
    if message="$(gh_ auth setup-git --hostname "$HOST" 2>&1)"; then
        say "git uses gh for $HOST URLs."
    else
        say "WARNING: gh auth setup-git failed: $message"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# git on its own
# ---------------------------------------------------------------------------
login_with_git() {
    local names new i target
    command -v git >/dev/null 2>&1 || { say "WARNING: no git and no gh here."; return 1; }
    # A ~/.git-credentials that is a link into a password store or a dotfiles
    # repository is written where it really lives: replacing the link with a
    # regular file would quietly detach it from what maintains it.
    if [ -L "$CREDENTIALS" ]; then
        target="$(readlink -f "$CREDENTIALS" 2>/dev/null || true)"
        [ -n "$target" ] && CREDENTIALS="$target"
    fi
    # Both files this function writes are copied before either is touched: the
    # credentials, and ~/.gitconfig, which gains the ``store`` helper below.
    # A copy that cannot be made stops the whole thing, including when the gh
    # path already tried and failed to keep the same copy.
    keep_before_change "$CREDENTIALS" "git-credentials-before-sorcar" || return 1
    keep_before_change "$HOME/.gitconfig" "gitconfig-before-sorcar" || return 1
    names="$(printf '%s,' "${LOGINS[@]}")"
    new="$CREDENTIALS.sorcar-new"
    rm -f "$new"
    : > "$new"
    chmod 600 "$new"
    # The accounts that arrived go first, and backwards, so that the one that
    # is active on the machine they came from is the very first line: asked
    # about github.com without a user name -- which is what an ordinary https
    # remote does -- git takes the first line that matches and leaves the
    # rest, including any github.com account this machine already had.
    for (( i = ${#LOGINS[@]} - 1; i >= 0; i-- )); do
        printf 'https://%s:%s@%s\n' "${LOGINS[$i]}" "${TOKENS[$i]}" "$HOST" >> "$new"
    done
    # Every line for another host, and every line for an account this payload
    # says nothing about, is somebody else's credential and stays.
    if [ -f "$CREDENTIALS" ]; then
        awk -v names="$names" -v host="$HOST" '
            BEGIN { n = split(names, a, ","); for (i = 1; i <= n; i++) if (a[i] != "") mine[a[i]] = 1 }
            {
                keep = 1
                if (index($0, "@" host) > 0) {
                    user = substr($0, index($0, "://") + 3)
                    user = substr(user, 1, index(user, ":") - 1)
                    if (user in mine) keep = 0
                }
                if (keep) print
            }' "$CREDENTIALS" >> "$new"
    fi
    mv -f "$new" "$CREDENTIALS"
    # --add, so a helper that is already configured here keeps working: git
    # asks them in turn and uses the first that answers.
    if ! git config --global --get-all "credential.https://$HOST.helper" 2>/dev/null \
         | grep -qx 'store'; then
        git config --global --add "credential.https://$HOST.helper" store
    fi
    say "git reads the credentials of ${#LOGINS[@]} account(s) from ~/.git-credentials."
}

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
keep_before_change "$HOSTS_YML" "gh-hosts-before-sorcar" || exit 1
command -v gh >/dev/null 2>&1 || install_gh || true

if command -v gh >/dev/null 2>&1; then
    mkdir -p "$GH_CONFIG_HOME"
    login_with_gh || { say "falling back to git's own credential store."; login_with_git; }
else
    say "gh is not available here; installing the credentials for git only."
    login_with_git
fi

# What is here now, tokens masked -- ``gh auth status`` only reveals one when
# it is asked to with --show-token.
if command -v gh >/dev/null 2>&1; then
    gh_ auth status --hostname "$HOST" 2>&1 | sed 's/^/         /' || true
fi
