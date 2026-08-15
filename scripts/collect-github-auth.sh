#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Collect this machine's GitHub.com credentials, for
# scripts/install-github-auth.sh to install on the remote machine.
#
# Usage:  scripts/collect-github-auth.sh            (payload on stdout)
#
# The payload is one line per account,
#
#     account <login> <token>
#
# with the account that is active here printed LAST: the installer logs the
# accounts in in the order they arrive, and gh makes the last login the
# active one.  Everything else -- warnings, what was found where -- goes to
# stderr, so stdout is the payload and nothing else.  A token is never
# printed to stderr, and never appears as a command-line argument (the
# arguments of a process are readable by anyone on the machine, its standard
# input is not).
#
# Where the credentials are looked for, in order:
#
#   1. ``gh``.  This is where they are on a machine somebody has run
#      ``gh auth login`` on, and it is the only source that knows about more
#      than one account.  On macOS the token itself lives in the keychain --
#      ~/.config/gh/hosts.yml then holds the account names and no token at
#      all -- so it is read back with ``gh auth token``.
#   2. git's own credential helpers, through ``git credential fill``: the
#      macOS keychain, ~/.git-credentials, whatever else is configured.
#   3. $GH_TOKEN / $GITHUB_TOKEN, which is how a machine that runs
#      automation has one.
#
# Each token is then offered to https://api.github.com/user.  That proves
# GitHub still accepts it -- shipping a revoked token would leave the remote
# failing at its first push with nothing to explain it -- and the answer
# carries the login GitHub knows the token by, which is the name the remote
# stores it under.  It says nothing about what the token is allowed to do;
# the scopes are the ones the account gave it, here and there alike.  Only an
# outright rejection drops a token: anything else, including no answer at
# all, sends it with a warning.
#
# ``set -e`` is deliberately not used: every source here is allowed to fail,
# that is what makes it a list of sources.  The exit status says whether
# anything was found (0) or not (1).
set -uo pipefail

HOST="github.com"
API_URL="https://api.github.com/user"

note() { printf '%s\n' "$*" >&2; }

# The whole payload, built up line by line.  A string rather than an array
# because this runs on macOS, whose /bin/bash is 3.2.
PAYLOAD=""

# ---------------------------------------------------------------------------
# gh
# ---------------------------------------------------------------------------

# The accounts gh is logged in to on this host, the active one last.
#
# ``gh auth status`` is the only interface that lists them; it prints
# unadorned text when it is not talking to a terminal, which is the case
# here.  The account name is read as the word after " account " rather than
# by field number, so a line gaining or losing a leading tick still parses.
#
# Both streams are read and the exit status is ignored on purpose: when any
# one account cannot be checked -- a token somebody revoked, a timeout -- gh
# prints the whole report to standard error and exits 1.  Reading only
# standard output would then find no accounts at all, and the accounts that
# are perfectly fine would be left behind because of the one that is not.
# The failure lines say "Failed to log in to", which is not "Logged in to",
# so they do not parse as accounts.
#
# Listing more than one account needs gh 2.40 or newer (December 2023),
# which is where multiple accounts, "Active account:" and ``auth token
# --user`` arrived.  An older gh parses as nothing here and the credentials
# are picked up from git's helpers below instead.
gh_logins() {
    command -v gh >/dev/null 2>&1 || return 0
    { gh auth status --hostname "$HOST" 2>&1 || true; } | awk '
        index($0, " account ") > 0 && index($0, "Logged in to ") > 0 {
            rest = substr($0, index($0, " account ") + 9)
            split(rest, word, " ")
            name = word[1]
            order[++n] = name
        }
        index($0, "Active account: true") > 0 { active = name }
        END {
            for (i = 1; i <= n; i++) if (order[i] != active) print order[i]
            if (active != "") print active
        }'
}

# The token gh holds for one account.  ``--user`` is what makes the accounts
# that are not active reachable; a gh too old to have it can only answer for
# the active account, which is the right answer when that is the only one.
gh_token() {
    local login="$1" only="$2" token
    token="$(gh auth token --hostname "$HOST" --user "$login" 2>/dev/null | tr -d '[:space:]')"
    if [ -z "$token" ] && [ "$only" = "1" ]; then
        token="$(gh auth token --hostname "$HOST" 2>/dev/null | tr -d '[:space:]')"
    fi
    printf '%s' "$token"
}

# ---------------------------------------------------------------------------
# git's credential helpers
# ---------------------------------------------------------------------------

# What git itself would use to talk to github.com, as "<username> <token>".
# GIT_TERMINAL_PROMPT=0 keeps a helper that has nothing from asking for it.
git_credential() {
    command -v git >/dev/null 2>&1 || return 0
    printf 'protocol=https\nhost=%s\n\n' "$HOST" \
        | GIT_TERMINAL_PROMPT=0 git credential fill 2>/dev/null \
        | awk '
            index($0, "username=") == 1 { user = substr($0, 10) }
            index($0, "password=") == 1 { pass = substr($0, 10) }
            END { if (pass != "") print user " " pass }'
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Ask GitHub what login this token belongs to, as "<http status> <login>".
#
# The token goes to curl through a configuration file on standard input, so
# that it is not in curl's arguments where ``ps`` would show it.  ``-q`` has
# to come first: without it curl also reads ~/.curlrc, and a ``verbose`` or
# ``trace`` line there would print the Authorization header -- the token
# itself -- into the deploy's output or into a file.  The timeouts are what
# keeps a black hole between here and GitHub from stalling a deploy at its
# last step; a check that times out is a check that did not happen, which is
# handled below.
api_login() {
    local token="$1" body http login
    body="$(mktemp)" || return 1
    http="$(printf 'header = "Authorization: Bearer %s"\n' "$token" \
            | curl -q -K - -s --connect-timeout 10 --max-time 30 \
                   -o "$body" -w '%{http_code}' \
                   -H 'Accept: application/vnd.github+json' "$API_URL")"
    login="$(python3 -c 'import json, sys
try:
    print(json.load(open(sys.argv[1])).get("login", ""))
except Exception:
    pass' "$body" 2>/dev/null)"
    rm -f "$body"
    printf '%s %s' "${http:-000}" "$login"
}

# Add one account to the payload, unless GitHub says the token is dead or the
# same token is already there (gh and git's helpers hand back the same one).
add_account() {
    local login="$1" token="$2" answer http verified
    [ -n "$token" ] || return 0
    case "$PAYLOAD" in
        *" $token"$'\n'*) return 0 ;;
    esac

    answer="$(api_login "$token")"
    http="${answer%% *}"
    verified="${answer#* }"
    case "$http" in
        200)
            [ -n "$verified" ] && login="$verified"
            ;;
        401)
            note "warning: GitHub no longer accepts the token for ${login:-this account}" \
                 "-- it is not copied."
            return 0
            ;;
        *)
            # Only 401 means "this credential is not valid".  403 is what a
            # rate limit answers, and what a perfectly good token gets for a
            # while after somebody else mistyped a password, so a token is
            # not thrown away over one.
            note "warning: GitHub did not confirm ${login:-this account}'s token" \
                 "(HTTP $http); copying it unchecked."
            ;;
    esac
    [ -n "$login" ] || login="x-access-token"
    PAYLOAD="${PAYLOAD}account $login $token"$'\n'
    note "found the credentials of $login"
}

# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------
LOGINS="$(gh_logins)"
LOGIN_COUNT="$(printf '%s' "$LOGINS" | grep -c . || true)"
if [ -n "$LOGINS" ]; then
    ONLY=0
    [ "$LOGIN_COUNT" = "1" ] && ONLY=1
    while IFS= read -r login; do
        [ -n "$login" ] || continue
        add_account "$login" "$(gh_token "$login" "$ONLY")"
    done <<< "$LOGINS"
fi

if [ -z "$PAYLOAD" ]; then
    CRED="$(git_credential)"
    [ -n "$CRED" ] && add_account "${CRED%% *}" "${CRED#* }"
fi

if [ -z "$PAYLOAD" ]; then
    for env_token in "${GH_TOKEN:-}" "${GITHUB_TOKEN:-}"; do
        [ -n "$env_token" ] || continue
        add_account "" "$env_token"
    done
fi

if [ -z "$PAYLOAD" ]; then
    note "no GitHub.com credentials on this machine (looked at gh, git's credential" \
         "helpers, \$GH_TOKEN and \$GITHUB_TOKEN)."
    exit 1
fi

printf '%s' "$PAYLOAD"
