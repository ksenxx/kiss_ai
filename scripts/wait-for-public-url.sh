#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Wait for the kiss-web just (re)started on this machine to answer locally,
# and then for the public tunnel URL it publishes.
#
# Usage:  bash ~/.kiss/wait-for-public-url.sh <port> [remote-url.json] [kiss-web-stderr.log]
#
# Prints ``SORCAR_PUBLIC_URL=<url>`` as its last line once a URL exists.
# Exit 1 when kiss-web does not answer locally, or publishes no URL at all.
#
# ./rsorcar ships this into ~/.kiss (step 4) and runs it at the end of its
# remote bootstrap (step 7d).  The URL is verified from *this* machine as a
# courtesy only: the deploying machine verifies it again (rsorcar step 9),
# and that is the verdict that matters, because this machine's own view of
# its fresh tunnel is the one that goes wrong.  A quick tunnel's hostname
# reaches DNS a few seconds after cloudflared reports it, and a stub
# resolver (systemd-resolved, the default on Debian and Ubuntu cloud images)
# asked in those seconds caches the "no such name" answer for the zone's
# negative TTL -- 30 minutes for trycloudflare.com -- while the rest of the
# world resolves the name fine.  So a name that does not resolve gets the
# resolver's cache dropped before the next attempt (``sudo -n resolvectl
# flush-caches``, when this machine allows it), and a URL that still does not
# answer from here after the wait is a warning, not a failed deploy: the
# deploy is finished, the web app is up, and the URL is checked from outside
# next.
#
#   WAIT_LOCAL_TRIES   attempts, 1 s apart, for the local endpoint (90)
#   WAIT_TUNNEL_TRIES  attempts, 3 s apart, for the public URL (60)
set -euo pipefail

PORT="${1:-}"
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "Usage: $0 <port> [remote-url.json] [kiss-web-stderr.log]" >&2; exit 2; }
URL_FILE="${2:-$HOME/.kiss/remote-url.json}"
LOG_FILE="${3:-$HOME/.kiss/kiss-web-stderr.log}"
LOCAL_TRIES="${WAIT_LOCAL_TRIES:-90}"
TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-60}"

rinfo() { printf '\033[0;36m[%s]\033[0m %s\n' "$(hostname -s 2>/dev/null || hostname)" "$*"; }

# --- The local endpoint ---------------------------------------------------------
code=""
for _ in $(seq 1 "$LOCAL_TRIES"); do
    code="$(curl -sk -o /dev/null -w '%{http_code}' "https://127.0.0.1:$PORT/" 2>/dev/null || true)"
    [ "$code" = "200" ] && break
    sleep 1
done
if [ "$code" != "200" ]; then
    echo "ERROR: kiss-web did not answer on https://127.0.0.1:$PORT" >&2
    tail -20 "$LOG_FILE" >&2 2>/dev/null || true
    exit 1
fi
rinfo "kiss-web answers on https://127.0.0.1:$PORT"

# --- The public URL -------------------------------------------------------------
# The file is rewritten if the tunnel is replaced, so it is re-read on every
# attempt instead of trusting the first value.
url=""
candidate=""
code=""
for _ in $(seq 1 "$TUNNEL_TRIES"); do
    candidate="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("tunnel", ""))' \
                 "$URL_FILE" 2>/dev/null || true)"
    if [ -n "$candidate" ]; then
        rc=0
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$candidate/" 2>/dev/null)" || rc=$?
        if [ "$code" = "200" ]; then
            url="$candidate"
            break
        fi
        # curl 6: the name did not resolve.  Asked before the name existed,
        # systemd-resolved keeps that answer for the zone's negative TTL;
        # dropping its cache lets the next attempt ask DNS again.
        if [ "$rc" = 6 ] && command -v resolvectl >/dev/null 2>&1; then
            sudo -n resolvectl flush-caches >/dev/null 2>&1 || true
        fi
    fi
    sleep 3
done
if [ -n "$url" ]; then
    rinfo "Public tunnel is live: $url"
elif [ -n "$candidate" ]; then
    rinfo "WARNING: $candidate does not answer from this machine yet (HTTP ${code:-000}); the deploying machine verifies it next."
    url="$candidate"
else
    echo "ERROR: kiss-web published no public tunnel URL (no tunnel in $URL_FILE)" >&2
    tail -20 "$LOG_FILE" >&2 2>/dev/null || true
    exit 1
fi
echo "SORCAR_PUBLIC_URL=$url"
