#!/bin/bash
# End-to-end test for scripts/wait-for-public-url.sh.
# Run: bash scripts/test_wait_for_public_url.sh
#
# The script polls with curl, reads ~/.kiss/remote-url.json and, when a name
# does not resolve, drops systemd-resolved's cache through ``sudo -n
# resolvectl flush-caches``.  Here curl, sudo and resolvectl are PATH stubs
# whose behaviour a fixture file drives, so each case runs in seconds.
#   1. kiss-web never answers locally: exit 1, the log's tail is shown, the
#      tunnel is never polled;
#   2. the URL appears and answers 200: "Public tunnel is live", the last
#      line is SORCAR_PUBLIC_URL=<url>, no cache is flushed;
#   3. the name never resolves (curl exit 6): the cache is flushed before
#      each retry, and the run ends with the WARNING, the URL line and exit 0
#      -- the deploying machine verifies the URL, not this one;
#   4. the name resolves after a flush: live, and the flush happened once;
#   5. no URL is ever published: exit 1 with the "published no public tunnel
#      URL" error;
#   6. a URL that answers something other than 200 (Cloudflare's 530 while
#      routing) is retried and ends as a WARNING, with no flush;
#   7. no port is a usage error.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/wait-for-public-url.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

mkdir -p "$WORK/bin" "$WORK/home/.kiss"
URL_FILE="$WORK/home/.kiss/remote-url.json"
LOG_FILE="$WORK/home/.kiss/kiss-web-stderr.log"
echo "the web app's last words" > "$LOG_FILE"
TUNNEL=https://fresh-quick-tunnel.trycloudflare.com

# curl stub.  LOCAL_CODE: what https://127.0.0.1:<port>/ answers.  For the
# tunnel URL, the n-th call takes the n-th word of TUNNEL_PLAN (the last word
# repeats): "200" answers 200, "6" fails to resolve (exit 6, prints 000),
# "530" answers 530.  Every call is logged.
cat > "$WORK/bin/curl" <<'EOF'
#!/bin/bash
url="${@: -1}"
echo "curl $url" >> "$FIX/calls.log"
case "$url" in
    https://127.0.0.1:*) printf '%s' "${LOCAL_CODE:-200}"; exit 0 ;;
esac
n=$(grep -c '^curl https://fresh' "$FIX/calls.log")
set -- $TUNNEL_PLAN
step="${!n:-${@: -1}}"
case "$step" in
    6) printf '000'; exit 6 ;;
    *) printf '%s' "$step"; exit 0 ;;
esac
EOF
cat > "$WORK/bin/sudo" <<'EOF'
#!/bin/bash
echo "sudo $*" >> "$FIX/calls.log"
exit 0
EOF
printf '#!/bin/bash\nexit 0\n' > "$WORK/bin/resolvectl"
chmod +x "$WORK/bin/"*

# $1: LOCAL_CODE, $2: TUNNEL_PLAN, $3: 1 to write the URL file; prints the
# combined output and returns the script's status.
run() {
    : > "$WORK/calls.log"
    rm -f "$URL_FILE"
    [ "${3:-1}" = 1 ] && printf '{"local": "https://localhost:8787", "tunnel": "%s"}\n' "$TUNNEL" > "$URL_FILE"
    FIX="$WORK" LOCAL_CODE="$1" TUNNEL_PLAN="$2" HOME="$WORK/home" PATH="$WORK/bin:$PATH" \
    WAIT_LOCAL_TRIES=2 WAIT_TUNNEL_TRIES=3 \
        bash "$SCRIPT" 8787 "$URL_FILE" "$LOG_FILE" 2>&1
}

# --- 1. kiss-web never answers locally ---------------------------------------------
if OUT="$(run 000 200)"; then fail "a dead local endpoint passed: $OUT"; fi
echo "$OUT" | grep -q "ERROR: kiss-web did not answer on https://127.0.0.1:8787" || fail "no local error: $OUT"
echo "$OUT" | grep -q "the web app's last words" || fail "the log tail is not shown: $OUT"
grep -q "curl https://fresh" "$WORK/calls.log" && fail "the tunnel was polled although kiss-web is down"
pass "a web app that does not come up fails the wait, with its log"

# --- 2. the URL answers -------------------------------------------------------------------
OUT="$(run 200 200)" || fail "a live tunnel failed: $OUT"
echo "$OUT" | grep -q "kiss-web answers on https://127.0.0.1:8787" || fail "the local answer is not reported: $OUT"
echo "$OUT" | grep -q "Public tunnel is live: $TUNNEL" || fail "the live tunnel is not reported: $OUT"
[ "$(echo "$OUT" | tail -1)" = "SORCAR_PUBLIC_URL=$TUNNEL" ] || fail "the last line is not the URL line: $OUT"
grep -q "sudo" "$WORK/calls.log" && fail "the cache was flushed although the name resolved"
pass "a URL that answers is reported live, the URL line last"

# --- 3. the name never resolves from this machine ----------------------------------------
OUT="$(run 200 6)" || fail "an unresolvable name failed the deploy: $OUT"
echo "$OUT" | grep -q "WARNING: $TUNNEL does not answer from this machine yet (HTTP 000); the deploying machine verifies it next" \
    || fail "no warning for the unresolvable name: $OUT"
[ "$(echo "$OUT" | tail -1)" = "SORCAR_PUBLIC_URL=$TUNNEL" ] || fail "the URL line is missing after the warning: $OUT"
[ "$(grep -c 'sudo -n resolvectl flush-caches' "$WORK/calls.log")" = 3 ] \
    || fail "the resolver cache was not flushed before each retry: $(cat "$WORK/calls.log")"
[ "$(grep -c 'curl https://fresh' "$WORK/calls.log")" = 3 ] || fail "the tunnel was not polled three times"
pass "a name this machine cannot resolve is a warning, with the cache flushed before each retry"

# --- 4. the name resolves after a flush ----------------------------------------------------
OUT="$(run 200 "6 200")" || fail "a name that resolved on the second try failed: $OUT"
echo "$OUT" | grep -q "Public tunnel is live: $TUNNEL" || fail "not reported live after the flush: $OUT"
[ "$(grep -c 'sudo -n resolvectl flush-caches' "$WORK/calls.log")" = 1 ] || fail "expected exactly one flush: $(cat "$WORK/calls.log")"
[ "$(grep -c 'curl https://fresh' "$WORK/calls.log")" = 2 ] || fail "expected exactly two tunnel polls"
pass "a name that resolves after the flush is reported live"

# --- 5. no URL is ever published ------------------------------------------------------------
if OUT="$(run 200 200 0)"; then fail "no published URL passed: $OUT"; fi
echo "$OUT" | grep -q "ERROR: kiss-web published no public tunnel URL (no tunnel in $URL_FILE)" || fail "no error for a missing URL: $OUT"
echo "$OUT" | grep -q "the web app's last words" || fail "the log tail is not shown: $OUT"
grep -q "curl https://fresh" "$WORK/calls.log" && fail "a URL was polled although none was published"
pass "a web app that publishes no URL fails the wait, with its log"

# --- 6. Cloudflare still routing (530) -------------------------------------------------------
OUT="$(run 200 530)" || fail "a 530 failed the deploy: $OUT"
echo "$OUT" | grep -q "WARNING: $TUNNEL does not answer from this machine yet (HTTP 530)" || fail "no warning for HTTP 530: $OUT"
grep -q "sudo" "$WORK/calls.log" && fail "the cache was flushed for an HTTP answer"
[ "$(grep -c 'curl https://fresh' "$WORK/calls.log")" = 3 ] || fail "a 530 was not retried"
pass "an HTTP answer other than 200 is retried, then a warning, and no cache flush"

# --- 7. usage --------------------------------------------------------------------------------
if OUT="$(bash "$SCRIPT" 2>&1)"; then fail "no port accepted: $OUT"; fi
echo "$OUT" | grep -q "^Usage:" || fail "no usage line: $OUT"
pass "a missing port is a usage error"

echo "ALL TESTS PASSED"
