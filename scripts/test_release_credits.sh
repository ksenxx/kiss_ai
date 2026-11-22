#!/bin/bash
# End-to-end test for who public history names, as built by scripts/release.sh.
# Run: bash scripts/test_release_credits.sh
# Builds a scratch repo + bare "public" remote whose releases were published
# under a leaked placeholder identity, then verifies that .mailmap replaces
# that identity on every release commit and tag, that scripts/credits.txt
# puts Co-authored-by trailers on the named release only, that the rewritten
# chain rebuilds to itself, and that dropping a credit removes the trailer.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_SH="$REPO_ROOT/scripts/release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

OWNER="Owner <owner@users.noreply.github.com>"
JANE="Jane Doe <1+jane@users.noreply.github.com>"
RAVI="Ravi K <2+ravi@users.noreply.github.com>"

# Print every author, committer and tagger identity reachable in <git-dir>.
identities() {
    git -C "$1" log --all --format='%an <%ae>%n%cn <%ce>'
    git -C "$1" for-each-ref --format='%(taggername) %(taggeremail)' refs/tags
}

# Push the rebuilt chain and its re-pointed tags the way a release does.
push_chain() {
    git push -q --atomic "$WORK/public.git" \
        "--force-with-lease=refs/heads/main:$(git rev-parse "${PUBLIC_BRANCH_NS}/main")" \
        "${RETAG_LEASES[@]}" "${RETAG_REFSPECS[@]}" "${RELEASE_CHAIN_TIP}:refs/heads/main"
    fetch_public_refs
}

git init -q -b main "$WORK/repo"
git init -q -b main --bare "$WORK/public.git"
cd "$WORK/repo"
git config user.email "test@test.com"
git config user.name "Test"
git remote add public "$WORK/public.git"
mkdir -p scripts
echo '["scripts/exclude.json"]' > scripts/exclude.json
echo one > app.py
git add -A && git commit -q -m "first"

source "$RELEASE_SH"
PUBLIC_REPO_URL="$WORK/public.git"
PUBLIC_REPO_SSH="$WORK/public.git"

# Two releases published while the checkout carried the placeholder identity.
create_public_commit "$(git rev-parse HEAD)" "1.0.0" "" > /dev/null
push_public_snapshot "$PUBLIC_COMMIT" v1.0.0 "$(release_tag v1.0.0 1.0.0 "$PUBLIC_COMMIT")" "" > /dev/null
echo two > app.py
git commit -q -am "second"
FIRST_RELEASE="$PUBLIC_COMMIT"
create_public_commit "$(git rev-parse HEAD)" "1.1.0" "$FIRST_RELEASE" > /dev/null
push_public_snapshot "$PUBLIC_COMMIT" v1.1.0 "$(release_tag v1.1.0 1.1.0 "$PUBLIC_COMMIT")" "$FIRST_RELEASE" > /dev/null
OLD_TIP="$PUBLIC_COMMIT"
OLD_DATES=$(git log --format='%ad %cd' --date=raw "$OLD_TIP")

# --- Test 1: no .mailmap, no credits file: identity copied, chain reused ---
[[ "$(identities "$WORK/public.git" | sort -u)" == "Test <test@test.com>" ]] ||
    fail "setup: public history is not all 'Test': $(identities "$WORK/public.git" | sort -u)"
[[ "$(git log -1 --format=%B "$OLD_TIP")" == "Release 1.1.0" ]] || fail "release message is not the bare subject"
build_release_chain > /dev/null
[[ "$RELEASE_CHAIN_TIP" == "$OLD_TIP" && ${#RETAG_REFSPECS[@]} -eq 0 ]] ||
    fail "chain without .mailmap and credits was not reused as is"
pass "without .mailmap and credits the identity is copied and the chain is reused"

# --- Test 2: .mailmap and credits rewrite every release commit and tag ---
echo "$OWNER Test <test@test.com>" > .mailmap
cat > scripts/credits.txt <<EOF
# <release> <Name> <email>

1.0.0 $JANE
1.0.0 $RAVI
1.1 Nobody <nobody@example.org>
EOF
build_release_chain > "$WORK/chain.log"
grep -q "2 public release tag(s) do not point at release-only commits yet" "$WORK/chain.log" ||
    fail "re-pointing not announced: $(cat "$WORK/chain.log")"
[[ ${#RETAG_REFSPECS[@]} -eq 2 && "$RELEASE_CHAIN_TIP" != "$OLD_TIP" ]] || fail "chain was not rebuilt"
push_chain
[[ "$(identities "$WORK/public.git" | sort -u)" == "$OWNER" ]] ||
    fail "placeholder identity survived: $(identities "$WORK/public.git" | sort -u)"
[[ "$(git log --format='%ad %cd' --date=raw "$RELEASE_CHAIN_TIP")" == "$OLD_DATES" ]] ||
    fail "rebuilt commits did not keep their dates"
[[ "$(git -C "$WORK/public.git" log -1 --format=%B 'v1.0.0^{commit}')" == \
    "Release 1.0.0"$'\n\n'"Co-authored-by: $JANE"$'\n'"Co-authored-by: $RAVI" ]] ||
    fail "Release 1.0.0 lacks its credits: $(git -C "$WORK/public.git" log -1 --format=%B 'v1.0.0^{commit}')"
[[ "$(git -C "$WORK/public.git" log -1 --format=%B main)" == "Release 1.1.0" ]] ||
    fail "a credit for '1.1' leaked onto Release 1.1.0"
[[ "$(git -C "$WORK/public.git" rev-parse 'main~1')" == "$(git -C "$WORK/public.git" rev-parse 'v1.0.0^{commit}')" &&
    "$(git -C "$WORK/public.git" cat-file -t refs/tags/v1.1.0)" == "tag" ]] ||
    fail "tags were not re-pointed to the rebuilt commits as annotated tags"
[[ "$(git -C "$WORK/public.git" diff --stat "$OLD_TIP" main)" == "" ]] || fail "rebuild changed the published tree"
NEW_TIP="$RELEASE_CHAIN_TIP"
build_release_chain > /dev/null
[[ "$RELEASE_CHAIN_TIP" == "$NEW_TIP" && ${#RETAG_REFSPECS[@]} -eq 0 ]] || fail "rebuilt chain is not reused"
pass ".mailmap replaced the placeholder identity; credits.txt added trailers to the named release only"

# --- Test 3: a new release cut under the placeholder identity is canonical ---
echo three > app.py
git commit -q -am "third"
[[ "$(git log -1 --format='%an <%ae>')" == "Test <test@test.com>" ]] || fail "setup: dev commit is not by Test"
create_public_commit "$(git rev-parse HEAD)" "1.2.0" "$NEW_TIP" > /dev/null
[[ "$(git log -1 --format='%an <%ae> %cn <%ce>' "$PUBLIC_COMMIT")" == "$OWNER $OWNER" ]] ||
    fail "new release commit carries $(git log -1 --format='%an <%ae> %cn <%ce>' "$PUBLIC_COMMIT")"
TAG_OBJECT=$(release_tag v1.2.0 1.2.0 "$PUBLIC_COMMIT")
git cat-file tag "$TAG_OBJECT" | grep -q "^tagger $OWNER " || fail "new release tag is not tagged by the owner"
pass "a release cut under the placeholder identity is published under the mapped one"

# --- Test 4: dropping a credit removes its trailer from public history ---
grep -v "Ravi" scripts/credits.txt > "$WORK/credits" && mv "$WORK/credits" scripts/credits.txt
build_release_chain > /dev/null
[[ ${#RETAG_REFSPECS[@]} -eq 2 ]] || fail "dropping a credit did not rebuild the chain"
push_chain
[[ "$(git -C "$WORK/public.git" log --all --format=%B | grep -c 'Co-authored-by')" == "1" ]] ||
    fail "expected exactly Jane's trailer: $(git -C "$WORK/public.git" log --all --format=%B)"
git -C "$WORK/public.git" log --all --format=%B | grep -q "Co-authored-by: $JANE" || fail "Jane lost her credit"
pass "dropping a credit line removes the trailer on the next rebuild"

echo
echo "ALL TESTS PASSED"
