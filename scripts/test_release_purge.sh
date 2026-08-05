#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

# End-to-end test for purge_public_history() in scripts/release.sh.
# Run: bash scripts/test_release_purge.sh
# Every test builds a scratch repo plus a real bare "public" remote that already
# holds UNFILTERED history (the legacy state of github.com/ksenxx/kiss_ai), runs
# the release script's purge against it, then inspects the bare remote itself.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_SH="$REPO_ROOT/scripts/release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

# Source release.sh functions without running main (BASH_SOURCE guard).
source "$RELEASE_SH"

# Create scratch repo "$1" with a bare public remote and cd into it. The release
# script only rewrites a remote whose URL is the configured public repo, so both
# spellings of that constant point at this scenario's throwaway remote.
new_scenario() {
    local name="$1"
    git init -q -b main "$WORK/$name"
    git init -q --bare "$WORK/$name-public.git"
    cd "$WORK/$name"
    git config user.email "test@test.com"
    git config user.name "Test"
    git config commit.gpgsign false
    git config tag.gpgsign false
    git remote add "$PUBLIC_REMOTE" "$WORK/$name-public.git"
    PUBLIC_REPO_URL="$WORK/$name-public.git"
    PUBLIC_REPO_SSH="$WORK/$name-public.git"
    mkdir -p scripts
}

# Write scripts/exclude.json listing the given paths (no arguments => []).
write_exclude() {
    python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@" > scripts/exclude.json
}

# Print each commit of <repo>'s refs whose tree contains <path>.
commits_containing() {
    local repo="$1" path="$2" commit
    for commit in $(git -C "$repo" rev-list --all); do
        if git -C "$repo" cat-file -e "$commit:$path" 2>/dev/null; then
            echo "$commit"
        fi
    done
}

# Print the number of commits reachable from <repo>'s branch main.
main_commit_count() {
    git -C "$1" rev-list --count main
}

# =============================================================================
# Scenario 1: legacy unfiltered history is fully purged
# =============================================================================
new_scenario legacy
PUB="$WORK/legacy-public.git"

write_exclude reports scripts/exclude.json
echo "code" > app.py
git add -A
git commit -q -m "c1 code"

mkdir reports
echo "<html>audit</html>" > reports/audit.html
git add -A
git commit -q -m "c2 reports only"
git tag -a v1 -m "v1"
REPORT_BLOB=$(git rev-parse "HEAD:reports/audit.html")
EXCLUDE_BLOB=$(git rev-parse "HEAD:scripts/exclude.json")

echo "more code" >> app.py
echo "second" > reports/second.html
git add -A
git commit -q -m "c3 mixed"
git tag -a v2 -m "v2"

git rm -q -r reports
git commit -q -m "c4 drop reports"
LOCAL_HEAD=$(git rev-parse HEAD)
# What a release would publish today: the tip's tree minus the excluded paths.
FINAL_TREE=$(filtered_tree "$LOCAL_HEAD")

# Publish the raw history, exactly as the old release script used to.
git push -q "$PUBLIC_REMOTE" main
git push -q "$PUBLIC_REMOTE" v1 v2
[[ "$(main_commit_count "$PUB")" == "4" ]] || fail "setup: public should start with 4 commits"

fetch_public_refs
[[ -n "$(excluded_paths_in_history "$(pwd)" refs/remotes/"$PUBLIC_REMOTE"/main)" ]] ||
    fail "excluded_paths_in_history missed the leaked history"
pass "excluded_paths_in_history detects excluded paths in published history"

purge_public_history > "$WORK/legacy.log" 2>&1 || fail "purge failed: $(cat "$WORK/legacy.log")"

[[ -z "$(commits_containing "$PUB" reports)" ]] || fail "reports/ still in purged public history"
[[ -z "$(commits_containing "$PUB" scripts/exclude.json)" ]] ||
    fail "scripts/exclude.json still in purged public history"
pass "no public commit contains an excluded path after the purge"

# Only c1 and c3 touched non-excluded files: "c2 reports only" (added a report)
# and "c4 drop reports" (deleted reports) have nothing left to say.
[[ "$(main_commit_count "$PUB")" == "2" ]] ||
    fail "expected 2 commits after pruning the reports-only commits, got $(main_commit_count "$PUB")"
for subject in "c2 reports only" "c4 drop reports"; do
    git -C "$PUB" log --format='%s' main | grep -qx "$subject" &&
        fail "commit '$subject' only touched excluded paths but survived"
done
pass "commits whose only content was excluded paths are gone"

[[ "$(git -C "$PUB" rev-parse "main^{tree}")" == "$FINAL_TREE" ]] ||
    fail "purged tip does not match the filtered tree of the published tip"
[[ "$(git -C "$PUB" show main:app.py)" == "$(printf 'code\nmore code')" ]] ||
    fail "code change from the mixed commit was lost"
for subject in "c1 code" "c3 mixed"; do
    git -C "$PUB" log --format='%s' main | grep -qx "$subject" || fail "lost commit '$subject'"
done
pass "non-excluded content and commits are preserved verbatim"

[[ "$(git -C "$PUB" cat-file -t refs/tags/v2)" == "tag" ]] || fail "v2 is no longer annotated"
[[ "$(git -C "$PUB" log -1 --format='%s' v2)" == "c3 mixed" ]] || fail "v2 re-pointed to the wrong commit"
[[ "$(git -C "$PUB" cat-file -t refs/tags/v1)" == "tag" ]] || fail "v1 is no longer annotated"
[[ -z "$(commits_containing "$PUB" reports)" ]] || fail "a tag still reaches excluded content"
pass "annotated tags stay annotated and point at purged commits"

git -C "$PUB" rev-list --objects --all | grep -q "$REPORT_BLOB" && fail "report blob still reachable"
git -C "$PUB" rev-list --objects --all | grep -q "$EXCLUDE_BLOB" && fail "exclude.json blob still reachable"
pass "excluded blobs are unreachable from every public ref"

[[ "$(git rev-parse HEAD)" == "$LOCAL_HEAD" ]] || fail "purge moved the local branch"
git cat-file -e "HEAD~1:reports/second.html" || fail "purge rewrote the local (origin) history"
[[ "$(git remote get-url "$PUBLIC_REMOTE")" == "$PUB" ]] || fail "purge dropped the public remote"
git diff --quiet && git diff --cached --quiet || fail "purge dirtied the working tree"
pass "local repo, its history, remotes and working tree are untouched"

[[ "$(git rev-parse "$PUBLIC_REMOTE/main")" == "$(git -C "$PUB" rev-parse main)" ]] ||
    fail "local mirror of public/main not refreshed after the purge"
pass "public/main mirror is refreshed to the rewritten tip"

# =============================================================================
# Scenario 2: purging an already clean history is a no-op
# =============================================================================
PURGED_HEAD=$(git -C "$PUB" rev-parse main)
purge_public_history > "$WORK/legacy2.log" 2>&1 || fail "second purge failed"
grep -q "no excluded paths in any kiss_ai branch or tag" "$WORK/legacy2.log" ||
    fail "clean history was not reported as clean: $(cat "$WORK/legacy2.log")"
[[ "$(git -C "$PUB" rev-parse main)" == "$PURGED_HEAD" ]] || fail "no-op purge still rewrote history"
pass "purge is idempotent: clean public history is left alone"

# =============================================================================
# Scenario 3: a path added to exclude.json later is purged from old commits
# =============================================================================
new_scenario grew
PUB="$WORK/grew-public.git"

write_exclude
echo "code" > app.py
echo "secret" > old_secret.txt
git add -A
git commit -q -m "c1 code and secret"
git rm -q old_secret.txt
git commit -q -m "c2 remove secret"
git push -q "$PUBLIC_REMOTE" main
fetch_public_refs

purge_public_history > "$WORK/grew1.log" 2>&1 || fail "purge with empty exclude list failed"
[[ "$(main_commit_count "$PUB")" == "2" ]] || fail "empty exclude list rewrote history"
[[ ! -s "$WORK/grew1.log" ]] || fail "empty exclude list produced output: $(cat "$WORK/grew1.log")"
pass "empty exclude list makes the purge a silent no-op"

# The tip is already clean, so only a history purge can remove the secret.
write_exclude old_secret.txt
[[ -z "$(excluded_paths_in_history "$(pwd)" main)" ]] &&
    fail "setup: secret should still be in history"
purge_public_history > "$WORK/grew2.log" 2>&1 || fail "purge failed: $(cat "$WORK/grew2.log")"
[[ -z "$(commits_containing "$PUB" old_secret.txt)" ]] || fail "newly excluded path survived the purge"
[[ "$(main_commit_count "$PUB")" == "1" ]] ||
    fail "commit that only removed the secret should be pruned, got $(main_commit_count "$PUB")"
[[ "$(git -C "$PUB" show main:app.py)" == "code" ]] || fail "code lost while purging the secret"
pass "path added to exclude.json after publication is purged from old commits"

# =============================================================================
# Scenario 4: excluded content reachable only from a tag is purged too
# =============================================================================
new_scenario tagonly
PUB="$WORK/tagonly-public.git"

write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1 code"
git checkout -q -b side
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "side reports"
git tag -a legacy-tag -m "legacy"
git checkout -q main
echo "more code" >> app.py
git add -A
git commit -q -m "c2 code"
git push -q "$PUBLIC_REMOTE" main
git push -q "$PUBLIC_REMOTE" legacy-tag
[[ -n "$(commits_containing "$PUB" reports)" ]] || fail "setup: tag should publish reports/"

fetch_public_refs
purge_public_history > "$WORK/tagonly.log" 2>&1 || fail "purge failed: $(cat "$WORK/tagonly.log")"
[[ -z "$(commits_containing "$PUB" reports)" ]] ||
    fail "content reachable only from a tag was not purged"
[[ "$(main_commit_count "$PUB")" == "2" ]] || fail "main history changed unexpectedly"
# The tagged commit only added a report, so it is pruned and the tag must be
# re-pointed at its surviving ancestor rather than deleted or left dangling.
[[ "$(git -C "$PUB" cat-file -t refs/tags/legacy-tag)" == "tag" ]] ||
    fail "legacy-tag is missing or no longer annotated"
[[ "$(git -C "$PUB" log -1 --format='%s' refs/tags/legacy-tag)" == "c1 code" ]] ||
    fail "legacy-tag was not re-pointed at its surviving ancestor"
pass "excluded content reachable only from a tag is purged and the tag is remapped"

# =============================================================================
# Scenario 5: merge commits survive the purge
# =============================================================================
new_scenario merged
PUB="$WORK/merged-public.git"

write_exclude reports
echo "base" > app.py
git add -A
git commit -q -m "c1 base"
git checkout -q -b feature
mkdir reports
echo "<html>f</html>" > reports/f.html
echo "feature" > feature.py
git add -A
git commit -q -m "feature work with report"
git checkout -q main
echo "main change" > main_only.py
git add -A
git commit -q -m "main work"
git merge -q --no-ff -m "merge feature" feature
git push -q "$PUBLIC_REMOTE" main
fetch_public_refs
purge_public_history > "$WORK/merged.log" 2>&1 || fail "purge failed: $(cat "$WORK/merged.log")"

[[ -z "$(commits_containing "$PUB" reports)" ]] || fail "reports/ survived in merged history"
MERGE_SHA=$(git -C "$PUB" rev-list --merges main)
[[ -n "$MERGE_SHA" ]] || fail "merge commit was flattened away"
PARENT_COUNT=$(git -C "$PUB" log -1 --format='%P' "$MERGE_SHA" | wc -w | tr -d ' ')
[[ "$PARENT_COUNT" == "2" ]] || fail "merge commit has $PARENT_COUNT parent(s), expected 2"
git -C "$PUB" cat-file -e "main:feature.py" || fail "feature.py lost"
git -C "$PUB" cat-file -e "main:main_only.py" || fail "main_only.py lost"
pass "merge commits and both sides of the merge survive the purge"

# =============================================================================
# Scenario 6: nothing published yet
# =============================================================================
new_scenario empty
write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1"
fetch_public_refs
OUT=$(purge_public_history 2>&1) || fail "purge failed against an empty public repo"
echo "$OUT" | grep -q "no branches or tags yet" || fail "empty public repo not reported: $OUT"
pass "purge skips an empty public repo"

# =============================================================================
# Scenario 7: git-filter-repo is resolvable
# =============================================================================
FILTER_REPO_CMD=()
resolve_filter_repo || fail "git-filter-repo could not be resolved"
[[ ${#FILTER_REPO_CMD[@]} -gt 0 ]] || fail "FILTER_REPO_CMD not set"
"${FILTER_REPO_CMD[@]}" --version > /dev/null || fail "resolved filter-repo command does not run"
pass "resolve_filter_repo finds a working git-filter-repo: ${FILTER_REPO_CMD[*]}"

# =============================================================================
# Scenario 8: a broken exclude.json aborts the purge instead of publishing
# =============================================================================
new_scenario broken
write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1"
git push -q "$PUBLIC_REMOTE" main
fetch_public_refs
printf '{bad' > scripts/exclude.json
if purge_public_history > "$WORK/broken.log" 2>&1; then
    fail "purge did not fail on malformed exclude.json"
fi
pass "malformed exclude.json aborts the purge"

# =============================================================================
# Scenario 9: a failing history scan is an error, never a clean report
# =============================================================================
write_exclude reports
if OUT=$(excluded_paths_in_history "$(pwd)" refs/heads/does-not-exist 2>&1); then
    fail "scanning a missing ref reported success: $OUT"
fi
if OUT=$(excluded_paths_in_history "$WORK/does-not-exist" main 2>&1); then
    fail "scanning a missing repo reported success: $OUT"
fi
pass "history scan fails loudly instead of reporting a leaking history as clean"

# =============================================================================
# Scenario 10: several branches, including one with a slash in its name
# =============================================================================
new_scenario branches
PUB="$WORK/branches-public.git"

write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1 code"
git checkout -q -b release/2026
mkdir reports
echo "<html>r</html>" > reports/r.html
echo "released" > release.txt
git add -A
git commit -q -m "release branch work"
git checkout -q main
git push -q "$PUBLIC_REMOTE" main release/2026
fetch_public_refs
purge_public_history > "$WORK/branches.log" 2>&1 || fail "purge failed: $(cat "$WORK/branches.log")"

[[ -z "$(commits_containing "$PUB" reports)" ]] || fail "reports/ survived on a side branch"
git -C "$PUB" cat-file -e "refs/heads/release/2026:release.txt" ||
    fail "slash-named branch lost its content"
[[ "$(git -C "$PUB" rev-list --count refs/heads/release/2026)" == "2" ]] ||
    fail "slash-named branch history was mangled"
[[ "$(git -C "$PUB" rev-parse main)" == "$(git rev-parse "$PUBLIC_REMOTE/main")" ]] ||
    fail "main mirror out of sync after multi-branch purge"
pass "every public branch is purged, including branch names containing a slash"

# =============================================================================
# Scenario 11: a concurrent push to the public repo is refused, not clobbered
# =============================================================================
new_scenario raced
PUB="$WORK/raced-public.git"

write_exclude reports
echo "code" > app.py
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "c1 code and report"
git push -q "$PUBLIC_REMOTE" main
fetch_public_refs

# Somebody else pushes while this release is running.
git clone -q "$PUB" "$WORK/raced-other"
git -C "$WORK/raced-other" config user.email "other@test.com"
git -C "$WORK/raced-other" config user.name "Other"
echo "their work" > "$WORK/raced-other/theirs.txt"
git -C "$WORK/raced-other" add -A
git -C "$WORK/raced-other" commit -q -m "concurrent work"
git -C "$WORK/raced-other" push -q origin main
CONCURRENT_HEAD=$(git -C "$PUB" rev-parse main)

if purge_public_history > "$WORK/raced.log" 2>&1; then
    fail "purge overwrote a concurrent push instead of refusing"
fi
[[ "$(git -C "$PUB" rev-parse main)" == "$CONCURRENT_HEAD" ]] ||
    fail "concurrent commit was clobbered by the purge"
git -C "$PUB" cat-file -e "main:theirs.txt" || fail "concurrent work was lost"
pass "stale purge is refused by the push lease and the concurrent push survives"

# =============================================================================
# Scenario 12: a single rejected ref leaves the public repo untouched (--atomic)
# =============================================================================
new_scenario atomic
PUB="$WORK/atomic-public.git"

write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1 code"
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "c2 report"
git tag -a protected -m "protected"
echo "more" >> app.py
git add -A
git commit -q -m "c3 code"
git push -q "$PUBLIC_REMOTE" main
git push -q "$PUBLIC_REMOTE" protected
BEFORE_MAIN=$(git -C "$PUB" rev-parse main)
BEFORE_TAG=$(git -C "$PUB" rev-parse refs/tags/protected)

# Server-side hook that refuses to move the tag, like a protected-tag rule.
cat > "$PUB/hooks/update" <<'HOOK'
#!/bin/bash
[[ "$1" == "refs/tags/protected" ]] && exit 1
exit 0
HOOK
chmod +x "$PUB/hooks/update"

fetch_public_refs
if purge_public_history > "$WORK/atomic.log" 2>&1; then
    fail "purge reported success although a ref update was rejected"
fi
[[ "$(git -C "$PUB" rev-parse main)" == "$BEFORE_MAIN" ]] ||
    fail "main was rewritten even though the tag update was rejected"
[[ "$(git -C "$PUB" rev-parse refs/tags/protected)" == "$BEFORE_TAG" ]] ||
    fail "protected tag moved despite the hook"
rm -f "$PUB/hooks/update"
pass "a rejected ref update leaves the public repo completely unchanged (--atomic)"

# =============================================================================
# Scenario 13: refs that cannot be rewritten are reported, not ignored
# =============================================================================
new_scenario pullref
PUB="$WORK/pullref-public.git"

write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1 code"
git checkout -q -b pr
mkdir reports
echo "<html>pr</html>" > reports/pr.html
git add -A
git commit -q -m "pr work with report"
# GitHub publishes pull requests under read-only refs/pull/*; mimic one.
git push -q "$PUBLIC_REMOTE" HEAD:refs/pull/1/head
git checkout -q main
git push -q "$PUBLIC_REMOTE" main
fetch_public_refs

purge_public_history > "$WORK/pullref.log" 2>&1 || fail "purge failed: $(cat "$WORK/pullref.log")"
grep -q "cannot rewrite" "$WORK/pullref.log" ||
    fail "excluded content behind refs/pull/* was not reported: $(cat "$WORK/pullref.log")"
grep -q "refs/pull/1/head contains: reports" "$WORK/pullref.log" ||
    fail "the leaking ref and path were not named: $(cat "$WORK/pullref.log")"
[[ -z "$(git for-each-ref --format='%(refname)' "$PUBLIC_OTHER_NS")" ]] ||
    fail "scratch refs used to inspect refs/pull/* were left behind"
pass "excluded content behind unrewritable refs is detected and reported"

# =============================================================================
# Scenario 14: an unexpected public remote is never rewritten
# =============================================================================
new_scenario wrongremote
PUB="$WORK/wrongremote-public.git"

write_exclude reports
echo "code" > app.py
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "c1 code and report"
git push -q "$PUBLIC_REMOTE" main
BEFORE=$(git -C "$PUB" rev-parse main)
fetch_public_refs

PUBLIC_REPO_URL="https://github.com/ksenxx/kiss_ai.git"
PUBLIC_REPO_SSH="git@github.com:ksenxx/kiss_ai.git"
if purge_public_history > "$WORK/wrongremote.log" 2>&1; then
    fail "purge rewrote a remote that is not the public repo"
fi
grep -q "Refusing to rewrite an unexpected repository" "$WORK/wrongremote.log" ||
    fail "unexpected remote was not reported: $(cat "$WORK/wrongremote.log")"
[[ "$(git -C "$PUB" rev-parse main)" == "$BEFORE" ]] || fail "unexpected remote was rewritten"
pass "a remote that is not the configured public repo is refused"

# ssh and https spellings of the same repository must compare equal
SSH_FORM=$(normalize_repo_url "git@github.com:ksenxx/kiss_ai.git")
HTTPS_FORM=$(normalize_repo_url "https://github.com/ksenxx/kiss_ai")
OTHER_FORM=$(normalize_repo_url "git@github.com:someone/kiss_ai.git")
[[ "$SSH_FORM" == "$HTTPS_FORM" ]] || fail "ssh and https URLs of the same repo differ: $SSH_FORM vs $HTTPS_FORM"
[[ "$SSH_FORM" != "$OTHER_FORM" ]] || fail "different repositories normalize to the same value"
pass "public repo URLs normalize across ssh/https without matching other repos"

# =============================================================================
# Scenario 15: a branch literally named HEAD is purged like any other branch
# =============================================================================
new_scenario headbranch
PUB="$WORK/headbranch-public.git"

write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1 code"
git checkout -q -b odd
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "odd branch report"
git push -q "$PUBLIC_REMOTE" HEAD:refs/heads/HEAD
git checkout -q main
git push -q "$PUBLIC_REMOTE" main
[[ -n "$(commits_containing "$PUB" reports)" ]] || fail "setup: refs/heads/HEAD should publish reports/"

fetch_public_refs
purge_public_history > "$WORK/headbranch.log" 2>&1 || fail "purge failed: $(cat "$WORK/headbranch.log")"
[[ -z "$(commits_containing "$PUB" reports)" ]] ||
    fail "content on a branch named HEAD was not purged"
git -C "$PUB" rev-parse --verify --quiet refs/heads/HEAD > /dev/null ||
    fail "branch named HEAD disappeared"
pass "a branch literally named HEAD is purged instead of being skipped"

# =============================================================================
# Scenario 16: refusing to leave the public repo without a main branch
# =============================================================================
new_scenario emptymain
PUB="$WORK/emptymain-public.git"

write_exclude reports scripts
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "only excluded content"
git push -q "$PUBLIC_REMOTE" main
BEFORE=$(git -C "$PUB" rev-parse main)
fetch_public_refs

if purge_public_history > "$WORK/emptymain.log" 2>&1; then
    fail "purge silently emptied the public repo's main branch"
fi
grep -q "Nothing would be left to publish" "$WORK/emptymain.log" ||
    fail "emptied main was not explained: $(cat "$WORK/emptymain.log")"
[[ "$(git -C "$PUB" rev-parse main)" == "$BEFORE" ]] || fail "main was changed anyway"
pass "a rewrite that would delete main aborts with an explanation"

# =============================================================================
# Scenario 17: inspecting unrewritable refs leaves other local refs alone
# =============================================================================
new_scenario scratch
PUB="$WORK/scratch-public.git"

write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1 code"
KEEP=$(git rev-parse HEAD)
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "pr report"
git push -q "$PUBLIC_REMOTE" HEAD:refs/pull/7/head
git checkout -q -B main "$KEEP"
git push -q "$PUBLIC_REMOTE" main
# A ref that happens to live in the scratch parent namespace already.
git update-ref "${PUBLIC_OTHER_NS}/pull/7/head" "$KEEP"

fetch_public_refs
purge_public_history > "$WORK/scratch.log" 2>&1 || fail "purge failed: $(cat "$WORK/scratch.log")"
grep -q "refs/pull/7/head contains: reports" "$WORK/scratch.log" ||
    fail "unrewritable ref not reported on the already-clean path"
[[ "$(git rev-parse "${PUBLIC_OTHER_NS}/pull/7/head")" == "$KEEP" ]] ||
    fail "a pre-existing ref in the scratch namespace was clobbered or deleted"
[[ -z "$(git for-each-ref --format='%(refname)' "${PUBLIC_OTHER_NS}/$$")" ]] ||
    fail "this run's scratch refs were left behind"
pass "unrewritable-ref inspection uses its own namespace and cleans up after itself"

# Scratch refs left by an interrupted run are cleared even when the remote has no
# refs to inspect this time.
new_scenario scratch2
PUB="$WORK/scratch2-public.git"
write_exclude reports
echo "code" > app.py
git add -A
git commit -q -m "c1 code"
git push -q "$PUBLIC_REMOTE" main
git update-ref "${PUBLIC_OTHER_NS}/$$/pull/9/head" "$(git rev-parse HEAD)"
fetch_public_refs
purge_public_history > /dev/null 2>&1 || fail "purge failed on a clean repo with no extra refs"
[[ -z "$(git for-each-ref --format='%(refname)' "${PUBLIC_OTHER_NS}/$$")" ]] ||
    fail "stale scratch refs from an earlier run were not cleared"
pass "stale scratch refs are cleared even when there is nothing to inspect"

# =============================================================================
# Scenario 18: the post-push verification really rescans published history
# =============================================================================
new_scenario rescan
PUB="$WORK/rescan-public.git"

write_exclude reports
echo "code" > app.py
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "c1 code and report"
git push -q "$PUBLIC_REMOTE" main

if verify_public_history_clean > "$WORK/rescan.log" 2>&1; then
    fail "verification passed on a public repo that still has excluded paths"
fi
grep -q "excluded paths in its history" "$WORK/rescan.log" ||
    fail "verification did not explain the leak: $(cat "$WORK/rescan.log")"
purge_public_history > "$WORK/rescan2.log" 2>&1 || fail "purge failed: $(cat "$WORK/rescan2.log")"
verify_public_history_clean > /dev/null || fail "verification failed after a successful purge"
pass "verify_public_history_clean detects published leaks and passes once purged"

# =============================================================================
# Scenario 19: the release push itself is atomic and leased
# =============================================================================
new_scenario snapshot
PUB="$WORK/snapshot-public.git"

write_exclude reports
echo "code" > app.py
mkdir reports
echo "<html>x</html>" > reports/x.html
git add -A
git commit -q -m "c1 code and report"
git push -q "$PUBLIC_REMOTE" main
fetch_public_refs
purge_public_history > /dev/null 2>&1 || fail "setup purge failed"

PARENT=$(git rev-parse "${PUBLIC_BRANCH_NS}/main")
echo "release change" >> app.py
git add -A
git commit -q -m "release commit"
create_public_commit "$(git rev-parse HEAD)" "1.0.0" "$PARENT" > /dev/null
git tag -a v1.0.0 -m "Release 1.0.0" "$PUBLIC_COMMIT"
push_public_snapshot "$PUBLIC_COMMIT" v1.0.0 "$PARENT" > /dev/null ||
    fail "publishing a release snapshot failed"
[[ "$(git -C "$PUB" rev-parse main)" == "$PUBLIC_COMMIT" ]] || fail "main was not published"
[[ "$(git -C "$PUB" cat-file -t refs/tags/v1.0.0)" == "tag" ]] ||
    fail "release tag missing or not annotated"
[[ -z "$(commits_containing "$PUB" reports)" ]] || fail "release push published excluded paths"
pass "release snapshot and its tag are published in one atomic push"

# Re-using a published tag for different content must be refused, not moved.
BEFORE_TAG=$(git -C "$PUB" rev-parse refs/tags/v1.0.0)
echo "sneaky change" >> app.py
git add -A
git commit -q -m "sneaky commit"
create_public_commit "$(git rev-parse HEAD)" "1.0.0" "$(git -C "$PUB" rev-parse main)" > /dev/null
git tag -f -a v1.0.0 -m "Release 1.0.0 again" "$PUBLIC_COMMIT" > /dev/null
if push_public_snapshot "$PUBLIC_COMMIT" v1.0.0 "$(git -C "$PUB" rev-parse main)" \
    > "$WORK/snapshot2.log" 2>&1; then
    fail "an existing public tag was overwritten"
fi
[[ "$(git -C "$PUB" rev-parse refs/tags/v1.0.0)" == "$BEFORE_TAG" ]] ||
    fail "published tag was moved to different content"
pass "reusing a published release tag for different content is refused"

# A concurrent push to main must abort the release, tag included.
git clone -q "$PUB" "$WORK/snapshot-other"
git -C "$WORK/snapshot-other" config user.email "other@test.com"
git -C "$WORK/snapshot-other" config user.name "Other"
echo "their work" > "$WORK/snapshot-other/theirs.txt"
git -C "$WORK/snapshot-other" add -A
git -C "$WORK/snapshot-other" commit -q -m "concurrent work"
git -C "$WORK/snapshot-other" push -q origin main
CONCURRENT_HEAD=$(git -C "$PUB" rev-parse main)

echo "later change" >> app.py
git add -A
git commit -q -m "later release commit"
create_public_commit "$(git rev-parse HEAD)" "1.0.1" "$PARENT" > /dev/null
git tag -a v1.0.1 -m "Release 1.0.1" "$PUBLIC_COMMIT"
if push_public_snapshot "$PUBLIC_COMMIT" v1.0.1 "$PARENT" > "$WORK/snapshot3.log" 2>&1; then
    fail "release push overwrote a concurrent push"
fi
[[ "$(git -C "$PUB" rev-parse main)" == "$CONCURRENT_HEAD" ]] || fail "concurrent commit clobbered"
git -C "$PUB" rev-parse --verify --quiet refs/tags/v1.0.1 > /dev/null &&
    fail "tag was published although the branch update was rejected"
git rev-parse --verify --quiet refs/tags/v1.0.1 > /dev/null &&
    fail "the unpublished release tag was left behind locally"
pass "a concurrent push aborts the release push without publishing branch or tag"

echo
echo "ALL TESTS PASSED"
