#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

# Script to release to public GitHub repository and publish to PyPI.
# Repository: https://github.com/ksenxx/kiss_ai
# PyPI: https://pypi.org/project/kiss-agent-framework/
#
# Workflow:
# 1. Stash any uncommitted changes
# 2. Purge paths listed in scripts/exclude.json from every branch and tag of the
#    kiss_ai repo's existing history (drops commits whose only content was
#    excluded paths, re-points tags, pushes atomically under a lease per ref);
#    a no-op when that history is already clean
# 3. Check if origin is ahead of kiss_ai repo, whose main must consist of one
#    "Release X.Y.Z" commit per release and nothing else (no development
#    commits); a main of any other shape is replaced
# 4. If ahead, bump version in _version.py, README.md, SYSTEM.md
# 5. Commit changes with "Version bumped"
# 6. Push to origin
# 7. Push one "Release X.Y.Z" commit holding origin's tree minus the paths
#    listed in scripts/exclude.json, parented on the previous release commit,
#    as the kiss_ai repo's main, and tag it with the version; every earlier
#    release tag is re-pointed to its release commit when it is not one yet
# 8. Create GitHub release
# 9. Publish to PyPI
# 10. Run ./install.sh so this machine ends up running the version it just
#     released (non-interactive, without opening the webapp); the
#     nothing-to-release exit in step 3 runs it too before exiting
# 11. Restore stashed changes

set -e  # Exit on error

# Suppress the "A new release of gh is available" notice during release output.
# Update gh out-of-band when convenient; the release flow does not need it.
export GH_NO_UPDATE_NOTIFIER=1

# =============================================================================
# Constants
# =============================================================================
PUBLIC_REMOTE="public"
PUBLIC_REPO_URL="https://github.com/ksenxx/kiss_ai.git"
PUBLIC_REPO_SSH="git@github.com:ksenxx/kiss_ai.git"
# The version literal is single-sourced in src/kiss/core/_version.py.
VERSION_FILE="src/kiss/core/_version.py"
README_FILE="README.md"
SYSTEM_FILE="src/kiss/SYSTEM.md"
PYPI_PACKAGE_NAME="kiss-agent-framework"
# JSON list of literal file/folder paths (repo-relative, no globs) that MUST
# NOT be pushed to the public kiss_ai repo. Everything listed here is stripped
# from every commit of the history pushed to $PUBLIC_REPO_URL while remaining
# tracked in origin. The file is required; use [] to exclude nothing.
# Purging the public repo's existing history (see purge_public_history) matches
# these paths literally too and does not follow renames, so a file that was
# published under an earlier path must list that old path as well.
EXCLUDE_FILE="scripts/exclude.json"
# Local ref namespaces mirroring the public repo's branches and tags. They are
# fetched here instead of into refs/tags/* (and instead of relying on
# refs/remotes/$PUBLIC_REMOTE/*, whose symbolic HEAD entry would shadow a branch
# literally named HEAD) so the exact set of published refs is known and cannot be
# confused with branches or tags that exist only in origin.
PUBLIC_BRANCH_NS="refs/kiss-public-branches"
PUBLIC_TAG_NS="refs/kiss-public-tags"
# Parent of the scratch ref namespaces used to inspect public refs that are
# neither branches nor tags (GitHub's read-only refs/pull/*). Each run fetches
# into its own child namespace and deletes it again once scanned.
PUBLIC_OTHER_NS="refs/kiss-public-other"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

get_version() {
    if [[ ! -f "$VERSION_FILE" ]]; then
        print_error "Version file not found: $VERSION_FILE"
        exit 1
    fi
    VERSION=$(grep -oP '__version__\s*=\s*"\K[^"]+' "$VERSION_FILE" 2>/dev/null || \
              grep '__version__' "$VERSION_FILE" | sed 's/.*"\(.*\)".*/\1/')
    if [[ -z "$VERSION" ]]; then
        print_error "Could not extract version from $VERSION_FILE"
        exit 1
    fi
    echo "$VERSION"
}

bump_version() {
    local current_version="$1"
    local cur_year cur_month cur_minor
    IFS='.' read -r cur_year cur_month cur_minor <<< "$current_version"

    local now_year now_month
    now_year=$(date +%Y)
    now_month=$(date +%-m)  # no leading zero

    if [[ "$cur_year" == "$now_year" && "$cur_month" == "$now_month" ]]; then
        echo "${now_year}.${now_month}.$(( cur_minor + 1 ))"
    else
        echo "${now_year}.${now_month}.0"
    fi
}

update_version_file() {
    local new_version="$1"
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"${new_version}\"/" "$VERSION_FILE"
    rm -f "${VERSION_FILE}.bak"
    print_info "Updated $VERSION_FILE to version $new_version"
}

update_readme_version() {
    local version="$1"
    if [[ ! -f "$README_FILE" ]]; then
        print_warn "README file not found: $README_FILE - skipping"
        return
    fi
    local old_version
    old_version=$(grep -oP 'badge/version-\K[0-9][0-9.]*(?=-blue)' "$README_FILE" 2>/dev/null || \
                  grep 'badge/version-' "$README_FILE" | sed 's/.*badge\/version-\([0-9][0-9.]*\)-blue.*/\1/' | head -1)
    if [[ -n "$old_version" && "$old_version" != "$version" ]]; then
        sed -i.bak "s/${old_version}/${version}/g" "$README_FILE"
        rm -f "${README_FILE}.bak"
        print_info "Updated all occurrences of $old_version to $version in $README_FILE"
    elif [[ -z "$old_version" ]]; then
        print_warn "Version badge not found in $README_FILE - skipping"
    else
        print_info "README already at version $version"
    fi
}

update_system_md_version() {
    local version="$1"
    if [[ ! -f "$SYSTEM_FILE" ]]; then
        print_warn "SYSTEM file not found: $SYSTEM_FILE - skipping"
        return
    fi
    local old_version updated=false found=false

    # Replace "Version: X.Y.Z" (in the <identity> block: "... · Version: X.Y.Z").
    # Use a portable POSIX BRE so this works on macOS BSD grep and Linux GNU grep alike.
    old_version=$(grep 'Version: ' "$SYSTEM_FILE" | sed -n 's/.*Version: \([0-9][0-9.]*\).*/\1/p' | head -1)
    if [[ -n "$old_version" ]]; then
        found=true
        if [[ "$old_version" != "$version" ]]; then
            sed -i.bak "s/Version: ${old_version}/Version: ${version}/g" "$SYSTEM_FILE"
            rm -f "${SYSTEM_FILE}.bak"
            print_info "Updated 'Version:' from $old_version to $version in $SYSTEM_FILE"
            updated=true
        fi
    fi

    # Replace "Your version is X.Y.Z" (legacy phrasing, kept for safety).
    old_version=$(grep 'Your version is ' "$SYSTEM_FILE" | sed -n 's/.*Your version is \([0-9][0-9.]*\).*/\1/p' | head -1)
    if [[ -n "$old_version" ]]; then
        found=true
        if [[ "$old_version" != "$version" ]]; then
            sed -i.bak "s/Your version is ${old_version}/Your version is ${version}/g" "$SYSTEM_FILE"
            rm -f "${SYSTEM_FILE}.bak"
            print_info "Updated 'Your version is' from $old_version to $version in $SYSTEM_FILE"
            updated=true
        fi
    fi

    if [[ "$updated" == false ]]; then
        if [[ "$found" == false ]]; then
            print_warn "Version not found in $SYSTEM_FILE - skipping"
        else
            print_info "SYSTEM.md already at version $version"
        fi
    fi
}

ensure_remote() {
    if ! git remote get-url "$PUBLIC_REMOTE" &>/dev/null; then
        print_info "Adding remote '$PUBLIC_REMOTE'..."
        git remote add "$PUBLIC_REMOTE" "$PUBLIC_REPO_SSH"
    fi
}

# Mirror the public repo's branches into $PUBLIC_BRANCH_NS/* and its tags into
# $PUBLIC_TAG_NS/*, pruning refs that no longer exist there. The usual
# refs/remotes/$PUBLIC_REMOTE/* mirror is kept up to date as well so that
# "git log public/main" keeps working, but the purge reads the namespaces above.
fetch_public_refs() {
    git fetch --prune --no-tags --quiet "$PUBLIC_REMOTE" \
        "+refs/heads/*:${PUBLIC_BRANCH_NS}/*" \
        "+refs/tags/*:${PUBLIC_TAG_NS}/*" \
        "+refs/heads/*:refs/remotes/${PUBLIC_REMOTE}/*"
}

# Print "<branch> <object-id>" for every branch of the public repo, as mirrored
# by fetch_public_refs. The object id is what that branch pointed at when it was
# fetched, which is the lease the purge pushes against.
public_branches() {
    git for-each-ref --format='%(refname:lstrip=2) %(objectname)' "$PUBLIC_BRANCH_NS"
}

# Print "<tag> <object-id>" for every tag of the public repo. For annotated tags
# the object id is the tag object, which is what the remote advertises.
public_tags() {
    git for-each-ref --format='%(refname:lstrip=2) %(objectname)' "$PUBLIC_TAG_NS"
}

# Print the local ref of every public branch and tag mirrored by
# fetch_public_refs, one per line.
public_mirror_refs() {
    git for-each-ref --format='%(refname)' "$PUBLIC_BRANCH_NS" "$PUBLIC_TAG_NS"
}

# Print "<host>/<owner>/<repo>" for a git URL so that the ssh and https spellings
# of one repository compare equal. Local paths are returned unchanged apart from
# a trailing "/" or ".git".
normalize_repo_url() {
    printf '%s\n' "$1" | sed -e 's#^[a-zA-Z0-9+.-]*://##' -e 's#^[^/@]*@##' \
        -e 's#:#/#' -e 's#/*$##' -e 's#\.git$##'
}

# Print the push URL of $PUBLIC_REMOTE, but only after confirming it really is
# the public repo. purge_public_history force-updates and deletes refs there, so
# a stale, mistyped or repurposed remote must never be rewritten by accident.
public_push_url() {
    local url expected
    url=$(git remote get-url --push "$PUBLIC_REMOTE") || return 1
    for expected in "$PUBLIC_REPO_URL" "$PUBLIC_REPO_SSH"; do
        if [[ "$(normalize_repo_url "$url")" == "$(normalize_repo_url "$expected")" ]]; then
            printf '%s\n' "$url"
            return 0
        fi
    done
    print_error "Remote '$PUBLIC_REMOTE' pushes to $url, not $PUBLIC_REPO_URL" >&2
    print_error "Refusing to rewrite an unexpected repository" >&2
    return 1
}

# Print the paths listed in $EXCLUDE_FILE, one per line. The file must contain
# a JSON list of strings, e.g. ["secrets/", "notes.md"]. Each entry is treated
# as a LITERAL file or folder path relative to the repo root (no globs).
# The file is required so that its accidental absence cannot silently publish
# everything: use [] to exclude nothing. Fails on malformed JSON.
read_exclude_paths() {
    if [[ ! -f "$EXCLUDE_FILE" ]]; then
        echo "$EXCLUDE_FILE not found - create it (use [] to exclude nothing)" >&2
        return 1
    fi
    python3 - "$EXCLUDE_FILE" <<'PYEOF'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
    sys.exit(f"{sys.argv[1]} must contain a JSON list of path strings")
for path in data:
    if any(c in path for c in "\n\r\t\x00"):
        sys.exit(f"{sys.argv[1]}: path {path!r} must not contain newlines, tabs or NUL")
    if path.startswith("/") or any(part == ".." for part in path.split("/")):
        sys.exit(f"{sys.argv[1]}: path {path!r} must be repo-relative without '..'")
    path = path.rstrip("/")
    if path:
        print(path)
PYEOF
}

# Print the tree sha of <commit> with all paths listed in $EXCLUDE_FILE
# removed. Uses a temporary index so neither the real index nor the working
# tree is touched. Prints the commit's own tree when nothing is excluded.
filtered_tree() {
    local commit="$1"
    local paths tmp_index path
    if ! paths=$(read_exclude_paths); then
        print_error "Failed to read $EXCLUDE_FILE - aborting" >&2
        return 1
    fi
    tmp_index=$(mktemp)
    if ! GIT_INDEX_FILE="$tmp_index" git read-tree "$commit"; then
        rm -f "$tmp_index"
        return 1
    fi
    if [[ -n "$paths" ]]; then
        # :(literal) treats each path verbatim - no glob/wildcard expansion -
        # so an entry like "foo[1]" cannot accidentally remove "foo1". One
        # git rm for all paths: this runs once per public release tag.
        local pathspecs=()
        while IFS= read -r path; do
            pathspecs+=(":(literal)$path")
        done <<< "$paths"
        if ! GIT_INDEX_FILE="$tmp_index" \
            git rm -r -f -q --cached --ignore-unmatch -- "${pathspecs[@]}" >/dev/null; then
            print_error "Failed to exclude the paths of $EXCLUDE_FILE from public tree" >&2
            rm -f "$tmp_index"
            return 1
        fi
    fi
    GIT_INDEX_FILE="$tmp_index" git write-tree
    local status=$?
    rm -f "$tmp_index"
    return $status
}

# Print "<tag> <object-id>" for every public tag named v<version>, oldest
# version first: the release tags whose snapshots make up the release-only
# history. Tags with any other name are not releases and are left alone.
public_release_tags() {
    public_tags | grep '^v[0-9][0-9.]* ' | sort -V
}

# Print the parent commit ids of <commit>, space-separated ("" for a root).
commit_parents() {
    git rev-list --parents -n 1 "$1" | cut -s -d' ' -f2-
}

# Print the id of a "Release <version>" commit holding <tree> on top of
# <parent> ("" for the first release). Author, committer and dates are copied
# from <source-commit> (the released commit, or the previously published commit
# a release-only history is rebuilt from), so the same inputs always yield the
# same commit id and build_release_chain can recognise its own work.
release_commit() {
    local version="$1" tree="$2" parent="$3" source_commit="$4"
    local ident=() line parent_args=()
    while IFS= read -r line; do
        ident+=("$line")
    done < <(git log -1 --date=raw "$source_commit" --format='GIT_AUTHOR_NAME=%an
GIT_AUTHOR_EMAIL=%ae
GIT_AUTHOR_DATE=%ad
GIT_COMMITTER_NAME=%cn
GIT_COMMITTER_EMAIL=%ce
GIT_COMMITTER_DATE=%cd')
    if [[ ${#ident[@]} -ne 6 ]]; then
        print_error "Cannot read the author and committer of $source_commit" >&2
        return 1
    fi
    if [[ -n "$parent" ]]; then
        parent_args=(-p "$parent")
    fi
    env "${ident[@]}" git commit-tree "$tree" "${parent_args[@]}" -m "Release $version"
}

# Print the id of an annotated tag object <tag> ("Release <version>") pointing
# at release commit <commit>, tagged by that commit's committer. Only the
# object is created - no local ref - so pushing it is the only side effect.
release_tag() {
    local tag="$1" version="$2" commit="$3" tagger
    tagger=$(git cat-file commit "$commit" |
        awk '/^$/ { exit } /^committer / { sub(/^committer /, ""); print; exit }')
    printf 'object %s\ntype commit\ntag %s\ntagger %s\n\nRelease %s\n' \
        "$commit" "$tag" "$tagger" "$version" | git mktag
}

# Succeed when <commit> is the "Release <version>" commit holding <tree> on top
# of <parent> ("" for a root): the shape every commit of public main must have.
release_commit_shape_ok() {
    local commit="$1" tree="$2" parent="$3" version="$4"
    [[ "$(git rev-parse "${commit}^{tree}")" == "$tree" &&
        "$(commit_parents "$commit")" == "$parent" &&
        "$(git log -1 --format=%s "$commit")" == "Release $version" ]]
}

# Build the history that the public repo's main must consist of: one
# "Release <version>" commit per public release tag, oldest first, each holding
# the tagged tree minus the excluded paths and parented on the previous release
# commit - nothing else is ever reachable from public main. A tagged commit
# that already has exactly that shape is reused as is, so a public repo that is
# already in this form rebuilds to its own commits, and because release_commit
# is deterministic so does any rebuilt chain. Sets:
#   RELEASE_CHAIN_TIP  the newest release commit ("" when there is no release tag)
#   RETAG_LEASES       --force-with-lease arguments, one per tag whose commit
#                      was replaced, pinned to the tag's current public object
#   RETAG_REFSPECS     the matching "<tag object>:refs/tags/<tag>" refspecs
build_release_chain() {
    local tag oid version commit tree parent=""
    RELEASE_CHAIN_TIP=""
    RETAG_LEASES=()
    RETAG_REFSPECS=()
    while read -r tag oid; do
        version="${tag#v}"
        if ! commit=$(git rev-parse --verify --quiet "${oid}^{commit}"); then
            print_error "Public tag $tag does not point at a commit" >&2
            return 1
        fi
        tree=$(filtered_tree "$commit") || return 1
        if release_commit_shape_ok "$commit" "$tree" "$parent" "$version"; then
            parent="$commit"
            continue
        fi
        parent=$(release_commit "$version" "$tree" "$parent" "$commit") || return 1
        RETAG_LEASES+=("--force-with-lease=refs/tags/${tag}:${oid}")
        RETAG_REFSPECS+=("$(release_tag "$tag" "$version" "$parent"):refs/tags/${tag}")
    done < <(public_release_tags)
    RELEASE_CHAIN_TIP="$parent"
    if [[ ${#RETAG_REFSPECS[@]} -gt 0 ]]; then
        print_info "${#RETAG_REFSPECS[@]} public release tag(s) do not point at release-only commits yet and will be re-pointed"
    fi
}

# Decide whether origin's HEAD, whose tree minus the excluded paths is
# <filtered-tree>, must be published to the public repo, whose main tip is
# <public-head>. Nothing is to release exactly when public main already IS the
# release-only history (build_release_chain's tip), every release tag points
# at its commit of that history (a tag moved back onto a development commit
# rebuilds to the very same release commit, so the tip alone cannot tell), and
# that tip holds <filtered-tree>. Comparing trees, not commits, means a change
# plus its revert publishes nothing; a main of any other shape (development
# history, a squashed snapshot outside the chain) is replaced. A public tip
# that cannot be inspected counts as needing a release rather than being
# skipped silently. Returns 1 when there is nothing to release.
release_needed() {
    local filtered_tree="$1" public_head="$2" public_commit
    if ! public_commit=$(git rev-parse --verify --quiet "${public_head}^{commit}"); then
        print_info "kiss_ai main cannot be inspected - proceeding with release"
        return 0
    fi
    if [[ "$public_commit" != "$RELEASE_CHAIN_TIP" ]]; then
        print_info "kiss_ai main is not the release-only history - proceeding with release"
        return 0
    fi
    if [[ ${#RETAG_REFSPECS[@]} -gt 0 ]]; then
        print_info "${#RETAG_REFSPECS[@]} kiss_ai release tag(s) do not point at release-only commits - proceeding with release"
        return 0
    fi
    if [[ "$(git rev-parse "${public_commit}^{tree}")" != "$filtered_tree" ]]; then
        print_info "Origin differs from kiss_ai - proceeding with release"
        return 0
    fi
    return 1
}

# Create the commit published as the public repo's new main: a
# "Release <version>" commit holding <source-commit>'s tree minus the excluded
# paths, parented on <parent> (the previous release commit, "" for the first
# release) - never on origin's own commits, so only release commits are ever
# reachable from public history and excluded content never is. Sets
# PUBLIC_COMMIT.
create_public_commit() {
    local source_commit="$1" version="$2" parent="$3" tree
    tree=$(filtered_tree "$source_commit") || return 1
    PUBLIC_COMMIT=$(release_commit "$version" "$tree" "$parent" "$source_commit") || return 1
    print_info "Created public release commit $PUBLIC_COMMIT (snapshot of $source_commit)"
}

# Fail if any excluded path is still present in <commit>'s tree. Matches each
# excluded entry literally, as an exact file path or a folder prefix.
verify_no_excluded_paths() {
    local commit="$1"
    local paths path file leaked
    if ! paths=$(read_exclude_paths); then
        print_error "Failed to read $EXCLUDE_FILE - aborting"
        return 1
    fi
    if [[ -z "$paths" ]]; then
        return 0
    fi
    while IFS= read -r path; do
        leaked=""
        # -z: NUL-delimited raw filenames, so unusual names (quotes, non-ASCII)
        # are matched verbatim rather than in git's quoted form.
        while IFS= read -r -d '' file; do
            if [[ "$file" == "$path" || "$file" == "$path"/* ]]; then
                leaked+="$file"$'\n'
            fi
        done < <(git ls-tree -r --name-only -z "$commit")
        if [[ -n "$leaked" ]]; then
            print_error "Excluded path '$path' leaked into public commit $commit:"
            printf '%s' "$leaked"
            return 1
        fi
    done <<< "$paths"
    print_info "Verified: no excluded paths present in public commit"
}

# Fail if any commit reachable from <commit> contains an excluded path. The new
# release commit sits on the chain of all earlier release commits, so checking
# the tip's tree alone (verify_no_excluded_paths) is not enough: a path must
# not appear in any earlier release either.
verify_no_excluded_paths_in_history() {
    local commit="$1" leaks
    if ! leaks=$(excluded_paths_in_history "$(pwd)" "$commit"); then
        return 1
    fi
    if [[ -n "$leaks" ]]; then
        print_error "Excluded paths reachable from public commit $commit - not pushing:"
        printf '%s\n' "$leaks" | head -20
        return 1
    fi
    print_info "Verified: no excluded paths in any commit reachable from the public tip"
}

# Print "<commit>:<path>" for every commit that is reachable from the given
# rev-list arguments (run inside <repo>) and whose tree still contains a path
# listed in $EXCLUDE_FILE. Prints nothing when that history is already clean.
# One batched cat-file pass over the commit x excluded-path cross product keeps
# this fast enough to run on every release: an object that resolves means the
# path is still in that commit's tree, "missing" means it is not.
excluded_paths_in_history() {
    local repo="$1"
    shift
    local paths commits queries checked
    if ! paths=$(read_exclude_paths); then
        print_error "Failed to read $EXCLUDE_FILE - aborting" >&2
        return 1
    fi
    if [[ -z "$paths" ]]; then
        return 0
    fi
    # Every git step is checked separately: piping straight into the final grep
    # would hide a failing rev-list or cat-file behind grep's exit status and
    # report a leaking history as clean.
    if ! commits=$(git -C "$repo" rev-list "$@"); then
        print_error "Failed to list commits of $* in $repo" >&2
        return 1
    fi
    if [[ -z "$commits" ]]; then
        return 0
    fi
    queries=$(printf '%s\n' "$commits" \
        | KISS_EXCLUDE_PATHS="$paths" awk '
            BEGIN { n = split(ENVIRON["KISS_EXCLUDE_PATHS"], excluded, "\n") }
            { for (i = 1; i <= n; i++) print $0 ":" excluded[i] }')
    if ! checked=$(printf '%s\n' "$queries" | git -C "$repo" cat-file --batch-check); then
        print_error "Failed to inspect the trees of $repo" >&2
        return 1
    fi
    # cat-file answers one line per query in order but echoes the query only for
    # objects it could not find, so pair the two lists back up: a verdict of
    # "<query> missing" means the path is absent from that commit's tree, any
    # other verdict ("<oid> <type> <size>") means the path is still there.
    paste -d'\t' <(printf '%s\n' "$queries") <(printf '%s\n' "$checked") \
        | { grep -v ' missing$' || true; } | cut -f1
}

# Set FILTER_REPO_CMD to a command that runs git-filter-repo, the tool the git
# project recommends for history rewrites (git-filter-branch is deprecated and
# ~20x slower here). Prefers an installed copy and otherwise runs the PyPI
# package through uv, which the release already depends on for build/publish.
resolve_filter_repo() {
    if git filter-repo --version >/dev/null 2>&1; then
        FILTER_REPO_CMD=(git filter-repo)
    elif python3 -m git_filter_repo --version >/dev/null 2>&1; then
        FILTER_REPO_CMD=(python3 -m git_filter_repo)
    elif command -v uvx >/dev/null 2>&1 &&
        uvx --from git-filter-repo git-filter-repo --version >/dev/null 2>&1; then
        FILTER_REPO_CMD=(uvx --from git-filter-repo git-filter-repo)
    else
        return 1
    fi
}

# Split the refs staged in <purge-repo> into force-push refspecs for the ones
# the rewrite kept (UPDATE_REFSPECS) and delete refspecs for the ones it removed
# (DELETE_REFSPECS), naming the latter in PURGE_GONE_REFS. A ref the rewrite
# dropped must be deleted on the public repo: left alone it would still point at
# an unpurged commit.
classify_purged_refs() {
    local purge_repo="$1"
    shift
    UPDATE_REFSPECS=()
    DELETE_REFSPECS=()
    PURGE_GONE_REFS=""
    local ref
    for ref in "$@"; do
        if git -C "$purge_repo" rev-parse --verify --quiet "$ref" >/dev/null; then
            UPDATE_REFSPECS+=("${ref}:${ref}")
        else
            DELETE_REFSPECS+=(":${ref}")
            PURGE_GONE_REFS+="${ref}"$'\n'
        fi
    done
}

# Rewrite the public repo's history so that no commit reachable from any of its
# branches or tags contains a path listed in $EXCLUDE_FILE: the paths are
# stripped from every such commit, commits whose only content was excluded paths
# disappear entirely, surviving commits keep their non-excluded changes, and tags
# are re-pointed at the rewritten commits. The result is pushed under a lease per
# ref and atomically, so a concurrent push to kiss_ai aborts the purge instead of
# being overwritten. Refs that GitHub does not let anyone rewrite (refs/pull/*)
# are reported by warn_unrewritable_public_refs rather than silently ignored.
# Cheap no-op when the public history is already clean, so a release pays for
# the rewrite only after $EXCLUDE_FILE gains a path that was published earlier.
purge_public_history() {
    local paths leaks commit_count public_url purge_dir purge_repo
    if ! paths=$(read_exclude_paths); then
        print_error "Failed to read $EXCLUDE_FILE - aborting"
        return 1
    fi
    if [[ -z "$paths" ]]; then
        return 0
    fi

    if ! public_url=$(public_push_url); then
        return 1
    fi

    # source_refs: where the published refs are mirrored locally.
    # staged_refs: the same refs under their real names in the throwaway repo.
    # leases: what each public ref pointed at when it was fetched, so the final
    # force-push refuses to clobber anything pushed to kiss_ai meanwhile.
    local source_refs=() staged_refs=() stage_refspecs=() leases=() name oid
    while IFS=' ' read -r name oid; do
        [[ -n "$name" ]] || continue
        source_refs+=("${PUBLIC_BRANCH_NS}/${name}")
        staged_refs+=("refs/heads/${name}")
        stage_refspecs+=("+${PUBLIC_BRANCH_NS}/${name}:refs/heads/${name}")
        leases+=("--force-with-lease=refs/heads/${name}:${oid}")
    done < <(public_branches)
    while IFS=' ' read -r name oid; do
        [[ -n "$name" ]] || continue
        source_refs+=("${PUBLIC_TAG_NS}/${name}")
        staged_refs+=("refs/tags/${name}")
        stage_refspecs+=("+${PUBLIC_TAG_NS}/${name}:refs/tags/${name}")
        leases+=("--force-with-lease=refs/tags/${name}:${oid}")
    done < <(public_tags)
    if [[ ${#source_refs[@]} -eq 0 ]]; then
        print_info "Public repo has no branches or tags yet - no history to purge"
        return 0
    fi

    if ! leaks=$(excluded_paths_in_history "$(pwd)" "${source_refs[@]}"); then
        return 1
    fi
    if [[ -z "$leaks" ]]; then
        print_info "Verified: no excluded paths in any kiss_ai branch or tag"
        warn_unrewritable_public_refs "$public_url"
        return 0
    fi
    commit_count=$(printf '%s\n' "$leaks" | cut -d: -f1 | sort -u | wc -l | tr -d ' ')
    print_step "Purging $commit_count commit(s) carrying excluded paths from kiss_ai history..."

    if ! resolve_filter_repo; then
        print_error "git-filter-repo is required to purge kiss_ai history but was not found"
        print_info "Install it with: uv tool install git-filter-repo (or brew install git-filter-repo)"
        return 1
    fi

    purge_dir=$(mktemp -d)
    purge_repo="$purge_dir/kiss_ai.git"
    # Rewrite a throwaway bare repo that holds exactly the public refs: origin
    # keeps the excluded paths, and git-filter-repo (which deletes remotes and
    # expires reflogs of the repo it rewrites) never touches this checkout.
    if ! git init --quiet --bare "$purge_repo" ||
        ! git push --quiet "$purge_repo" "${stage_refspecs[@]}"; then
        print_error "Failed to stage kiss_ai history for purging"
        rm -rf "$purge_dir"
        return 1
    fi

    # --invert-paths keeps everything except the listed paths; --prune-empty auto
    # drops commits left with no changes (a commit that only committed excluded
    # paths), re-parenting its children on its nearest surviving ancestor.
    local filter_args=(--force --invert-paths --prune-empty auto)
    while IFS= read -r name; do
        filter_args+=(--path "$name")
    done <<< "$paths"
    if ! (cd "$purge_repo" && "${FILTER_REPO_CMD[@]}" "${filter_args[@]}"); then
        print_error "git-filter-repo failed to purge kiss_ai history"
        rm -rf "$purge_dir"
        return 1
    fi

    if ! leaks=$(excluded_paths_in_history "$purge_repo" --all); then
        rm -rf "$purge_dir"
        return 1
    fi
    if [[ -n "$leaks" ]]; then
        print_error "Purge left excluded paths in the rewritten history - not pushing:"
        printf '%s\n' "$leaks" | head -20
        rm -rf "$purge_dir"
        return 1
    fi

    classify_purged_refs "$purge_repo" "${staged_refs[@]}"
    if [[ -n "$PURGE_GONE_REFS" ]]; then
        # Deleting main is refused by every server (it is the default branch), so
        # say why instead of pushing something that cannot succeed.
        if printf '%s' "$PURGE_GONE_REFS" | grep -qx "refs/heads/main"; then
            print_error "Every commit of kiss_ai's main branch consists of excluded paths only"
            print_error "Nothing would be left to publish - shorten $EXCLUDE_FILE"
            rm -rf "$purge_dir"
            return 1
        fi
        print_warn "The rewrite removed these public refs entirely; deleting them:"
        printf '%s' "$PURGE_GONE_REFS"
    fi

    # One push updates the rewritten branches/tags and drops the vanished ones:
    # --atomic so a single rejected ref cannot leave kiss_ai half rewritten, and
    # a lease per ref (instead of a blanket --force) so anything pushed to
    # kiss_ai since fetch_public_refs is refused rather than silently discarded.
    # --mirror is deliberately avoided: it fails on GitHub's read-only
    # refs/pull/* refs and would delete refs this script never staged.
    if ! git -C "$purge_repo" push --atomic --quiet "$public_url" \
        "${leases[@]}" "${UPDATE_REFSPECS[@]}" "${DELETE_REFSPECS[@]}"; then
        print_error "Failed to push purged history to $public_url - nothing was changed"
        print_info "A rejected lease means kiss_ai moved since this release started; rerun the release"
        rm -rf "$purge_dir"
        return 1
    fi
    rm -rf "$purge_dir"

    # Re-mirror so callers see the rewritten tip instead of the purged one, and
    # re-scan: a branch or tag created on kiss_ai while the rewrite ran is not
    # covered by the leases above and could still carry excluded paths.
    verify_public_history_clean || return 1

    print_info "Purged excluded paths from kiss_ai history (${#UPDATE_REFSPECS[@]} ref(s) rewritten)"
    print_warn "History was rewritten: the purged commits remain reachable by sha on GitHub"
    print_warn "until it garbage-collects, and source archives of re-pointed tags now differ"
    warn_unrewritable_public_refs "$public_url"
}

# Delete every ref under <namespace>. Used to clear scratch refs both before and
# after they are needed, so a failed fetch cannot leave any behind.
delete_refs_under() {
    local namespace="$1" ref
    while IFS= read -r ref; do
        [[ -n "$ref" ]] || continue
        git update-ref -d "$ref"
    done < <(git for-each-ref --format='%(refname)' "$namespace")
}

# Re-mirror the public repo and fail if any of its branches or tags still reaches
# a path listed in $EXCLUDE_FILE. Run after every push to the public repo: a ref
# created there concurrently cannot be covered by a push lease, so this is the
# only way to know that what is published is actually clean.
verify_public_history_clean() {
    local refs=() ref leaks
    fetch_public_refs
    while IFS= read -r ref; do
        [[ -n "$ref" ]] || continue
        refs+=("$ref")
    done < <(public_mirror_refs)
    if [[ ${#refs[@]} -eq 0 ]]; then
        return 0
    fi
    if ! leaks=$(excluded_paths_in_history "$(pwd)" "${refs[@]}"); then
        return 1
    fi
    if [[ -n "$leaks" ]]; then
        print_error "kiss_ai has excluded paths in its history (refs changed meanwhile?):"
        printf '%s\n' "$leaks" | head -20
        return 1
    fi
    print_info "Verified: no excluded paths in any kiss_ai branch or tag"
}

# Publish release <commit> (with the chain of earlier release commits below it)
# as the public repo's main branch and tag object <tag-object> as its tag
# <tag> in one atomic push, leased on main still being at <expected-main> (""
# when the public repo has no main yet) and on <tag> not existing yet. Any
# further arguments are build_release_chain's RETAG_LEASES followed by its
# RETAG_REFSPECS, re-pointing earlier release tags in the same push. The
# leases double as the force that replacing a main of another shape (a
# non-fast-forward update) needs. Anything pushed to the public repo since
# fetch_public_refs therefore aborts the release instead of being silently
# overwritten, and a rejected ref leaves the repo untouched.
push_public_snapshot() {
    local commit="$1" tag="$2" tag_object="$3" expected_main="$4" public_url
    shift 4
    if ! public_url=$(public_push_url); then
        return 1
    fi
    # An empty <expect> in a lease means "this ref must not exist yet".
    if ! git push --atomic "$public_url" \
        "--force-with-lease=refs/heads/main:${expected_main}" \
        "--force-with-lease=refs/tags/${tag}:" \
        "$@" "${commit}:refs/heads/main" "${tag_object}:refs/tags/${tag}"; then
        print_error "Failed to publish $tag to $public_url - nothing was changed"
        print_info "A rejected lease means kiss_ai moved since this release started; rerun the release"
        return 1
    fi
    verify_public_history_clean
}

# Warn when the public repo advertises refs that this script cannot rewrite and
# that still contain excluded paths. GitHub keeps refs/pull/* read-only, so a
# pull request opened against a commit with excluded content keeps that content
# fetchable (and unreachable objects alive) no matter how often branches and tags
# are purged; only closing those pull requests and asking GitHub Support to run a
# server-side garbage collection removes them.
warn_unrewritable_public_refs() {
    local public_url="$1"
    # A namespace of its own per run: it may be deleted wholesale afterwards
    # without touching refs that existed before, even if the fetch below only
    # partially succeeds.
    local scan_ns="${PUBLIC_OTHER_NS}/$$"
    local sha ref fetch_refspecs=() scan_refs=() leaks
    # Drop anything an interrupted earlier run left in this namespace first, so
    # nothing stale survives even when there is nothing to scan this time.
    delete_refs_under "$scan_ns"
    while IFS=$'\t' read -r sha ref; do
        case "$ref" in
            refs/heads/* | refs/tags/* | HEAD | '') continue ;;
        esac
        fetch_refspecs+=("+${ref}:${scan_ns}/${ref#refs/}")
        scan_refs+=("${scan_ns}/${ref#refs/}")
    done < <(git ls-remote "$public_url" 2>/dev/null)
    if [[ ${#scan_refs[@]} -eq 0 ]]; then
        return 0
    fi
    if ! git fetch --quiet --no-tags "$public_url" "${fetch_refspecs[@]}"; then
        print_warn "Could not inspect ${#scan_refs[@]} non-branch ref(s) of kiss_ai for excluded paths"
        delete_refs_under "$scan_ns"
        return 0
    fi
    # Scanned one ref at a time so the warning can name the offending ref.
    local leaking=""
    for ref in "${scan_refs[@]}"; do
        if ! leaks=$(excluded_paths_in_history "$(pwd)" "$ref"); then
            print_warn "Could not scan ${ref} of kiss_ai for excluded paths"
        elif [[ -n "$leaks" ]]; then
            leaking+="  refs/${ref#"${scan_ns}"/} contains: "
            leaking+="$(printf '%s\n' "$leaks" | cut -d: -f2- | sort -u | tr '\n' ' ')"$'\n'
        fi
    done
    delete_refs_under "$scan_ns"
    if [[ -z "$leaking" ]]; then
        return 0
    fi
    print_warn "Excluded paths stay reachable through refs this script cannot rewrite:"
    printf '%s' "$leaking"
    print_warn "GitHub keeps pull-request refs read-only: close and delete the pull requests"
    print_warn "that reference them, then ask GitHub Support to garbage-collect the repo."
}

# Run ./install.sh on this machine, non-interactively and without launching an
# webapp, so the machine that ran the release ends up running this checkout.
# Called from every successful exit of main() — the full release and the
# nothing-to-release path — while the pre-release stash is still held, so the
# install always builds from the clean tree. A failure propagates (set -e) and
# the EXIT trap restores the stash.
run_local_install() {
    print_step "Running ./install.sh to install this checkout locally..."
    KISS_SKIP_LAUNCH=1 bash ./install.sh --non-interactive
    print_info "Local install completed"
}

# PyPI rejects any file over 100 MiB (https://docs.pypi.org/project-management/storage-limits/).
# uv publish uploads the wheel before the sdist and a rejected sdist leaves a
# wheel-only release behind that can never be completed (PyPI refuses to
# re-upload a version), so the size of every file is checked before the first
# upload starts. $1...: the built distribution files.
PYPI_FILE_SIZE_LIMIT=$((100 * 1024 * 1024))
check_pypi_file_sizes() {
    local file size
    for file in "$@"; do
        size=$(wc -c < "$file")
        if (( size > PYPI_FILE_SIZE_LIMIT )); then
            print_error "$file is $(( size / 1024 / 1024 )) MiB; PyPI rejects files over $(( PYPI_FILE_SIZE_LIMIT / 1024 / 1024 )) MiB"
            print_info "Nothing was uploaded. Trim [tool.hatch.build.targets.sdist] in pyproject.toml and rerun"
            return 1
        fi
    done
}

publish_to_pypi() {
    local version="$1"
    
    print_step "Building package for PyPI..."
    rm -rf dist/*.tar.gz dist/*.whl
    uv build
    
    if [[ -z "$(ls dist/*.tar.gz dist/*.whl 2>/dev/null)" ]]; then
        print_error "Build failed - no .tar.gz or .whl files in dist/"
        return 1
    fi
    
    print_info "Built packages:"
    ls -la dist/*.tar.gz dist/*.whl
    check_pypi_file_sizes dist/*.tar.gz dist/*.whl || return 1
    
    print_step "Uploading to PyPI..."
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
        print_error "UV_PUBLISH_TOKEN environment variable is not set"
        print_info "Please set it with: export UV_PUBLISH_TOKEN='pypi-your-token-here'"
        return 1
    fi
    
    uv publish
    
    print_info "Successfully published version $version to PyPI"
    print_info "View at: https://pypi.org/project/${PYPI_PACKAGE_NAME}/${version}/"
}

# =============================================================================
# Main Release Process
# =============================================================================
main() {
    print_step "Starting release process"
    echo "Public repo: $PUBLIC_REPO_URL"
    echo

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Get current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_info "Current branch: $CURRENT_BRANCH"

    # Ensure public remote exists
    ensure_remote

    # The exclude list must exist, parse, and be committed: uncommitted edits
    # to it would be stashed away below and the release would silently run
    # with stale exclusion rules.
    if [[ ! -f "$EXCLUDE_FILE" ]]; then
        print_error "$EXCLUDE_FILE not found - create it (use [] to exclude nothing)"
        exit 1
    fi
    if [[ -n "$(git status --porcelain -- "$EXCLUDE_FILE")" ]]; then
        print_error "$EXCLUDE_FILE has uncommitted changes - commit them before releasing"
        exit 1
    fi
    read_exclude_paths > /dev/null

    # Step 1: Stash uncommitted changes and sync with origin
    print_step "Syncing with origin and checking kiss_ai repo..."
    STASHED=false
    if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        print_info "Stashing uncommitted changes..."
        git stash push --include-untracked -m "release-script: pre-release stash"
        STASHED=true
    fi
    trap 'if [[ "$STASHED" == true ]]; then print_warn "Restoring stashed changes after failure..."; git stash pop; fi' EXIT
    git fetch origin
    fetch_public_refs
    git pull --rebase origin "$CURRENT_BRANCH"

    # Step 2: Strip excluded paths from the public repo's existing history before
    # doing anything else: filtering new snapshots is not enough once
    # $EXCLUDE_FILE gains a path that an earlier release already published, and
    # this must run even when there is nothing new to release (step 3 may exit).
    purge_public_history

    # Step 3: Check whether origin is ahead of the kiss_ai repo
    ORIGIN_HEAD=$(git rev-parse HEAD)
    PUBLIC_HEAD=$(git rev-parse --verify --quiet "${PUBLIC_BRANCH_NS}/main" || echo "")

    # Filter origin's tree and rebuild the release-only history up front: a
    # broken exclude.json aborts the release here, before any side effects
    # (set -e catches the failure), and release_needed can check that public
    # main IS that history (one "Release X.Y.Z" commit per release, nothing
    # else) rather than development commits or a snapshot outside the chain.
    FILTERED_ORIGIN_TREE=$(filtered_tree "$ORIGIN_HEAD")
    build_release_chain

    if [[ -z "$PUBLIC_HEAD" ]]; then
        print_info "Public repo has no main branch yet - will create it"
    elif ! release_needed "$FILTERED_ORIGIN_TREE" "$PUBLIC_HEAD"; then
        print_info "kiss_ai already matches origin (minus excluded paths) - nothing to release"
        # Even with nothing to publish, finish by installing this checkout
        # locally (see run_local_install); the EXIT trap restores the stash.
        run_local_install
        exit 0
    fi

    # Step 4: Bump version in _version.py and README.md
    CURRENT_VERSION=$(get_version)
    VERSION=$(bump_version "$CURRENT_VERSION")
    TAG_NAME="v$VERSION"
    
    print_info "Current version: $CURRENT_VERSION"
    print_info "New version: $VERSION (tag: $TAG_NAME)"
    
    print_step "Bumping version..."
    update_version_file "$VERSION"
    update_readme_version "$VERSION"
    update_system_md_version "$VERSION"

    # Step 5: Commit the version bump
    print_step "Committing version bump..."
    git add -A
    git commit -m "Version bumped to $VERSION"
    print_info "Committed version bump"

    # Step 6: Pull latest from origin (rebase), then push (with retry)
    print_step "Syncing with origin..."
    for attempt in 1 2 3; do
        git pull --rebase origin "$CURRENT_BRANCH"
        if git push origin "$CURRENT_BRANCH"; then
            break
        fi
        if [[ $attempt -eq 3 ]]; then
            print_error "Failed to push to origin after 3 attempts"
            exit 1
        fi
        print_warn "Push to origin failed (attempt $attempt/3), retrying in 2s..."
        sleep 2
    done
    print_info "Pushed to origin"

    # Step 7: Push one "Release $VERSION" commit to the kiss_ai repo: origin's
    # tree minus the paths listed in scripts/exclude.json, parented on the
    # previous release commit. Public main is that chain of release commits
    # and nothing else; development commits and excluded content are never
    # reachable from it. Earlier release tags that still point at commits of
    # another shape are re-pointed to their release commits in the same push.
    print_step "Pushing release $VERSION to kiss_ai repo (excluding paths listed in $EXCLUDE_FILE)..."
    create_public_commit "$(git rev-parse HEAD)" "$VERSION" "$RELEASE_CHAIN_TIP"
    verify_no_excluded_paths "$PUBLIC_COMMIT"
    verify_no_excluded_paths_in_history "$PUBLIC_COMMIT"
    TAG_OBJECT=$(release_tag "$TAG_NAME" "$VERSION" "$PUBLIC_COMMIT")
    push_public_snapshot "$PUBLIC_COMMIT" "$TAG_NAME" "$TAG_OBJECT" "$PUBLIC_HEAD" \
        "${RETAG_LEASES[@]}" "${RETAG_REFSPECS[@]}"
    print_info "Pushed release commit and tag $TAG_NAME to kiss_ai repo"
    if [[ ${#RETAG_REFSPECS[@]} -gt 0 ]]; then
        print_warn "Re-pointed ${#RETAG_REFSPECS[@]} earlier release tag(s) to release-only commits;"
        print_warn "clones must 'git fetch --tags --force' (install.sh does), and GitHub's source archives for those tags changed"
    fi

    # Step 8: Create GitHub release
    print_step "Creating GitHub release..."
    gh release create "$TAG_NAME" \
        --repo ksenxx/kiss_ai \
        --title "KISS $VERSION" \
        --notes "Release $VERSION"
    print_info "GitHub release created: https://github.com/ksenxx/kiss_ai/releases/tag/$TAG_NAME"

    # Step 9: Publish to PyPI
    print_step "Publishing to PyPI..."
    publish_to_pypi "$VERSION"

    # Step 10: Install the released version on this machine, before the stash
    # is restored so the install builds from the exact tree that was published.
    run_local_install

    # Step 11: Restore stashed changes
    trap - EXIT
    if [[ "$STASHED" == true ]]; then
        print_step "Restoring stashed changes..."
        git stash pop
        print_info "Stashed changes restored"
    fi

    echo
    print_info "========================================"
    print_info "Release completed successfully!"
    print_info "========================================"
    print_info "GitHub:  $PUBLIC_REPO_URL"
    print_info "PyPI:    https://pypi.org/project/${PYPI_PACKAGE_NAME}/"
    print_info "Version: $VERSION"
    print_info "Tag:     $TAG_NAME"
    echo
}

# Run main only when executed directly, so tests can source the functions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
