#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Put this checkout and its deployment on a remote host into total sync, by
# way of ``origin``.
#
# Usage:  scripts/sync-repo.sh user@ip-address <local-repo> <remote-dir> [branch]
#         scripts/sync-repo.sh --sync <repo-dir> [branch]
#
# Called by ./sorcar-cloud in place of copying the project folder, and useful
# on its own to refresh a running deployment's source.  The second form syncs
# one repository with its ``origin`` and is what runs on the remote host: the
# driver feeds *this file* to ``bash -s`` over ssh, so both machines run the
# same code even on a first deploy, when the remote has no checkout yet.
#
# The driver performs three passes, which is what makes the two working trees
# converge rather than merely trade files:
#
#   1. this machine <-> origin, so every commit and every uncommitted change
#      here is on origin before the remote looks at it;
#   2. the remote <-> origin, which creates the checkout if it is missing and
#      pushes back whatever the remote gained since the last deploy (an agent
#      that ran there has commits of its own);
#   3. this machine <-> origin again, to collect what pass 2 pushed.
#
# Afterwards both repositories hold the same commit on every branch, both
# working trees are clean, and origin holds all of it.  The last step of each
# pass asks origin what it really has (``git ls-remote``) and compares, so
# "in sync" is a checked statement rather than a hopeful one.
#
# What "sync a branch" means here, for the union of the local branches and
# origin's (the whole set, in both directions -- a full mirror):
#
#   * missing on one side .......... created there;
#   * one side strictly ahead ...... the other is fast-forwarded (a branch
#                                    that is not checked out is moved by
#                                    ``update-ref``; the checked-out one is
#                                    fast-forwarded by ``merge --ff-only``,
#                                    so its files follow);
#   * diverged ..................... merged, in the working tree for the
#                                    checked-out branch and in a throw-away
#                                    worktree for the others.
#
# Nothing is ever deleted or force-pushed:
#
#   * uncommitted work becomes a real commit first (``git add -A``), because
#     only commits travel through origin.  A commit that would delete more
#     than half of the files the branch has is refused -- that is not a
#     working tree to be shipped, it is an accident;
#   * a branch deleted on one side is recreated from the other rather than
#     deleted on both.  Deleting a branch everywhere is a decision, not a
#     side effect of a deploy;
#   * a commit that only a detached HEAD holds gets a branch of its own
#     ("sorcar-rescued-<time>") before that HEAD is left for the branch being
#     deployed, because nothing would point at it afterwards;
#   * ``.gitignore``d files (``.venv``, ``tmp/``, build output) are not part
#     of the repository and therefore do not travel;
#   * two kinds of branch are reported and left alone instead of being
#     synced: the agent's own ``kiss/wt-*`` scratch branches, and any branch
#     checked out in another worktree -- moving those refs from the outside
#     would desynchronize a running task's index.
#
# When a branch cannot be brought in line -- a merge that conflicts, a push
# origin refuses -- it is named in a warning and left as it is on both sides.
# That is fatal only for the branch the repository has checked out, because a
# deployment that is not on the branch it was asked for, or a laptop whose own
# branch silently stayed behind, is worse than a deploy that stops and says
# so.
#
# Environment overrides (all optional, driver only):
#   SORCAR_GITHUB_USER  GitHub account the remote commits and pushes as
#                       (default: ksenxx)
#   SORCAR_GIT_NAME     git author name on the remote (default: the account)
#   SORCAR_GIT_EMAIL    git author email on the remote
#                       (default: <account>@users.noreply.github.com)
set -euo pipefail

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
step() { printf '\033[0;34m[STEP]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()  { printf '\033[0;31m[ERR]\033[0m  %s\n' "$*" >&2; exit 1; }

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP_TAG="$(date -u +%Y%m%dT%H%M%SZ)"   # the same instant, usable in a ref name
HOST="$(hostname -s 2>/dev/null || hostname)"

# A value that survives being pasted into a remote shell command: everything
# is wrapped in single quotes, and a single quote inside ends the quoting,
# escapes itself and starts it again.  Without this, a git author name like
# O'Brien breaks the ssh command line -- or worse, extends it.
shquote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

# ---------------------------------------------------------------------------
# Reading the repository
# ---------------------------------------------------------------------------
current_branch() { git symbolic-ref -q --short HEAD 2>/dev/null || true; }

# The branches no deploy may touch: the agent's scratch branches, and the ones
# checked out in another worktree.  A branch can be checked out in one
# worktree only, so every branch in ``git worktree list`` that is not the one
# here belongs to someone else -- the same answer as comparing directories,
# without depending on how each path is spelled.
locked_branches() {
    local mine listed
    mine="$(current_branch)"
    listed="$(git worktree list --porcelain 2>/dev/null \
        | awk '/^branch /{ sub(/^refs\/heads\//, "", $2); print $2 }')"
    [[ -n "$mine" ]] && listed="$(printf '%s\n' "$listed" | grep -vxF "$mine" || true)"
    printf '%s\n' "$listed" | grep -v '^$' || true
}

# The union of the local branches and origin's, minus the untouchable ones.
syncable_branches() {
    local locked
    locked="$(locked_branches)"
    {
        git for-each-ref --format='%(refname:short)' refs/heads
        git for-each-ref --format='%(refname:lstrip=3)' refs/remotes/origin
    } | sort -u | while read -r branch; do
        case "$branch" in ""|HEAD|kiss/wt-*) continue ;; esac
        printf '%s\n' "$locked" | grep -qxF "$branch" && continue
        printf '%s\n' "$branch"
    done
}

# ---------------------------------------------------------------------------
# Turning the working tree into a commit
#
# Only commits travel through origin, so uncommitted work is committed where
# it is.  The guard is there because this function also runs on a checkout
# that was just attached to a history it did not come from (a first deploy
# adopting files that were already in the folder): if that ever leaves the
# working tree emptier than the history it now claims, the deploy stops
# instead of committing -- and pushing -- the deletion of the project.  The
# comparison is against the commit, not the index, so adding files cannot
# licence deleting others.
# ---------------------------------------------------------------------------
commit_working_tree() {
    local had removed
    git add -A
    git diff --cached --quiet && return 0
    if git rev-parse -q --verify HEAD >/dev/null; then
        had="$(git ls-tree -r --name-only HEAD | wc -l | tr -d ' ')"
        removed="$(git diff --cached --name-only --diff-filter=D | wc -l | tr -d ' ')"
        ((removed * 2 <= had)) \
            || die "Refusing to commit: this would delete $removed of the $had files on $(current_branch) in $PWD."
    fi
    git commit -q -m "sorcar-cloud: uncommitted changes from $HOST at $STAMP"
    info "Committed the working tree of $PWD on $(current_branch)."
}

# ---------------------------------------------------------------------------
# Attaching a folder to a branch without deleting anything
#
# Only for a repository that has no commit of its own yet: a first deploy,
# where the folder may already hold files (an older deploy copied them there)
# or nothing at all.  ``reset --mixed`` puts the index on origin's commit
# while leaving every file alone, and the files that the history has and the
# folder does not are then restored -- a full checkout when the folder was
# empty, only the missing pieces when it was not.  Local modifications
# survive both, and commit_working_tree turns them into the next commit.
# ---------------------------------------------------------------------------
adopt_branch() {
    local want="$1" base="" candidate sha
    for candidate in "refs/remotes/origin/$want" refs/remotes/origin/main \
                     refs/remotes/origin/master; do
        if sha="$(git rev-parse -q --verify "$candidate^{commit}")"; then
            base="$sha"
            break
        fi
    done
    if [[ -z "$base" ]]; then
        git symbolic-ref HEAD "refs/heads/$want"
        warn "origin has no history to start $want from — $PWD begins empty."
        return 0
    fi
    git update-ref "refs/heads/$want" "$base"
    git symbolic-ref HEAD "refs/heads/$want"
    git reset -q --mixed "$base"
    # Writes the files the index has and the folder lacks, and -- without -f --
    # leaves the ones that are already there alone, so a file modified here
    # keeps its modifications.  It reports the files it skipped, hence -q and
    # the tolerated exit status.
    git checkout-index -a -q || true
    info "Attached $PWD to $want at $(git rev-parse --short "$base")."
}

# A detached HEAD has no branch to carry work to origin, so uncommitted work
# there would become a commit no branch points at -- lost from the working tree
# the moment the wanted branch is checked out.  A clean detached checkout is
# fine: there is nothing to carry.
refuse_detached_work() {
    [[ -n "$(current_branch)" ]] && return 0
    [[ -z "$(git status --porcelain)" ]] \
        || die "$PWD has a detached HEAD with uncommitted changes — check out a branch there first."
    return 0
}

# Give a name to commits that only a detached HEAD holds, before that HEAD is
# left behind for the branch being deployed.  Nothing points at such a commit
# afterwards: it stops being reachable, and git eventually collects it -- so
# somebody's work would be gone with no branch, no push and no warning to say
# where it went.  A branch is created instead, which the sync then mirrors to
# origin like any other.  Commits that a branch already contains need nothing.
rescue_detached_head() {
    [[ -z "$(current_branch)" ]] || return 0
    local head rescue
    head="$(git rev-parse -q --verify HEAD)" || return 0
    [[ -z "$(git for-each-ref --contains "$head" --count=1 \
             refs/heads refs/remotes 2>/dev/null)" ]] || return 0
    rescue="sorcar-rescued-$STAMP_TAG"
    while git rev-parse -q --verify "refs/heads/$rescue" >/dev/null; do
        rescue="$rescue+"
    done
    git branch "$rescue" "$head" \
        || die "$PWD has a detached HEAD with commits of its own that cannot be" \
               "put on a branch — check it out somewhere before deploying."
    warn "$PWD had commits only its detached HEAD held; they are on branch $rescue now."
}

# Put the requested branch in the working tree, keeping the work that is
# there.  A repository without a commit is adopted; one that sits on another
# branch commits that branch's work before leaving it.  A checkout that
# refuses (the branch is checked out in another worktree, an untracked file is
# in the way) is fatal: the deployment has to be on the branch it was asked
# for, and forcing it would throw away what is in the folder.
attach_branch() {
    local want="$1"
    if ! git rev-parse -q --verify HEAD >/dev/null; then
        adopt_branch "$want"
        return 0
    fi
    [[ "$(current_branch)" == "$want" ]] && return 0
    refuse_detached_work
    rescue_detached_head
    [[ -n "$(current_branch)" ]] && commit_working_tree
    if git rev-parse -q --verify "refs/heads/$want" >/dev/null; then
        git checkout -q "$want" || die "Cannot check out $want in $PWD."
    elif git rev-parse -q --verify "refs/remotes/origin/$want" >/dev/null; then
        git checkout -q -b "$want" --track "origin/$want" \
            || die "Cannot create $want from origin/$want in $PWD."
    else
        git checkout -q -b "$want" || die "Cannot create branch $want in $PWD."
    fi
    info "$PWD is now on $want."
}

# ---------------------------------------------------------------------------
# Merging a branch that is not the checked-out one
#
# A throw-away worktree, created outside the repository so no agent mistakes
# it for one of its own: ``git merge`` needs a working tree, and the real one
# belongs to another branch (and possibly to a running task).  The ref is
# moved with a compare-and-swap on the commit the merge started from.
# ---------------------------------------------------------------------------
merge_detached() {
    local branch="$1" local_sha="$2" origin_sha="$3" scratch merged=""
    scratch="$(mktemp -d)"
    rm -rf "$scratch"                      # git wants to create it itself
    if git worktree add -q --detach "$scratch" "$local_sha" 2>/dev/null; then
        if git -C "$scratch" merge --no-edit --no-stat -q \
                -m "sorcar-cloud: merge origin/$branch into $branch" \
                "$origin_sha" >/dev/null 2>&1; then
            merged="$(git -C "$scratch" rev-parse HEAD)"
        else
            git -C "$scratch" merge --abort >/dev/null 2>&1 || true
        fi
        git worktree remove --force "$scratch" >/dev/null 2>&1 || true
    fi
    rm -rf "$scratch"
    git worktree prune
    [[ -n "$merged" ]] || return 1
    git update-ref "refs/heads/$branch" "$merged" "$local_sha"
}

# ---------------------------------------------------------------------------
# One branch
#
# A ref rejected in pass one is unchanged when pass three starts: that pass is
# for collecting what the remote pushed, not for repeating the same upload.
# Remember failures only for this driver process; the next deployment retries
# in case permissions, history, or origin's policy have changed meanwhile.
FAILED_PUSH_REFS=""
queue_push_ref() {
    local spec="$1"
    printf '%s\n' "$FAILED_PUSH_REFS" | grep -qxF "$spec" && return 0
    PUSH_REFS[${#PUSH_REFS[@]}]="$spec"
}

remember_failed_pushes() {
    local spec
    for spec in "$@"; do
        FAILED_PUSH_REFS="${FAILED_PUSH_REFS}${FAILED_PUSH_REFS:+$'\n'}$spec"
    done
}

# Fills PUSH_REFS with the branches origin has to catch up on; the push
# itself is one command per repository (seventeen branches over https is one
# connection, not seventeen).  Whether a branch really ended up in sync is
# not decided here but in check_convergence, which asks origin.
# ---------------------------------------------------------------------------
sync_branch() {
    local branch="$1" local_sha origin_sha
    local_sha="$(git rev-parse -q --verify "refs/heads/$branch^{commit}" || true)"
    origin_sha="$(git rev-parse -q --verify "refs/remotes/origin/$branch^{commit}" || true)"

    if [[ -z "$origin_sha" ]]; then                       # only here: mirror it up
        queue_push_ref "refs/heads/$branch:refs/heads/$branch"
        return 0
    fi
    if [[ -z "$local_sha" ]]; then                        # only on origin: mirror it down
        if git branch -q "$branch" "refs/remotes/origin/$branch"; then
            git branch -q --set-upstream-to="origin/$branch" "$branch" 2>/dev/null || true
            info "Created $branch from origin/$branch."
        else
            warn "Could not create $branch from origin/$branch."
        fi
        return 0
    fi
    git branch -q --set-upstream-to="origin/$branch" "$branch" 2>/dev/null || true
    [[ "$local_sha" == "$origin_sha" ]] && return 0

    if git merge-base --is-ancestor "$origin_sha" "$local_sha"; then
        queue_push_ref "refs/heads/$branch:refs/heads/$branch"
    elif git merge-base --is-ancestor "$local_sha" "$origin_sha"; then
        if [[ "$branch" == "$(current_branch)" ]]; then
            git merge --ff-only -q "$origin_sha" \
                && info "Fast-forwarded the checked-out $branch to origin/$branch." \
                || warn "Could not fast-forward $branch to origin/$branch."
        else
            git update-ref "refs/heads/$branch" "$origin_sha" "$local_sha"
            info "Fast-forwarded $branch to origin/$branch."
        fi
    elif merge_branch "$branch" "$local_sha" "$origin_sha"; then
        info "Merged origin/$branch into the diverged $branch."
        queue_push_ref "refs/heads/$branch:refs/heads/$branch"
    else
        warn "$branch and origin/$branch have conflicting changes." \
             "Merge it by hand: git checkout $branch && git merge origin/$branch"
    fi
}

# Merge origin's side into a diverged branch, in the working tree when that
# is where the branch lives and off to the side when it is not.
merge_branch() {
    local branch="$1" local_sha="$2" origin_sha="$3"
    if [[ "$branch" != "$(current_branch)" ]]; then
        merge_detached "$branch" "$local_sha" "$origin_sha"
        return $?
    fi
    if git merge --no-edit --no-stat -q \
            -m "sorcar-cloud: merge origin/$branch into $branch" \
            "$origin_sha" >/dev/null 2>&1; then
        return 0
    fi
    git merge --abort >/dev/null 2>&1 || true
    return 1
}

# ---------------------------------------------------------------------------
# Did it work?
#
# origin is the meeting point, so origin is the authority: whatever the local
# operations reported, a branch is in sync only if origin has exactly the
# commit this repository has.  This also catches the branches a push refused,
# without having to guess which of the refs in one push command failed.
# ---------------------------------------------------------------------------
check_convergence() {
    local heads branch local_sha origin_sha here there
    OUT_OF_SYNC=0
    CURRENT_OUT_OF_SYNC=""
    heads="$(git ls-remote --heads origin)" \
        || die "Cannot ask origin what it has — the sync of $PWD is unverified."
    while read -r branch; do
        [[ -n "$branch" ]] || continue
        local_sha="$(git rev-parse -q --verify "refs/heads/$branch^{commit}" || true)"
        origin_sha="$(printf '%s\n' "$heads" \
            | awk -v ref="refs/heads/$branch" '$2 == ref { print $1 }')"
        [[ -n "$local_sha" && "$local_sha" == "$origin_sha" ]] && continue
        OUT_OF_SYNC=$((OUT_OF_SYNC + 1))
        here="${local_sha:0:7}"
        there="${origin_sha:0:7}"
        warn "$branch is not in sync: ${here:-nothing} here, ${there:-nothing} on origin."
        if [[ "$branch" == "$(current_branch)" ]]; then
            CURRENT_OUT_OF_SYNC="$branch"
        fi
    done <<< "$BRANCHES"
    return 0
}

# ---------------------------------------------------------------------------
# One repository: --sync <repo-dir> [branch]
#
# ``branch`` is given for the deployment (its working tree must end up on the
# branch being deployed) and omitted for a checkout that belongs to whoever is
# using it, which is never moved to another branch by a deploy.
# ---------------------------------------------------------------------------
sync_repo() {
    local repo="$1" want="${2:-}" branch skipped scratch
    command -v git >/dev/null 2>&1 || die "git is not installed on $HOST."
    mkdir -p "$repo" || die "Cannot create $repo."
    cd "$repo"
    repo="$PWD"                        # this function cd's, so keep it absolute

    if [[ ! -e .git ]]; then
        [[ -n "$want" ]] || die "$repo is not a git repository."
        # No "git init -b": that option is from git 2.28, and a server can
        # well be older.  HEAD is put on the wanted branch by adopt_branch.
        git init -q
        info "Initialized a git repository in $repo."
    fi
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die "$repo is not a git working tree."
    # A half-finished merge or rebase owns the working tree; committing or
    # merging on top of one produces a mess that is hard to unpick.
    [[ -e "$(git rev-parse --git-path MERGE_HEAD)" \
       || -e "$(git rev-parse --git-path rebase-merge)" \
       || -e "$(git rev-parse --git-path rebase-apply)" ]] \
        && die "$repo is in the middle of a merge or rebase — finish it, then deploy."
    # Registrations of worktrees whose directories are gone would otherwise
    # keep their branches locked out of the sync.
    git worktree prune

    if [[ -n "$want" ]]; then
        git config user.name "${SYNC_GIT_NAME:-$GIT_NAME_DEFAULT}"
        git config user.email "${SYNC_GIT_EMAIL:-$GIT_EMAIL_DEFAULT}"
        git config github.user "${SYNC_GITHUB_USER:-$GITHUB_USER_DEFAULT}"
        git config credential.username "${SYNC_GITHUB_USER:-$GITHUB_USER_DEFAULT}"
        # GitHub over ssh with the key this host was given, so no token is
        # ever needed (or shipped) for the deployment to fetch and push.
        git config url."git@github.com:".insteadOf "https://github.com/"
        if [[ -n "${SYNC_REMOTES_B64:-}" ]]; then
            printf '%s' "$SYNC_REMOTES_B64" | base64 -d | while IFS=$'\t' read -r name url; do
                [[ -n "$name" && -n "$url" ]] || continue
                git remote get-url "$name" >/dev/null 2>&1 \
                    && git remote set-url "$name" "$url" \
                    || git remote add "$name" "$url"
            done
        fi
        export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes'
    fi

    git remote get-url origin >/dev/null 2>&1 \
        || die "$repo has no 'origin' remote — there is nothing to sync through."
    # Every branch origin has, whatever this clone's own refspec says (a
    # --single-branch clone would otherwise see one), plus origin's tags.
    # Local-only tags are left alone: a tag marks a release, not a working
    # state, so a deploy must not publish one.
    step "Fetching origin into $repo ..."
    if ! git fetch --prune --tags --quiet origin '+refs/heads/*:refs/remotes/origin/*'; then
        # A local tag that names a different commit than origin's makes the
        # whole fetch fail, and moving it is exactly what must not happen: a
        # tag marks a release.  The branches are what a deploy needs, so they
        # are fetched on their own and the tag is left for its owner to sort
        # out.
        git fetch --prune --quiet --no-tags origin '+refs/heads/*:refs/remotes/origin/*' \
            || die "Cannot fetch from origin in $repo — nothing can be synced through it."
        warn "Some of origin's tags were not fetched into $repo (a local tag of the" \
             "same name names another commit); the branches were."
    fi

    if [[ -n "$want" ]]; then
        attach_branch "$want"
    else
        refuse_detached_work
        [[ -z "$(current_branch)" ]] \
            && warn "$repo has a detached HEAD — its branches are synced, but nothing is committed here."
    fi
    [[ -n "$(current_branch)" ]] && commit_working_tree

    BRANCHES="$(syncable_branches)"
    skipped="$(locked_branches | tr '\n' ' ')"
    [[ -n "${skipped// /}" ]] \
        && warn "Not syncing branches checked out in another worktree: ${skipped% }"
    scratch="$(git for-each-ref --format='%(refname:lstrip=2)' 'refs/heads/kiss/wt-*' | tr '\n' ' ')"
    [[ -n "${scratch// /}" ]] \
        && warn "Not syncing the agent's scratch branches: ${scratch% }"

    PUSH_REFS=()
    while read -r branch; do
        [[ -n "$branch" ]] && sync_branch "$branch"
    done <<< "$BRANCHES"

    # The checked-out branch is the deployment.  Push it independently so an
    # archival branch rejected by origin cannot reject the requested deploy in
    # the same receive transaction; auxiliary refs remain a best-effort mirror.
    PRIMARY_PUSH_REFS=()
    AUXILIARY_PUSH_REFS=()
    current="$(current_branch)"
    for spec in ${PUSH_REFS[@]+"${PUSH_REFS[@]}"}; do
        if [[ "$spec" == "refs/heads/$current:refs/heads/$current" ]]; then
            PRIMARY_PUSH_REFS[${#PRIMARY_PUSH_REFS[@]}]="$spec"
        else
            AUXILIARY_PUSH_REFS[${#AUXILIARY_PUSH_REFS[@]}]="$spec"
        fi
    done
    if ((${#PRIMARY_PUSH_REFS[@]} > 0)); then
        step "Pushing checked-out branch $current from $repo to origin ..."
        if ! git push --quiet origin \
                ${PRIMARY_PUSH_REFS[@]+"${PRIMARY_PUSH_REFS[@]}"}; then
            warn "The checked-out branch could not be pushed to origin."
            remember_failed_pushes "${PRIMARY_PUSH_REFS[@]}"
        fi
    fi
    if ((${#AUXILIARY_PUSH_REFS[@]} > 0)); then
        step "Pushing ${#AUXILIARY_PUSH_REFS[@]} other branch(es) from $repo to origin ..."
        if ! git push --quiet origin \
                ${AUXILIARY_PUSH_REFS[@]+"${AUXILIARY_PUSH_REFS[@]}"}; then
            warn "Some branches could not be pushed to origin."
            remember_failed_pushes "${AUXILIARY_PUSH_REFS[@]}"
        fi
    fi

    check_convergence
    [[ -z "$CURRENT_OUT_OF_SYNC" ]] \
        || die "$repo is on $CURRENT_OUT_OF_SYNC, which is not in sync with origin — see the warnings above."
    if ((OUT_OF_SYNC > 0)); then
        warn "$repo: $OUT_OF_SYNC branch(es) are not in sync with origin (the checked-out one is)."
    else
        info "$repo is in sync with origin on $(current_branch) at $(git log -1 --format='%h %s' 2>/dev/null || echo 'no commits')."
    fi
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
GITHUB_USER_DEFAULT="${SORCAR_GITHUB_USER:-ksenxx}"
GIT_NAME_DEFAULT="${SORCAR_GIT_NAME:-$GITHUB_USER_DEFAULT}"
GIT_EMAIL_DEFAULT="${SORCAR_GIT_EMAIL:-$GITHUB_USER_DEFAULT@users.noreply.github.com}"

usage() {
    awk '/^# Usage:/{p=1} p&&/^#/{sub(/^# ?/,""); print} p&&!/^#/{exit}' "$0"
}

if [[ "${1:-}" == "--sync" ]]; then
    [[ -n "${2:-}" ]] || die "Usage: $0 --sync <repo-dir> [branch]"
    sync_repo "$2" "${3:-}"
    exit 0
fi

TARGET="${1:-}"
LOCAL_REPO="${2:-}"
REMOTE_DIR="${3:-}"
BRANCH="${4:-}"
case "$TARGET" in
    ""|-h|--help) usage; [[ -n "$TARGET" ]] && exit 0 || exit 1 ;;
esac
[[ -n "$LOCAL_REPO" && -n "$REMOTE_DIR" ]] \
    || die "Usage: $0 user@ip-address <local-repo> <remote-dir> [branch]"
git -C "$LOCAL_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "$LOCAL_REPO is not a git working tree."
LOCAL_REPO="$(cd "$LOCAL_REPO" && pwd)"        # sync_repo cd's; passes 1 and 3
[[ "$REMOTE_DIR" == /* ]] \
    || die "The remote folder must be an absolute path, got '$REMOTE_DIR'."
if [[ -z "$BRANCH" ]]; then
    BRANCH="$(git -C "$LOCAL_REPO" symbolic-ref -q --short HEAD || echo main)"
fi
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# The remote is given the same remotes this checkout has, so it fetches and
# pushes to the same places.  base64 keeps the multi-line "name<TAB>url" list
# intact across the ssh command line, and reading the URLs one at a time keeps
# the ones containing spaces whole.
remote_list() {
    local name
    for name in $(git -C "$LOCAL_REPO" remote); do
        printf '%s\t%s\n' "$name" "$(git -C "$LOCAL_REPO" remote get-url "$name")"
    done
}
REMOTES_B64="$(remote_list | base64 | tr -d '\n')"

step "1/3  Syncing $LOCAL_REPO with origin ..."
sync_repo "$LOCAL_REPO"

step "2/3  Syncing $TARGET:$REMOTE_DIR with origin (branch $BRANCH) ..."
ssh "$TARGET" "SYNC_REMOTES_B64=$(shquote "$REMOTES_B64") \
    SYNC_GIT_NAME=$(shquote "$GIT_NAME_DEFAULT") \
    SYNC_GIT_EMAIL=$(shquote "$GIT_EMAIL_DEFAULT") \
    SYNC_GITHUB_USER=$(shquote "$GITHUB_USER_DEFAULT") \
    bash -s -- --sync $(shquote "$REMOTE_DIR") $(shquote "$BRANCH")" < "$SELF" \
    || die "Syncing $TARGET:$REMOTE_DIR failed."

step "3/3  Bringing what $TARGET pushed back into $LOCAL_REPO ..."
sync_repo "$LOCAL_REPO"

info "$LOCAL_REPO and $TARGET:$REMOTE_DIR are in sync through origin."
