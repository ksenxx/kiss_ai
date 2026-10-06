# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Background pre-creation of spare git worktrees.

``git worktree add`` performs a full checkout, which takes seconds to
tens of seconds on a large repository and dominates the delay between a
user submitting a task and the agent actually starting.  This module
keeps at most ONE ready-to-use spare worktree per repository, created
on a background thread while no task is waiting for it.  When the next
task starts, :meth:`WorktreeSorcarAgent._acquire_task_worktree`
consumes the spare and merely hard-resets it onto the tip of the
task's original branch — a near-instant operation when HEAD moved
little — instead of paying for the full checkout on the submit path.

A refill is scheduled when the daemon starts and when a worktree task
finishes — not when a task starts, so its disk traffic never competes
with the launch it would otherwise run beside.  Only the cheap steps
hold ``repo_lock``: the maintenance passes, then a ``--no-checkout``
registration of the spare (milliseconds); the checkout itself runs
with no lock held, so launches, stops and merges never wait for it.

The orphan-maintenance passes (``reclaim_orphaned_worktrees`` and
``sweep_orphaned_state``) formerly ran on the submit path too; a pool
refill runs them instead, so their cost is also moved off the
user-visible delay whenever a spare is available.

Safety properties:

* A spare is indistinguishable from a normal task worktree on disk
  (same ``kiss/wt-*`` branch naming, same ``.kiss-worktrees/``
  directory), so a spare orphaned by a crashed process is cleaned up
  by the existing reclaim pass like any other clean leftover worktree.
* Live spares are protected from THIS process's reclaim by
  :func:`spare_branches`, which callers union into the reclaim
  exclusion set (see ``WorktreeSorcarAgent._live_worktree_branches``),
  and from OTHER processes' reclaim by the owner pid that
  :meth:`GitWorktreeOps.create` stamps on every new worktree branch
  (the reclaim pass skips branches whose owner is still alive).
* All pool state is in-process; consumers re-validate that the spare's
  branch and directory still exist before using it and fall back to
  the plain inline ``git worktree add`` path on any failure.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_BRANCH_PREFIX,
    _WORKTREE_SUBDIR,
    GitWorktreeOps,
    _reclaim_process_lock,
    repo_lock,
)

logger = logging.getLogger(__name__)

# Setting this env var to "1" stops :func:`prewarm_async` from
# scheduling background refills.  The test suite sets it globally
# (see the root ``conftest.py``) so the hundreds of existing tests
# that create agents in temporary repositories are not raced by
# refill threads writing spare worktrees into directories that are
# being asserted on and torn down; the pool's own tests re-enable it.
_DISABLE_ENV = "KISS_DISABLE_WORKTREE_POOL"


def pool_enabled() -> bool:
    """Whether background spare-worktree refills may be scheduled.

    Returns:
        ``False`` when the ``KISS_DISABLE_WORKTREE_POOL`` environment
        variable is set to ``"1"``, ``True`` otherwise.
    """
    return os.environ.get(_DISABLE_ENV, "") != "1"


# Resolved repo root -> (branch, worktree dir) of the ready spare.
_spares: dict[str, tuple[str, Path]] = {}
# Resolved repo roots with a refill currently running (dedup guard).
_prewarming: set[str] = set()
# Resolved repo root -> branch of a registered spare whose checkout is
# still running outside ``repo_lock`` (see :func:`prewarm`).  Reported
# by :func:`spare_branches` so no reclaim pass touches it meanwhile.
_building: dict[str, str] = {}
# Resolved repo root -> the background refill thread last spawned by
# :func:`prewarm_async`; joined by :func:`discard_all` so an in-flight
# refill cannot publish a spare after the sweep.
_refill_threads: dict[str, threading.Thread] = {}
# Pool generation.  :func:`discard_all` advances it when it starts and
# again when it finishes; a refill captures it when it starts and
# publishes its spare only if it is unchanged.  This is what makes
# ``discard_all`` final: a refill scheduled after its thread snapshot
# (so never joined), or a snapshotted one that outlived the bounded
# join, cannot publish a spare into a pool the caller believes empty —
# the spare is removed instead.
_generation = 0
# Number of :func:`discard_all` calls currently between their spare
# snapshot and their final generation bump.  A refill that started
# after a discard's first bump (so its captured generation still
# matches) but reaches its publication point during that sweep would
# otherwise publish a spare the sweep never saw; the publication check
# requires this to be zero as well.  A COUNTER, not a boolean: with
# two overlapping discards a boolean was cleared by the first finisher
# while the second was still sweeping, so a refill started in between
# published a spare the second discard returned without ever seeing.
_active_discards = 0
_pool_lock = threading.Lock()


def _repo_key(repo: Path) -> str:
    """Return the pool key for *repo* (its resolved root path).

    Args:
        repo: Git repo root path.

    Returns:
        The resolved path string used to key the pool maps.
    """
    return str(repo.resolve())


def new_task_branch(repo: Path) -> str:
    """Mint a unique ``kiss/wt-*`` branch name for *repo*.

    Same naming scheme the inline worktree setup has always used:
    ``kiss/wt-<epoch>-<uuid8>`` with a numeric suffix appended in the
    (practically impossible) case of a collision.

    Args:
        repo: Git repo root path.

    Returns:
        A branch name that does not currently exist in *repo*.
    """
    branch = (
        f"{_WORKTREE_BRANCH_PREFIX}"
        f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    )
    base_branch = branch
    suffix = 1
    while GitWorktreeOps.branch_exists(repo, branch):  # pragma: no branch
        branch = f"{base_branch}-{suffix}"
        suffix += 1
    return branch


def spare_branches() -> set[str]:
    """Return the branch names of every pooled spare (all repos).

    Used to build reclaim exclusion sets so an orphan-worktree reclaim
    pass never merges or deletes a spare the pool is holding.

    Returns:
        Set of ``kiss/wt-*`` branch names currently pooled or still
        being populated by a refill.
    """
    with _pool_lock:
        return {branch for branch, _ in _spares.values()} | set(_building.values())


def take_spare(repo: Path) -> tuple[str, Path] | None:
    """Pop and validate the pooled spare worktree for *repo*.

    The spare is removed from the pool unconditionally; when its branch
    or directory has vanished (e.g. an external process reclaimed it)
    the stale entry is dropped and ``None`` is returned so the caller
    falls back to creating a worktree inline.

    Args:
        repo: Git repo root path.

    Returns:
        ``(branch, wt_dir)`` of a spare whose branch and directory both
        still exist, or ``None`` when the pool has nothing usable.
    """
    with _pool_lock:
        spare = _spares.pop(_repo_key(repo), None)
    if spare is None:
        return None
    branch, wt_dir = spare
    if not wt_dir.is_dir() or not GitWorktreeOps.branch_exists(repo, branch):
        logger.warning(
            "Pooled spare worktree %s (branch %s) vanished; "
            "falling back to inline creation",
            wt_dir,
            branch,
        )
        return None
    if GitWorktreeOps.current_branch(wt_dir) != branch:
        # An external git command switched the spare's checkout.  Using
        # it anyway would `reset --hard` — and later commit the task's
        # work onto — whatever branch is actually checked out, while
        # every metadata/merge path keeps using the recorded name.
        # Leave the directory alone (the content is not ours to
        # destroy); the reclaim pass handles the leftover.
        logger.warning(
            "Pooled spare worktree %s no longer has branch %s checked "
            "out; falling back to inline creation",
            wt_dir,
            branch,
        )
        return None
    if GitWorktreeOps.spare_has_content(repo, branch, wt_dir):
        # A spare is never written to, so content in it means an
        # external writer put something there.  Consuming it would
        # destroy that content (`reset --hard` + `git clean -fdq`),
        # which contradicts the preservation policy the reclaim pass
        # applies to the very same situation (same probe).  Leave the
        # directory for the reclaim pass to preserve; create inline
        # instead.
        logger.warning(
            "Pooled spare worktree %s (branch '%s') has unexpected "
            "content; preserving it and falling back to inline "
            "creation",
            wt_dir,
            branch,
        )
        return None
    return spare


def discard_all() -> None:
    """Remove every pooled spare worktree and forget the pool state.

    Joins the in-flight background refill threads first (bounded
    wait), so a refill that is still creating its worktree cannot
    publish a new spare right after the sweep.  Refills this cannot
    join — scheduled after the thread snapshot, or still running when
    the bounded join gives up — are fenced off by the pool generation
    (see :data:`_generation`): advanced here before the snapshot and
    once more on return, it makes every refill started before this
    call returns drop its spare instead of publishing it — together
    with the :data:`_active_discards` gate, held for the whole call,
    which also stops a refill started after the first bump (or after
    an OVERLAPPING discard_all finished) from publishing while this
    sweep is still running.  Best-effort
    cleanup hook for tests and embedders; a spare that cannot be
    removed is left for the reclaim pass to collect.
    """
    global _active_discards, _generation
    with _pool_lock:
        _generation += 1
        _active_discards += 1
        threads = list(_refill_threads.values())
        _refill_threads.clear()
    try:
        for thread in threads:
            thread.join(timeout=120)
        with _pool_lock:
            spares = dict(_spares)
            _spares.clear()
        for key, (branch, wt_dir) in spares.items():
            repo = Path(key)
            try:
                # Same guard the reclaim pass applies to orphaned
                # spares: a spare is never written to, so content in it
                # means an external writer put something there —
                # preserve it for the reclaim pass to inspect rather
                # than destroy it.  A spare whose directory is already
                # gone is plumbing only and is always cleaned up.
                if wt_dir.is_dir():
                    if GitWorktreeOps.spare_has_content(repo, branch, wt_dir):
                        logger.warning(
                            "Pooled spare worktree %s (branch '%s') has "
                            "unexpected content; preserving instead of "
                            "discarding",
                            wt_dir, branch,
                        )
                        continue
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
            except Exception:  # pragma: no cover — filesystem teardown race
                logger.warning(
                    "Failed to discard pooled spare %s", wt_dir, exc_info=True,
                )
    finally:
        # Always release the gate — a KeyboardInterrupt landing in the
        # sweep must not leave the counter raised forever, which would
        # silently disable every future refill publication.
        with _pool_lock:
            _generation += 1
            _active_discards -= 1


def prewarm(
    repo: Path,
    exclude_branches_fn: Callable[[], set[str]] | None = None,
) -> bool:
    """Ensure a spare worktree exists for *repo* (synchronous).

    Runs the orphan-maintenance passes (reclaim + sweep) and registers
    one empty spare worktree under ``repo_lock`` (see
    :func:`_register_spare`), then populates it with no lock held and
    publishes it.  A no-op when a spare is already pooled or another
    thread is refilling.

    Args:
        repo: Git repo root path.
        exclude_branches_fn: Callable evaluated right before the
            reclaim pass, returning the branches owned by live agents
            (they must never be reclaimed).  Evaluated late — not at
            scheduling time — because a task may have started between
            scheduling and lock acquisition.  ``None`` skips the
            maintenance passes entirely (no safe exclusion set is
            available, and reclaim without one could merge a live
            task's worktree).

    Returns:
        ``True`` when the pool holds a spare for *repo* on return.
    """
    key = _repo_key(repo)
    with _pool_lock:
        if key in _spares:
            return True
        if key in _prewarming:
            return False
        _prewarming.add(key)
        generation = _generation
    try:
        registered = _register_spare(repo, exclude_branches_fn)
        if registered is None:
            return False
        branch, wt_dir = registered
        # The full checkout runs with NO lock held: on a large
        # repository it takes many seconds, and while it ran under
        # ``repo_lock`` every task launch, stop and merge in that
        # window stalled behind it.  The worktree is already registered,
        # owner-stamped and spare-marked, and ``_building`` keeps it
        # out of this process's reclaim pass (which would otherwise
        # find a worktree owned by this pid whose every file shows as
        # deleted).  The second reset warms the index stat-cache:
        # right after the checkout every index entry is racily clean
        # (file mtimes equal the index write time), so the next
        # ``git reset --hard`` would re-hash every file — ~1 s on a
        # large repo, landing on the submit path when the spare is
        # consumed.
        populated = (
            GitWorktreeOps.reset_worktree_to(wt_dir, "HEAD")
            and GitWorktreeOps.reset_worktree_to(wt_dir, "HEAD")
        )
        with _pool_lock:
            if populated and generation == _generation and _active_discards == 0:
                _spares[key] = (branch, wt_dir)
                return True
        if populated:
            # discard_all ran (or is running) since this refill started:
            # the pool it would publish into has been declared empty,
            # so the spare is removed rather than leaked past the sweep.
            # ``_active_discards`` covers the refill that started after
            # the first generation bump — or after an overlapping
            # discard_all re-stabilised the generation — and would
            # otherwise publish while a sweep is still running.
            logger.info(
                "Dropping spare worktree %s for %s: the pool was "
                "discarded while it was being created",
                wt_dir, repo,
            )
        GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
        return False
    except Exception:  # pragma: no cover — unexpected git failure
        logger.warning(
            "Worktree-pool prewarm failed for %s", repo, exc_info=True,
        )
        return False
    finally:
        with _pool_lock:
            _building.pop(key, None)
            _prewarming.discard(key)


def _register_spare(
    repo: Path,
    exclude_branches_fn: Callable[[], set[str]] | None,
) -> tuple[str, Path] | None:
    """Run pool maintenance and register an empty spare worktree under ``repo_lock``.

    Everything that must not interleave with a task's own multi-step
    git operations happens here — the orphan reclaim and sweep, the
    repository setup, and the ``--no-checkout`` worktree registration
    with its owner and spare stamps — and all of it takes well under a
    second, so the lock is never held across the checkout itself.

    Args:
        repo: Git repo root path.
        exclude_branches_fn: See :func:`prewarm`.

    Returns:
        ``(branch, wt_dir)`` of the registered, still unpopulated spare
        (recorded in :data:`_building`), or ``None`` when it could not
        be created (nothing is left on disk).
    """
    key = _repo_key(repo)
    with repo_lock(repo):
        if exclude_branches_fn is not None:
            try:
                excluded = exclude_branches_fn() | spare_branches()
                GitWorktreeOps.reclaim_orphaned_worktrees(
                    repo, exclude_branches=excluded,
                )
                GitWorktreeOps.sweep_orphaned_state(repo)
            except Exception:
                logger.warning(
                    "Worktree-pool maintenance failed for %s",
                    repo,
                    exc_info=True,
                )
        try:
            GitWorktreeOps.prepare_repo(repo)
        except Exception:  # pragma: no cover — filesystem permission
            logger.warning(
                "Failed to update git exclude", exc_info=True,
            )
        branch = new_task_branch(repo)
        wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
        # The spare marker is saved BEFORE the worktree exists: an
        # unmarked dead-owner worktree is auto-committed and
        # squash-merged by the reclaim pass, and an unpopulated one
        # (empty index, every file staged as deleted) would merge those
        # deletions into the user's branch should this process die
        # between the registration and the checkout.  With the marker
        # already in place the reclaim pass preserves or discards it
        # instead.  Both steps run under the cross-process reclaim lock
        # (``create`` re-enters it) so no peer sweep can purge the
        # marker of a branch that does not exist yet; a marker left
        # behind by a failed registration is swept later.
        with _reclaim_process_lock(repo):
            registered = GitWorktreeOps.save_spare_marker(
                repo, branch,
            ) and GitWorktreeOps.create(repo, branch, wt_dir, checkout=False)
        if not registered:
            GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
            return None
        with _pool_lock:
            _building[key] = branch
    return branch, wt_dir


def prewarm_async(
    repo: Path,
    exclude_branches_fn: Callable[[], set[str]] | None = None,
) -> threading.Thread | None:
    """Refill the pool for *repo* on a background daemon thread.

    Called when the daemon starts and when a worktree task ends.
    Cheap and idempotent: returns immediately without spawning when a
    spare is already pooled or a refill is already running.

    Args:
        repo: Git repo root path.
        exclude_branches_fn: See :func:`prewarm`.

    Returns:
        The started thread, or ``None`` when no refill was needed or
        the pool is disabled (see :func:`pool_enabled`).
    """
    if not pool_enabled():
        return None
    key = _repo_key(repo)
    thread = threading.Thread(
        target=prewarm,
        args=(repo, exclude_branches_fn),
        name=f"worktree-pool-prewarm-{Path(key).name}",
        daemon=True,
    )
    # Dedup check, thread registration, and start share ONE lock
    # acquisition: ``_prewarming`` is only populated inside
    # :func:`prewarm` after its thread starts, so with separate lock
    # blocks two callers could both pass the check and the second
    # registration would overwrite the first thread — leaving
    # :func:`discard_all` joining only the loser while the winner
    # publishes a spare after the sweep.  A live registered thread is
    # therefore itself treated as "refill in flight", and ``start()``
    # happens under the lock so a registered thread is always alive
    # until its refill has finished (a thread is only "alive" between
    # ``start()`` and the end of ``run()``).  The started thread
    # merely blocks on this same lock in :func:`prewarm` until the
    # block exits; nothing here waits on it, so no deadlock.
    with _pool_lock:
        if key in _spares or key in _prewarming:
            return None
        existing = _refill_threads.get(key)
        if existing is not None and existing.is_alive():
            return None
        _refill_threads[key] = thread
        thread.start()
    return thread
