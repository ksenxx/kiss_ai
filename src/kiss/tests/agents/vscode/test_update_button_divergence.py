# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: the Update button must not fail on a diverged ``main``.

Background — the bug
====================
When users press *Update* in the settings panel the extension spawns a
terminal that runs ``bash install.sh``.  Before
commit ``993a997b`` ("fix: handle diverged branches when updating
repositories") the script's ``update_repo`` step did a plain
``git pull`` and crashed with::

    fatal: Not possible to fast-forward, aborting.
    hint: You have divergent branches and need to specify how to
          reconcile them.
       WARNING: git pull failed (offline or diverged); continuing
       with the current checkout.

This left the user stuck on the stale checkout — the rebuild then
reinstalled the *same* stale source forever.  Two independent
mechanisms must therefore guarantee that *Update* always converges to
``origin`` even after a force-push, even on machines whose extension
shipped with the OLD ``install.sh``:

1. The current ``install.sh``'s :func:`update_repo` (running in shell
   under ``set -eo pipefail``) detects fast-forward failure and falls
   back to ``git reset --hard '@{upstream}'``.
2. The *VS Code* extension's ``runUpdate`` (``SorcarSidebarView.ts``)
   runs a pre-flight ``git fetch``/``stash``/``reset --hard`` *before*
   invoking ``bash install.sh``, so a stale ``install.sh`` is replaced
   on disk *before* bash reads it (chicken-and-egg fix).

Both paths are exercised end-to-end here against a real ``git`` and a
real bare upstream that has been force-pushed (no mocks, no
monkeypatching) — exactly the failure mode reported by users.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SH = REPO / "install.sh"
SIDEBAR_TS = REPO / "src" / "kiss" / "agents" / "vscode" / "src" / "SorcarSidebarView.ts"

# A deterministic env that pins git's identity and disables any user-level
# config so the hermetic repo is unaffected by ``~/.gitconfig`` (which on a
# developer machine could set ``pull.rebase`` etc. and mask the bug).
HERMETIC_GIT_ENV = {
    "GIT_AUTHOR_NAME": "tester",
    "GIT_AUTHOR_EMAIL": "tester@example.com",
    "GIT_COMMITTER_NAME": "tester",
    "GIT_COMMITTER_EMAIL": "tester@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
}


def _git_env() -> dict[str, str]:
    """Return ``os.environ`` overlaid with the hermetic git settings."""
    env = dict(os.environ)
    env.update(HERMETIC_GIT_ENV)
    return env


def _run(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run *cmd* in *cwd* capturing stdout+stderr, with a 60 s timeout."""
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env if env is not None else _git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check,
        timeout=60,
    )


def _make_diverged_clone(
    root: Path,
    *,
    seed_install_sh: str,
    upstream_install_sh: str,
    add_local_commit: bool = True,
) -> tuple[Path, Path]:
    """Build a bare upstream that diverges from a local clone.

    Sequence of operations:

      1. Init a bare remote at ``root/remote.git``.
      2. Init a seed worktree, write ``install.sh`` = *seed_install_sh*,
         commit and push to ``origin/main``.
      3. Clone the remote into ``root/local`` (so ``local`` has the same
         commit as ``origin/main`` initially).
      4. In a second worktree, rewrite ``install.sh`` to
         *upstream_install_sh*, ``--amend`` the commit (rewrites SHA) and
         force-push — ``origin/main`` is now an unrelated tip.
      5. If *add_local_commit* is True, append a new commit to ``local``
         so the local branch also has a unique tip — this is what makes
         ``git pull --ff-only`` reject with "Not possible to fast-forward".

    Returns ``(remote, local)``.
    """
    remote = root / "remote.git"
    seed = root / "seed"
    pusher = root / "pusher"
    local = root / "local"

    _run(["git", "init", "--bare", "-b", "main", str(remote)], cwd=root)

    seed.mkdir()
    _run(["git", "init", "-b", "main"], cwd=seed)
    (seed / "install.sh").write_text(seed_install_sh, encoding="utf-8")
    _run(["git", "add", "install.sh"], cwd=seed)
    _run(["git", "commit", "-m", "seed"], cwd=seed)
    _run(["git", "remote", "add", "origin", str(remote)], cwd=seed)
    _run(["git", "push", "origin", "main"], cwd=seed)

    _run(["git", "clone", str(remote), str(local)], cwd=root)
    # Pin identity inside the local clone so commits there don't pick up
    # the developer machine's ``~/.gitconfig``.
    _run(["git", "config", "user.name", "tester"], cwd=local)
    _run(["git", "config", "user.email", "tester@example.com"], cwd=local)

    # Rewrite origin/main to a different tip with a different install.sh.
    _run(["git", "clone", str(remote), str(pusher)], cwd=root)
    _run(["git", "config", "user.name", "tester"], cwd=pusher)
    _run(["git", "config", "user.email", "tester@example.com"], cwd=pusher)
    (pusher / "install.sh").write_text(upstream_install_sh, encoding="utf-8")
    _run(["git", "commit", "-am", "rewrite upstream install.sh"], cwd=pusher)
    # ``--amend`` rewrites SHA, then force-push so origin/main is the new tip.
    _run(["git", "commit", "--amend", "--no-edit"], cwd=pusher)
    _run(["git", "push", "--force", "origin", "main"], cwd=pusher)

    if add_local_commit:
        (local / "LOCAL_ONLY.txt").write_text("local change\n", encoding="utf-8")
        _run(["git", "add", "LOCAL_ONLY.txt"], cwd=local)
        _run(["git", "commit", "-m", "local-only commit"], cwd=local)

    return remote, local


def _extract_function(install_sh_text: str, name: str) -> str:
    """Return the source of bash function *name* from *install_sh_text*.

    ``install.sh`` uses top-level function definitions formatted as
    ``name() {`` on one line and ``}`` alone at column 0 on the closing
    line.  An ``awk`` range pattern on this convention is the same way
    the harness shell extracts them in :func:`_run_update_repo`.
    """
    proc = subprocess.run(
        ["awk", f"/^{name}\\(\\) {{/,/^}}$/"],
        input=install_sh_text,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        timeout=10,
    )
    body = proc.stdout
    assert f"{name}() {{" in body, f"could not extract {name}() from install.sh"
    return body


def _run_update_repo(local: Path) -> subprocess.CompletedProcess:
    """Execute ``install.sh``'s ``update_repo`` against the *local* clone.

    The function is sliced out of the current ``install.sh`` and sourced
    in a fresh bash so this test fails exactly when the file's
    divergence-handling regresses — no copy of the logic lives in the
    test itself.
    """
    install_text = INSTALL_SH.read_text(encoding="utf-8")
    update_fn = _extract_function(install_text, "update_repo")
    restore_fn = _extract_function(install_text, "restore_stashed_changes")
    harness = textwrap.dedent(
        f"""
        set -eo pipefail
        PROJECT_DIR={local!s}
        STASHED_CHANGES=0
        {restore_fn}
        {update_fn}
        update_repo
        """
    )
    return subprocess.run(
        ["bash", "-c", harness],
        env=_git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )


def _head_sha(repo: Path, ref: str = "HEAD") -> str:
    """Return the commit SHA of *ref* in *repo*."""
    return str(_run(["git", "rev-parse", ref], cwd=repo).stdout).strip()


# ---------------------------------------------------------------------------
# Test 1 — install.sh recovers from divergence
# ---------------------------------------------------------------------------


def test_update_repo_recovers_from_diverged_upstream(tmp_path: Path) -> None:
    """``update_repo`` must reset to upstream when ``--ff-only`` fails.

    Reproduces the user's report: ``git pull`` returns "Not possible to
    fast-forward, aborting." because both sides have unique commits.
    The fix must (a) print the diverged-resetting banner, (b) succeed
    (returncode 0), and (c) leave ``HEAD`` pointing at the new
    ``origin/main`` tip.
    """
    _, local = _make_diverged_clone(
        tmp_path,
        seed_install_sh="echo SEED\n",
        upstream_install_sh="echo NEW_UPSTREAM\n",
    )
    # Sanity: a plain ``git pull --ff-only`` reproduces the original bug.
    diverged = subprocess.run(
        ["git", "-C", str(local), "pull", "--ff-only"],
        env=_git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=30,
    )
    assert diverged.returncode != 0, (
        "test setup did not actually create a diverged state:\n"
        f"{diverged.stdout}"
    )

    result = _run_update_repo(local)
    assert result.returncode == 0, result.stdout
    assert "Branches diverged" in result.stdout, result.stdout
    # Upstream tip and local HEAD must agree.
    upstream_sha = _head_sha(local, "origin/main")
    head_sha = _head_sha(local, "HEAD")
    assert head_sha == upstream_sha, (
        f"HEAD ({head_sha}) did not reset to origin/main ({upstream_sha}) "
        f"after update_repo:\n{result.stdout}"
    )
    # The new install.sh content must be on disk.
    assert "NEW_UPSTREAM" in (local / "install.sh").read_text(encoding="utf-8")


def test_update_repo_preserves_dirty_working_tree(tmp_path: Path) -> None:
    """A dirty working tree is stashed before reset and popped after.

    Otherwise the user's uncommitted edits would silently vanish whenever
    the divergence path runs — the exact failure mode that made the
    earlier "just always reset" patch unacceptable.
    """
    _, local = _make_diverged_clone(
        tmp_path,
        seed_install_sh="echo SEED\n",
        upstream_install_sh="echo NEW_UPSTREAM\n",
    )
    dirty_path = local / "user_edit.txt"
    dirty_path.write_text("user's uncommitted work\n", encoding="utf-8")

    result = _run_update_repo(local)
    assert result.returncode == 0, result.stdout
    assert "stashing local changes" in result.stdout, result.stdout
    assert "Restoring stashed local changes" in result.stdout, result.stdout
    assert dirty_path.read_text(encoding="utf-8") == "user's uncommitted work\n"


# ---------------------------------------------------------------------------
# Test 3 — runUpdate's pre-flight refreshes a stale install.sh on disk
# ---------------------------------------------------------------------------


def _extract_runupdate_preflight(ts_text: str, esc_dir: str, esc_script: str) -> str:
    """Reconstruct the shell string that ``runUpdate`` sends to the terminal.

    Reads the ``const preflight = [...]`` array literal directly from
    ``SorcarSidebarView.ts``, evaluates each JS string element with the
    same template substitutions ``runUpdate`` performs at runtime
    (``${escDir}`` and ``${escScript}``), and joins with ``'; '`` — same
    as the ``.join('; ')`` call in the TypeScript.  Drives the test
    directly off the production source so a future edit to the array is
    re-tested rather than silently bypassed.
    """
    start = ts_text.index("const preflight = [")
    end = ts_text.index("].join('; ');", start)
    body = ts_text[start + len("const preflight = [") : end]
    parts: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.endswith(","):
            line = line[:-1].rstrip()
        if not line:
            continue
        quote = line[0]
        assert line[-1] == quote, f"unterminated string literal: {raw!r}"
        inner = line[1:-1]
        if quote == "`":
            inner = inner.replace("${escDir}", esc_dir).replace(
                "${escScript}", esc_script
            )
        elif quote == '"':
            # JS double-quoted strings allow ``\"`` and ``\\``.
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        elif quote == "'":
            # JS single-quoted strings use no escapes in this file; pass through.
            pass
        else:
            pytest.fail(f"unexpected quoting in preflight array: {raw!r}")
        parts.append(inner)
    assert parts, "preflight array parse returned no entries"
    return "; ".join(parts)


def test_runupdate_preflight_refreshes_stale_install_sh(tmp_path: Path) -> None:
    """The Update button must overwrite a stale install.sh BEFORE bash reads it.

    Chicken-and-egg scenario: the user is on an extension whose
    ``install.sh`` on disk predates the divergence-handling fix.  If
    ``runUpdate`` simply called ``bash install.sh`` the stale script
    would swallow the divergence and never converge.  ``runUpdate``
    therefore runs a pre-flight ``git fetch`` + ``reset --hard`` first,
    so the *fresh* install.sh from ``origin`` is on disk *before* bash
    starts reading it.

    The test wires the on-disk install.sh to print ``STALE_MARKER`` and
    the upstream install.sh to print ``FRESH_MARKER``.  After the
    pre-flight, bash must execute the fresh one — i.e. ``FRESH_MARKER``
    appears and ``STALE_MARKER`` does NOT.
    """
    stale = "#!/bin/bash\necho STALE_MARKER\nexit 0\n"
    fresh = "#!/bin/bash\necho FRESH_MARKER\nexit 0\n"
    _, local = _make_diverged_clone(
        tmp_path,
        seed_install_sh=stale,
        upstream_install_sh=fresh,
    )
    # The on-disk install.sh in ``local`` is currently the OLD one (the
    # initial seed commit), because the local-only commit added an
    # unrelated file rather than touching install.sh.  Confirm before
    # running the pre-flight.
    on_disk = (local / "install.sh").read_text(encoding="utf-8")
    assert "STALE_MARKER" in on_disk, on_disk

    script_path = local / "install.sh"
    esc_dir = str(local).replace("'", "'\\''")
    esc_script = str(script_path).replace("'", "'\\''")
    preflight = _extract_runupdate_preflight(
        SIDEBAR_TS.read_text(encoding="utf-8"),
        esc_dir=esc_dir,
        esc_script=esc_script,
    )

    result = subprocess.run(
        ["bash", "-c", preflight],
        env=_git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "FRESH_MARKER" in result.stdout, (
        "runUpdate's pre-flight failed to refresh the stale install.sh — "
        f"bash executed the old one:\n{result.stdout}"
    )
    assert "STALE_MARKER" not in result.stdout, (
        "stale install.sh was executed even though the pre-flight should "
        f"have replaced it first:\n{result.stdout}"
    )
    # The disk content must now reflect the upstream tip.
    assert "FRESH_MARKER" in (local / "install.sh").read_text(encoding="utf-8")


def test_runupdate_preflight_preserves_user_edits(tmp_path: Path) -> None:
    """Pre-flight ``reset --hard`` must not lose dirty / untracked edits.

    The stash/pop wrapper inside the preflight is the only thing keeping
    a user's uncommitted work alive across the hard reset.  Regressions
    here would silently delete user files on every Update click.
    """
    stale = "#!/bin/bash\necho STALE_MARKER\nexit 0\n"
    fresh = "#!/bin/bash\necho FRESH_MARKER\nexit 0\n"
    _, local = _make_diverged_clone(
        tmp_path,
        seed_install_sh=stale,
        upstream_install_sh=fresh,
    )
    # An untracked file the user has authored locally — typical case.
    user_file = local / "my_notes.txt"
    user_file.write_text("important user notes\n", encoding="utf-8")

    script_path = local / "install.sh"
    esc_dir = str(local).replace("'", "'\\''")
    esc_script = str(script_path).replace("'", "'\\''")
    preflight = _extract_runupdate_preflight(
        SIDEBAR_TS.read_text(encoding="utf-8"),
        esc_dir=esc_dir,
        esc_script=esc_script,
    )

    result = subprocess.run(
        ["bash", "-c", preflight],
        env=_git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "FRESH_MARKER" in result.stdout, result.stdout
    assert user_file.exists(), (
        "pre-flight reset deleted the user's untracked file:\n"
        f"{result.stdout}"
    )
    assert user_file.read_text(encoding="utf-8") == "important user notes\n"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
