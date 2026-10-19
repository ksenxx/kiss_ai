# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""install.sh pulls a brand's checkout before running its post-install hook.

A white-label brand registers a hook in ``$KISS_HOME/post-install.d/``
that execs its installer out of a git checkout (s10s: ``exec
/path/s10s/scripts/install.sh --from-kiss-install``).  Step [5/5] of
install.sh calls ``update_brand_checkout`` on every hook before running
it: the checkout the exec'd script sits in is fetched and reset hard to
its upstream, so local edits and stray commits are discarded and the
branding steps run the brand's latest code.  A hook that execs nothing,
a script outside any git work tree, a checkout without an upstream, and
a checkout whose remote cannot be fetched are left alone, with a
warning in the last two cases.  The functions run under bash straight
out of install.sh against real git repositories.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

pytestmark = posix_only("runs install.sh's update_brand_checkout under bash")

_REPO = Path(__file__).resolve().parents[3]
_INSTALL = _REPO / "install.sh"
_FUNCTIONS = re.compile(
    r"^hook_checkout_dir\(\) \{\n.*?^\}\n\nupdate_brand_checkout\(\) \{\n.*?^\}\n",
    re.M | re.S,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
            "HOME": str(cwd.parent),
        },
    ).stdout.strip()


def _commit(repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"add {name}")
    return _git(repo, "rev-parse", "--short", "HEAD")


def _make_origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A bare ``origin`` with one commit and a clone tracking it."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    (seed / "scripts").mkdir()
    _commit(seed, "scripts/install.sh", "#!/bin/bash\necho v1\n")
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(origin)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    return seed, clone


def _installer_text(clone: Path) -> str:
    return (clone / "scripts" / "install.sh").read_text(encoding="utf-8")


def _write_hook(tmp_path: Path, body: str) -> Path:
    hook = tmp_path / "post-install.d" / "brand"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text(body, encoding="utf-8")
    hook.chmod(0o755)
    return hook


def _run(
    tmp_path: Path, func: str, hook: Path, env: dict[str, str] | None = None,
) -> tuple[str, int]:
    """Run ``func <hook>`` as install.sh defines it; return (output, rc)."""
    match = _FUNCTIONS.search(_INSTALL.read_text(encoding="utf-8"))
    assert match, "hook_checkout_dir() + update_brand_checkout() not found in install.sh"
    script = (
        'warn() { echo "   WARNING: $*"; }\n'
        + match.group(0)
        + f"{func} {str(hook)!r}\n"
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    return out.stdout + out.stderr, out.returncode


def _no_identity_env(tmp_path: Path) -> dict[str, str]:
    """An environment in which git can determine no committer identity.

    Mirrors an end user's machine with no ``user.name``/``user.email``
    and an empty passwd name: the global config only disables git's
    name/host auto-detection (``user.useConfigOnly``), so no identity
    comes from the developer's own ``~/.gitconfig`` or the ``GIT_*``
    variables of the test session either.
    """
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("[user]\n\tuseConfigOnly = true\n", encoding="utf-8")
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def test_the_hooks_checkout_is_the_work_tree_of_the_script_it_execs(tmp_path: Path) -> None:
    _seed, clone = _make_origin_and_clone(tmp_path)
    script = clone / "scripts" / "install.sh"
    hook = _write_hook(
        tmp_path, f"#!/bin/bash\n# written by x\nexec {script} --from-kiss-install\n",
    )
    out, rc = _run(tmp_path, "hook_checkout_dir", hook)
    assert rc == 0
    assert Path(out.strip()).resolve() == clone.resolve()
    # A quoted script path, as a path with spaces would be written.
    hook = _write_hook(tmp_path, f'#!/bin/bash\nexec "{script}" --from-kiss-install\n')
    out, _ = _run(tmp_path, "hook_checkout_dir", hook)
    assert Path(out.strip()).resolve() == clone.resolve()


def test_a_checkout_path_with_spaces_is_found(tmp_path: Path) -> None:
    """The brand installer writes the path with ``printf %q`` (backslash
    escapes); a single-quoted path is found too."""
    spaced = tmp_path / "brand checkout"
    _seed, clone = _make_origin_and_clone(tmp_path)
    clone.rename(spaced)
    script = spaced / "scripts" / "install.sh"
    quoted = subprocess.run(
        ["bash", "-c", 'printf %q "$1"', "_", str(script)],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "\\ " in quoted
    for body in (
        f"#!/bin/bash\nexec {quoted} --from-kiss-install\n",
        f"#!/bin/bash\nexec '{script}' --from-kiss-install\n",
    ):
        hook = _write_hook(tmp_path, body)
        out, rc = _run(tmp_path, "hook_checkout_dir", hook)
        assert rc == 0
        assert Path(out.strip()).resolve() == spaced.resolve()


@pytest.mark.parametrize(
    "body",
    [
        "#!/bin/bash\necho hello\n",
        "#!/bin/bash\nexec relative/script.sh\n",
        "#!/bin/bash\nexec {tmp}/nowhere/script.sh\n",
        "#!/bin/bash\nexec {tmp}/plain/script.sh\n",
    ],
)
def test_hooks_without_a_git_checkout_give_nothing(tmp_path: Path, body: str) -> None:
    """No exec line, a relative path, a missing directory, or a script in
    a plain directory: nothing to update."""
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    hook = _write_hook(tmp_path, body.replace("{tmp}", str(tmp_path)))
    out, rc = _run(tmp_path, "hook_checkout_dir", hook)
    assert (out.strip(), rc) == ("", 0)
    out, rc = _run(tmp_path, "update_brand_checkout", hook)
    assert (out.strip(), rc) == ("", 0)


def test_the_checkout_is_rebased_onto_its_upstream(tmp_path: Path) -> None:
    """Upstream commits are pulled; a local commit and a dirty edit ride along."""
    seed, clone = _make_origin_and_clone(tmp_path)
    hook = _write_hook(
        tmp_path, f"#!/bin/bash\nexec {clone / 'scripts' / 'install.sh'} --from-kiss-install\n",
    )
    # Up to date already: reported as such, nothing changes.
    head = _git(clone, "rev-parse", "--short", "HEAD")
    out, rc = _run(tmp_path, "update_brand_checkout", hook)
    assert rc == 0
    assert f"Updating brand checkout {clone} from origin/main" in out
    assert f"is up to date ({head})" in out
    # Upstream moves on; the clone has a local commit and a dirty edit
    # elsewhere: both survive on top of the new upstream.
    new = _commit(seed, "scripts/install.sh", "#!/bin/bash\necho v2\n")
    _git(seed, "push", "-q", str(tmp_path / "origin.git"), "main")
    local = _commit(clone, "local.txt", "mine\n")
    (clone / "notes.txt").write_text("draft\n", encoding="utf-8")
    _git(clone, "add", "notes.txt")
    out, rc = _run(tmp_path, "update_brand_checkout", hook)
    assert rc == 0
    assert "WARNING" not in out
    after = _git(clone, "rev-parse", "--short", "HEAD")
    assert f"updated {local} -> {after}" in out and after != new
    assert _git(clone, "rev-parse", "--short", "HEAD~1") == new
    assert _installer_text(clone) == "#!/bin/bash\necho v2\n"
    assert (clone / "local.txt").read_text(encoding="utf-8") == "mine\n"
    assert (clone / "notes.txt").read_text(encoding="utf-8") == "draft\n"
    assert _git(clone, "status", "--porcelain") == "A  notes.txt"


def test_a_machine_without_a_git_identity_still_rebases(tmp_path: Path) -> None:
    """Replaying the local commit needs a committer; a machine that has
    none configured gets a placeholder for the rebase instead of the
    forced update, and the commit keeps its author."""
    seed, clone = _make_origin_and_clone(tmp_path)
    hook = _write_hook(
        tmp_path, f"#!/bin/bash\nexec {clone / 'scripts' / 'install.sh'} --from-kiss-install\n",
    )
    new = _commit(seed, "scripts/install.sh", "#!/bin/bash\necho v2\n")
    _git(seed, "push", "-q", str(tmp_path / "origin.git"), "main")
    local = _commit(clone, "local.txt", "mine\n")
    env = _no_identity_env(tmp_path)
    assert subprocess.run(
        ["git", "-C", str(clone), "var", "GIT_COMMITTER_IDENT"], env=env, capture_output=True,
    ).returncode != 0
    out, rc = _run(tmp_path, "update_brand_checkout", hook, env)
    assert rc == 0
    assert "WARNING" not in out and "Please tell me who you are" not in out
    after = _git(clone, "rev-parse", "--short", "HEAD")
    assert f"updated {local} -> {after}" in out and after != new
    assert _git(clone, "rev-parse", "--short", "HEAD~1") == new
    assert _git(clone, "log", "-1", "--format=%an <%ae>|%cn <%ce>") == (
        "t <t@x>|kiss-install <kiss-install@localhost>"
    )
    assert (clone / "local.txt").read_text(encoding="utf-8") == "mine\n"
    assert _git(clone, "branch", "--list", "brand-update-backup-*") == ""


def test_a_conflicting_checkout_is_forced_to_upstream_with_a_backup(tmp_path: Path) -> None:
    """A local commit that conflicts with upstream cannot rebase: it is parked
    on a backup branch, the edits in a stash, and the tree reset to upstream."""
    seed, clone = _make_origin_and_clone(tmp_path)
    hook = _write_hook(
        tmp_path, f"#!/bin/bash\nexec {clone / 'scripts' / 'install.sh'} --from-kiss-install\n",
    )
    new = _commit(seed, "scripts/install.sh", "#!/bin/bash\necho v2\n")
    _git(seed, "push", "-q", str(tmp_path / "origin.git"), "main")
    local = _commit(clone, "scripts/install.sh", "#!/bin/bash\necho mine\n")
    (clone / "scratch.txt").write_text("untracked\n", encoding="utf-8")
    out, rc = _run(tmp_path, "update_brand_checkout", hook)
    assert rc == 0
    assert "WARNING: brand checkout" in out and "did not rebase onto origin/main cleanly" in out
    assert f"updated {local} -> {new}" in out
    assert _git(clone, "rev-parse", "--short", "HEAD") == new
    assert _installer_text(clone) == "#!/bin/bash\necho v2\n"
    assert _git(clone, "status", "--porcelain") == ""
    backups = [
        b.strip("* ") for b in _git(clone, "branch", "--list", "brand-update-backup-*").splitlines()
    ]
    assert len(backups) == 1 and backups[0] in out
    assert _git(clone, "rev-parse", "--short", backups[0]) == local
    assert "local edits before the forced update to origin/main" in _git(clone, "stash", "list")
    assert "scratch.txt" in _git(
        clone, "stash", "show", "--include-untracked", "--name-only", "stash@{0}",
    )


def test_a_dirty_edit_that_conflicts_with_upstream_is_stashed(tmp_path: Path) -> None:
    """The commits rebase but the autostashed edit conflicts: git exits 0
    with conflict markers in the tree.  The tree is cleaned (the edit stays
    in the stash) so the hook never runs a file with markers in it."""
    seed, clone = _make_origin_and_clone(tmp_path)
    hook = _write_hook(
        tmp_path, f"#!/bin/bash\nexec {clone / 'scripts' / 'install.sh'} --from-kiss-install\n",
    )
    new = _commit(seed, "scripts/install.sh", "#!/bin/bash\necho v2\n")
    _git(seed, "push", "-q", str(tmp_path / "origin.git"), "main")
    (clone / "scripts" / "install.sh").write_text("#!/bin/bash\necho mine\n", encoding="utf-8")
    out, rc = _run(tmp_path, "update_brand_checkout", hook)
    assert rc == 0
    assert "WARNING: brand checkout" in out and "did not re-apply cleanly" in out
    assert _git(clone, "rev-parse", "--short", "HEAD") == new
    assert _installer_text(clone) == "#!/bin/bash\necho v2\n"
    assert _git(clone, "status", "--porcelain") == ""
    assert "scripts/install.sh" in _git(clone, "stash", "show", "--name-only", "stash@{0}")


def test_no_upstream_or_unreachable_remote_warns_and_keeps_the_code(tmp_path: Path) -> None:
    seed, clone = _make_origin_and_clone(tmp_path)
    hook = _write_hook(
        tmp_path, f"#!/bin/bash\nexec {clone / 'scripts' / 'install.sh'} --from-kiss-install\n",
    )
    head = _git(clone, "rev-parse", "--short", "HEAD")
    # The remote vanishes: the fetch fails, the checkout stays.
    (tmp_path / "origin.git").rename(tmp_path / "gone.git")
    out, rc = _run(tmp_path, "update_brand_checkout", hook)
    assert rc == 0
    assert "WARNING: could not fetch origin/main" in out
    assert _git(clone, "rev-parse", "--short", "HEAD") == head
    # A checkout with no upstream at all: warned, untouched.
    del seed
    _git(clone, "branch", "--unset-upstream")
    out, rc = _run(tmp_path, "update_brand_checkout", hook)
    assert rc == 0
    assert "WARNING: brand checkout" in out and "tracks no upstream branch" in out
    assert _git(clone, "rev-parse", "--short", "HEAD") == head


def test_step_5_updates_every_hooks_checkout_before_running_it() -> None:
    """The hook loop of step [5/5] calls update_brand_checkout on each hook first."""
    text = _INSTALL.read_text(encoding="utf-8")
    loop = re.search(
        r'for hook in "\$KISS_HOME_DIR"/post-install\.d/\*; do\n(.*?)\n    done\n',
        text, re.S,
    )
    assert loop, "post-install hook loop not found"
    body = loop.group(1)
    assert body.index('update_brand_checkout "$hook"') < body.index('"$hook" < /dev/null')
