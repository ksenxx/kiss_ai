# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end guard that the committed ``uv.lock`` is one uv can install from.

Every install path (``install.sh``, the VS Code extension's
``DependencyInstaller.ts``, the Dockerfile) copies ``uv.lock`` next to
``pyproject.toml`` and runs ``uv sync`` / ``uv run`` against it.  A lock uv
cannot parse therefore breaks every Sorcar surface at once: on 2026-09-26 a
hand-edited lock dropped the ``[[package]]`` entry for ``websocket-client``
while ``pysher`` (pulled in by ``composio``) still listed it as a dependency,
so ``uv sync`` failed with "Failed to parse uv.lock" and no task could run.

Two checks, both against the real files in the repository:

* structural: every dependency name any locked package refers to has a
  ``[[package]]`` entry, and every ``[[package]]`` is reachable from the
  workspace roots (an unreachable entry is a leftover from hand editing);
* behavioural: ``uv lock --check --offline`` accepts the lock and finds it
  up to date with ``pyproject.toml`` (skipped when ``uv`` is not installed).
"""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
UV_LOCK = REPO_ROOT / "uv.lock"

# Tables of a ``[[package]]`` entry that list other packages by name.
_DEP_TABLES = ("dependencies", "dev-dependencies", "optional-dependencies")


def _load_lock() -> dict:
    with UV_LOCK.open("rb") as fh:
        return tomllib.load(fh)


def _dependency_names(pkg: dict) -> set[str]:
    """Return every package name ``pkg`` depends on, across all dep tables."""
    names: set[str] = set()
    for table in _DEP_TABLES:
        value = pkg.get(table)
        if value is None:
            continue
        # ``optional-dependencies`` / ``dev-dependencies`` map a group name
        # to a list of requirements; ``dependencies`` is a flat list.
        groups = value.values() if isinstance(value, dict) else [value]
        for reqs in groups:
            names.update(req["name"] for req in reqs)
    return names


def test_every_locked_dependency_has_a_package_entry() -> None:
    """Each name referenced from a ``dependencies`` table must be locked."""
    lock = _load_lock()
    packages = lock["package"]
    locked = {pkg["name"] for pkg in packages}
    missing = {
        (pkg["name"], dep)
        for pkg in packages
        for dep in _dependency_names(pkg)
        if dep not in locked
    }
    assert not missing, (
        "uv.lock references packages that have no [[package]] entry "
        f"(dependant, missing): {sorted(missing)}"
    )


def test_every_locked_package_is_reachable_from_the_workspace() -> None:
    """No orphan ``[[package]]`` entries: each one must be required by a root
    or, transitively, by something a root requires."""
    lock = _load_lock()
    # A universal lock may pin several versions of one name (e.g.
    # cryptography 48.x for Intel macOS and 50.x elsewhere), so merge the
    # dependency names of every entry that shares a name.
    deps_by_name: dict[str, set[str]] = {}
    roots: set[str] = set()
    for pkg in lock["package"]:
        deps_by_name.setdefault(pkg["name"], set()).update(_dependency_names(pkg))
        if "editable" in pkg.get("source", {}) or "virtual" in pkg.get("source", {}):
            roots.add(pkg["name"])
    assert roots, "uv.lock has no workspace root package"
    reachable: set[str] = set()
    frontier = list(roots)
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        # Names without an entry are reported by the test above.
        frontier.extend(dep for dep in deps_by_name[name] if dep in deps_by_name)
    orphans = sorted(set(deps_by_name) - reachable)
    assert not orphans, f"uv.lock contains packages nothing depends on: {orphans}"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_uv_accepts_the_lock_and_finds_it_up_to_date() -> None:
    """``uv lock --check`` must parse the lock and agree with pyproject.toml."""
    proc = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"uv lock --check failed (exit {proc.returncode}):\n{proc.stderr}"
    )
