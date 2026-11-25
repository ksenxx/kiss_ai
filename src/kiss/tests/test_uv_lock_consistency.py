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

Three checks, all against the real files in the repository:

* structural: every dependency name any locked package refers to has a
  ``[[package]]`` entry, and every ``[[package]]`` is reachable from the
  workspace roots (an unreachable entry is a leftover from hand editing);
* behavioural: ``uv lock --check --offline`` accepts the lock and finds it
  up to date with ``pyproject.toml`` (skipped when ``uv`` is not installed);
* installable without a compiler: on every Python line ``install.sh`` may
  build the app's environment on, each package ``uv sync`` installs has a
  wheel for the machine.  Seen on 2026-11-24: on a Mac without a Python of
  its own uv 0.13.10 downloads CPython 3.15.2, the lock has no 3.15 wheel
  of ``pydantic-core``, ``pyyaml``, ``aiohttp`` and four others, and
  ``uv sync`` then has to build them, which takes a C and a Rust compiler.
"""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename

from kiss.scripts import check

REPO_ROOT = Path(__file__).resolve().parents[3]
UV_LOCK = REPO_ROOT / "uv.lock"
INSTALL_SH = REPO_ROOT / "install.sh"

# Locked packages that are published as source only.  Both are pure
# Python: uv builds their wheel without a compiler.
SOURCE_ONLY_PACKAGES = {"pysher", "srt"}

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
    """``uv lock --check`` must parse the lock and agree with pyproject.toml.

    A task's worktree shares the main checkout's environment, and its own
    ``.venv`` is a file, there so that uv builds no second one
    (``GitWorktreeOps.block_own_environment``): uv refuses to run in such
    a tree unless it is told the environment, as the ``check`` script
    tells its stages (``check.stage_env``, the inherited environment
    outside a worktree).
    """
    proc = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=check.stage_env(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"uv lock --check failed (exit {proc.returncode}):\n{proc.stderr}"
    )


def _app_python_lines() -> list[tuple[int, int]]:
    """The CPython lines of ``APP_PYTHON_RANGE`` in install.sh (``>=3.A,<3.B``)."""
    match = re.search(
        r'^APP_PYTHON_RANGE=">=3\.(\d+),<3\.(\d+)"$', INSTALL_SH.read_text(), re.MULTILINE
    )
    assert match, "APP_PYTHON_RANGE not found in install.sh"
    return [(3, minor) for minor in range(int(match.group(1)), int(match.group(2)))]


def _platform_tags(system: str, machine: str) -> list[str]:
    """The wheel platform tags an installer accepts on a *system* / *machine*."""
    if system == "Darwin":
        # macOS 13, the oldest the Mac package installs on.
        return list(mac_platforms((13, 0), machine))
    # glibc 2.35 (Ubuntu 22.04) and every older manylinux baseline, under
    # both of its names (manylinux2014 is manylinux_2_17).
    names = [f"manylinux_2_{minor}" for minor in range(35, 4, -1)]
    names += ["manylinux2014", "manylinux2010", "manylinux1"]
    return [f"{name}_{machine}" for name in names]


def _installed_packages(lock: dict, environment: dict[str, str]) -> list[dict]:
    """The locked packages ``uv sync`` installs in *environment*.

    Follows the project's dependencies and its ``dev`` group (the group uv
    installs by default) through the edges whose marker holds there.
    """
    entries: dict[str, list[dict]] = {}
    for pkg in lock["package"]:
        entries.setdefault(pkg["name"], []).append(pkg)
    root = next(pkg for pkg in lock["package"] if "editable" in pkg["source"])
    edges = [*root["dependencies"], *root["dev-dependencies"]["dev"]]
    installed: dict[tuple[str, tuple[str, ...]], dict] = {}
    while edges:
        edge = edges.pop()
        if "marker" in edge and not Marker(edge["marker"]).evaluate(environment):
            continue
        # A name locked at two versions (cryptography) is told apart by the edge.
        (pkg,) = [p for p in entries[edge["name"]] if edge.get("version") in (None, p["version"])]
        extras = tuple(edge.get("extra", ()))
        if (pkg["name"], extras) in installed:
            continue
        installed[pkg["name"], extras] = pkg
        edges += pkg.get("dependencies", [])
        for extra in extras:
            edges += pkg["optional-dependencies"][extra]
    return list(installed.values())


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Darwin", "arm64"), ("Darwin", "x86_64"), ("Linux", "aarch64"), ("Linux", "x86_64")],
)
def test_every_installed_package_has_a_wheel_on_the_app_python_lines(
    system: str, machine: str
) -> None:
    """No dependency has to be compiled on a Python line install.sh may choose."""
    lock = _load_lock()
    platforms = _platform_tags(system, machine)
    lines = _app_python_lines()
    assert lines, "APP_PYTHON_RANGE is empty"
    to_compile = []
    for line in lines:
        version = f"{line[0]}.{line[1]}"
        environment = {
            "implementation_name": "cpython",
            "implementation_version": f"{version}.0",
            "os_name": "posix",
            "platform_machine": machine,
            "platform_python_implementation": "CPython",
            "platform_release": "",
            "platform_system": system,
            "platform_version": "",
            "python_full_version": f"{version}.0",
            "python_version": version,
            "sys_platform": system.lower(),
            "extra": "",
        }
        accepted = {
            *cpython_tags(line, platforms=platforms),
            *compatible_tags(line, f"cp{line[0]}{line[1]}", platforms),
        }
        packages = _installed_packages(lock, environment)
        assert len(packages) > 100, len(packages)
        for pkg in packages:
            if pkg["name"] in SOURCE_ONLY_PACKAGES:
                continue
            wheels = [wheel["url"].rsplit("/", 1)[1] for wheel in pkg.get("wheels", [])]
            if not any(parse_wheel_filename(name)[3] & accepted for name in wheels):
                to_compile.append(f"{pkg['name']} {pkg['version']} on CPython {version}")
    assert not to_compile, (
        f"uv.lock has no {system} {machine} wheel of: {to_compile}. Refresh the lock, or narrow "
        "APP_PYTHON_RANGE in install.sh to the Python lines it has wheels for."
    )
