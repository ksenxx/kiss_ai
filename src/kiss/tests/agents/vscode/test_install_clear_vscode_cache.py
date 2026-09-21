# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` must clear VS Code's on-disk caches before it builds and
installs the extension (steps [4/5] and [5/5]), so the freshly installed
build is never served through stale cached state.

The ``clear_vscode_cache`` function may only sweep the cache directories that
VS Code recreates on its own (``Cache``, ``CachedData``, ``CachedExtensions``,
``CachedExtensionVSIXs``, ``Code Cache``, ``GPUCache``) under the user-data
roots the installer targets (macOS Code, Linux Code, code-server incl. the
``XDG_DATA_HOME`` override) — and must never touch user state (``User/``,
``extensions/``, ``Local Storage``) or abort the install under ``set -e``
when a removal fails.

The tests extract ``clear_vscode_cache`` verbatim from ``install.sh`` and run
it for real in a sandboxed ``$HOME`` under ``set -eo pipefail`` — the same
shell options the install body runs with.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from kiss.tests.conftest import is_root, posix_only

pytestmark = posix_only("install.sh runs under bash with chmod-based removal failures")

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

CACHE_SUBDIRS = [
    "Cache",
    "CachedData",
    "CachedExtensions",
    "CachedExtensionVSIXs",
    "Code Cache",
    "GPUCache",
]


def extract_clear_function() -> str:
    """Return the ``clear_vscode_cache`` function body from install.sh."""
    text = INSTALL_SCRIPT.read_text()
    match = re.search(r"^clear_vscode_cache\(\) \{\n.*?^\}$", text, re.MULTILINE | re.DOTALL)
    assert match, "clear_vscode_cache() not found in install.sh"
    return match.group(0)


def run_clear(home: Path, extra_env: dict[str, str] | None = None) -> str:
    """Run the extracted function with ``$HOME`` sandboxed; return its stdout.

    The harness runs under ``set -eo pipefail`` to prove the function can
    never abort the surrounding install script, whatever it encounters.
    """
    harness = f"set -eo pipefail\n{extract_clear_function()}\nclear_vscode_cache\n"
    env = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"}
    env["HOME"] = str(home)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, timeout=30, env=env
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def make_data_root(root: Path, subdirs: list[str] | None = None) -> None:
    """Create a user-data root populated with cache dirs and user state."""
    for sub in CACHE_SUBDIRS if subdirs is None else subdirs:
        (root / sub).mkdir(parents=True)
        (root / sub / "stale.bin").write_text("stale")
    (root / "User").mkdir(parents=True, exist_ok=True)
    (root / "User" / "settings.json").write_text("{}")
    (root / "extensions").mkdir(exist_ok=True)
    (root / "extensions" / "kiss-sorcar").mkdir()
    (root / "Local Storage").mkdir(exist_ok=True)
    (root / "Local Storage" / "state.db").write_text("state")


def assert_user_state_intact(root: Path) -> None:
    """User settings, installed extensions and Local Storage must survive."""
    assert (root / "User" / "settings.json").read_text() == "{}"
    assert (root / "extensions" / "kiss-sorcar").is_dir()
    assert (root / "Local Storage" / "state.db").read_text() == "state"


def test_clears_all_cache_dirs_in_every_data_root(tmp_path: Path) -> None:
    """All six cache dirs vanish from all three user-data roots; user state stays."""
    roots = [
        tmp_path / "Library" / "Application Support" / "Code",
        tmp_path / ".config" / "Code",
        tmp_path / ".local" / "share" / "code-server",
    ]
    for root in roots:
        make_data_root(root)
    out = run_clear(tmp_path)
    for root in roots:
        for sub in CACHE_SUBDIRS:
            assert not (root / sub).exists(), f"{root / sub} was not cleared"
            assert f"Cleared {root / sub}" in out
        assert_user_state_intact(root)


def test_no_data_roots_reports_nothing_to_clear(tmp_path: Path) -> None:
    """A machine without VS Code state gets the no-op message and rc 0."""
    out = run_clear(tmp_path)
    assert "No VS Code caches found to clear." in out
    assert "Cleared" not in out


def test_data_root_without_caches_reports_nothing_to_clear(tmp_path: Path) -> None:
    """A data root holding only user state is left alone and reported as no-op."""
    root = tmp_path / ".config" / "Code"
    make_data_root(root, subdirs=[])
    out = run_clear(tmp_path)
    assert "No VS Code caches found to clear." in out
    assert_user_state_intact(root)


def test_clears_only_the_cache_dirs_that_exist(tmp_path: Path) -> None:
    """A partial cache set is swept without inventing paths for the rest."""
    root = tmp_path / ".config" / "Code"
    make_data_root(root, subdirs=["Cache", "Code Cache"])
    out = run_clear(tmp_path)
    assert not (root / "Cache").exists()
    assert not (root / "Code Cache").exists()
    assert f"Cleared {root / 'Cache'}" in out
    assert f"Cleared {root / 'Code Cache'}" in out
    assert "CachedData" not in out
    assert_user_state_intact(root)


def test_xdg_data_home_overrides_code_server_root(tmp_path: Path) -> None:
    """With XDG_DATA_HOME set, the code-server sweep follows it exclusively."""
    xdg = tmp_path / "xdg-data"
    xdg_root = xdg / "code-server"
    default_root = tmp_path / ".local" / "share" / "code-server"
    make_data_root(xdg_root, subdirs=["CachedExtensionVSIXs"])
    make_data_root(default_root, subdirs=["CachedExtensionVSIXs"])
    out = run_clear(tmp_path, extra_env={"XDG_DATA_HOME": str(xdg)})
    assert not (xdg_root / "CachedExtensionVSIXs").exists()
    assert f"Cleared {xdg_root / 'CachedExtensionVSIXs'}" in out
    # The default root is not a target when XDG_DATA_HOME points elsewhere.
    assert (default_root / "CachedExtensionVSIXs").is_dir()


@pytest.mark.skipif(is_root(), reason="root ignores directory write permissions")
def test_failed_removal_never_aborts_the_install(tmp_path: Path) -> None:
    """An undeletable cache entry is skipped best-effort under set -e (rc 0)."""
    root = tmp_path / ".config" / "Code"
    make_data_root(root, subdirs=["Cache"])
    locked = root / "Cache" / "locked"
    locked.mkdir()
    (locked / "pinned.bin").write_text("pinned")
    locked.chmod(0o555)  # rm cannot unlink pinned.bin -> rm -rf fails
    try:
        out = run_clear(tmp_path)
        # The failure is reported honestly, never as "Cleared".
        assert f"WARNING: could not fully clear {root / 'Cache'}" in out
        assert f"Cleared {root / 'Cache'}" not in out
        assert (locked / "pinned.bin").exists()  # the undeletable part survived
        assert_user_state_intact(root)
    finally:
        locked.chmod(0o755)


FAKE_NODE = '#!/bin/bash\n[ "$1" = "--version" ] && { echo v22.16.0; exit 0; }\nexit 0\n'
# npm answers --version (step [2/5] check) but fails ``npm ci`` (both the
# first attempt and the clean-node_modules retry), so the real install run
# below stops deterministically INSIDE step [4/5] — after the cache sweep,
# before anything is built or installed.
FAKE_NPM = '#!/bin/bash\n[ "$1" = "--version" ] && { echo 10.9.0; exit 0; }\nexit 1\n'
FAKE_NPX = "#!/bin/bash\nexit 0\n"
FAKE_CODE = '#!/bin/bash\n[ "$1" = "--version" ] && { echo 1.99.0; exit 0; }\nexit 0\n'


def test_install_run_clears_caches_before_the_extension_build_step(tmp_path: Path) -> None:
    """Running install.sh for real clears the caches before step [4/5].

    A sandboxed copy of install.sh runs end-to-end with fake ``node``,
    ``npm``, ``npx`` and ``code`` CLIs on PATH.  The fake ``npm ci`` fails,
    aborting the run inside step [4/5] — proving at runtime that the cache
    sweep already happened before the extension is built (step [4/5]) and
    installed (step [5/5]).
    """
    home = tmp_path / "home"
    home.mkdir()
    root = home / ".config" / "Code"
    make_data_root(root)

    checkout = tmp_path / "checkout"
    (checkout / "src" / "kiss" / "agents" / "vscode").mkdir(parents=True)
    (checkout / "install.sh").write_text(INSTALL_SCRIPT.read_text())

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name, body in [
        ("node", FAKE_NODE),
        ("npm", FAKE_NPM),
        ("npx", FAKE_NPX),
        ("code", FAKE_CODE),
    ]:
        tool = fakebin / name
        tool.write_text(body)
        tool.chmod(0o755)

    env = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"}
    env.update(
        HOME=str(home),
        PATH=f"{fakebin}:{env['PATH']}",
        KISS_NONINTERACTIVE="1",
        KISS_HEARTBEAT_INTERVAL="1",
    )
    proc = subprocess.run(
        ["bash", str(checkout / "install.sh")],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    # The fake ``npm ci`` failure aborts the run inside step [4/5].
    assert proc.returncode != 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert ">>> Clearing VS Code caches..." in out
    assert ">>> [4/5] Building VS Code extension..." in out
    assert out.index(">>> Clearing VS Code caches...") < out.index(
        ">>> [4/5] Building VS Code extension..."
    ), "the cache sweep must run before the [4/5] build step"
    assert ">>> [5/5]" not in out  # the run never reached the install step
    for sub in CACHE_SUBDIRS:
        assert not (root / sub).exists(), f"{root / sub} was not cleared by the real run"
        assert f"Cleared {root / sub}" in out
    assert_user_state_intact(root)
