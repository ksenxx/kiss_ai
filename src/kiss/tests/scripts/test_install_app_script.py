# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` installs KISS Sorcar into ``$KISS_HOME/kiss``: the
library byte-compiled (``.pyc`` only), the files a runtime reads as Python
scripts (the SEAs, the ``/cron`` agent, ``_version.py``) kept as source,
the webapp frontend (``media/``) and the project files uv builds the venv
from, and nothing from the tests or the webapp's JS tooling.

The tests extract the ``kiss-app-files`` block of the script verbatim and
run it for real in a sandbox: against this checkout (git) and against a
small fake checkout without git (the ``find`` fallback), then import the
compiled library and resolve every bundled SEA from the installed copy.
The Python environment is built with the real uv for a project without
dependencies.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only
from kiss.tests.scripts.test_install_script_heartbeat import _extract_signal_helpers

pytestmark = posix_only("install.sh is a bash installer (Darwin/Linux only)")

REPO = Path(__file__).resolve().parents[4]
INSTALL_APP = REPO / "install.sh"
MEDIA = "src/kiss/webapp/media"


def extract_block(name: str) -> str:
    """Return the block of install.sh between ``# BEGIN: <name>`` and ``# END: <name>``."""
    text = INSTALL_APP.read_text()
    match = re.search(
        rf"^# BEGIN: {name}\b.*?\n(.*?)^# END: {name}$", text, re.MULTILINE | re.DOTALL
    )
    assert match, f"{name} block not found in install.sh"
    return match.group(1)


def extract_app_files_block() -> str:
    """Return the ``kiss-app-files`` block of install.sh after the ``kiss-platform`` one it uses."""
    return extract_block("kiss-platform") + extract_block("kiss-app-files")


def run_block(project_dir: Path, home: Path, commands: str) -> subprocess.CompletedProcess[str]:
    """Run *commands* after the extracted block with the installer's variables set.

    ``$APP_DIR/.venv/bin/python`` is this test's interpreter, so
    ``compile_app_library`` writes ``.pyc`` files the test can import.
    """
    app_dir = home / "kiss"
    venv_bin = app_dir / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    python = venv_bin / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    harness = (
        "set -eo pipefail\n"
        f'PROJECT_DIR="{project_dir}"\n'
        f'KISS_HOME_DIR="{home}"\n'
        f'APP_DIR="{app_dir}"\n'
        f"{extract_app_files_block()}\n"
        f"{commands}\n"
    )
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, cwd=str(project_dir), check=False
    )


def installed_python_files(app_dir: Path) -> tuple[set[str], set[str]]:
    """Return the app-relative ``.py`` and ``.pyc`` paths under ``src``."""
    src = app_dir / "src"
    py = {str(p.relative_to(app_dir)) for p in src.rglob("*.py")}
    pyc = {str(p.relative_to(app_dir)) for p in src.rglob("*.pyc")}
    return py, pyc


@pytest.fixture(scope="module")
def installed_app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The checkout copied and byte-compiled into ``<home>/kiss``, once per module."""
    home = tmp_path_factory.mktemp("kiss-home")
    result = run_block(REPO, home, "copy_app_files\ncopy_brand_overlay\ncompile_app_library")
    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"Copied \d+ files to ", result.stdout), result.stdout
    assert re.search(r"Byte-compiled \d+ modules; \d+ Python scripts kept as source", result.stdout)
    return home / "kiss"


def test_bash_syntax_and_executable() -> None:
    """The script parses and is executable like install.sh."""
    assert os.access(INSTALL_APP, os.X_OK)
    subprocess.run(["bash", "-n", str(INSTALL_APP)], check=True)


def test_project_files_and_frontend_installed(installed_app: Path) -> None:
    """The venv inputs and the webapp frontend are in the app directory."""
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        assert (installed_app / name).is_file(), name
    for name in ("chat.html", "main.js", "main.css", "brand.json", "kiss-icon.svg"):
        assert (installed_app / MEDIA / name).is_file(), name
    # Data files the runtime reads next to its modules travel too.
    assert (installed_app / "src/kiss/SYSTEM.md").is_file()
    assert (installed_app / "src/kiss/core/models/MODEL_INFO.json").is_file()
    # The checkout's commit hash, shown next to the version in Settings.
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    assert head
    assert (installed_app / "src/kiss/core/_commit.txt").read_text().strip() == head


def test_tests_and_webapp_tooling_excluded(installed_app: Path) -> None:
    """No test files, and nothing of the webapp but its media/ frontend."""
    assert not (installed_app / "src/kiss/tests").exists()
    webapp = installed_app / "src/kiss/webapp"
    assert sorted(p.name for p in webapp.iterdir()) == ["media"]


def test_library_is_byte_compiled_and_scripts_kept(installed_app: Path) -> None:
    """Modules are ``.pyc`` only; path-loaded scripts stay ``.py`` with no ``.pyc`` twin."""
    py, pyc = installed_python_files(installed_app)
    assert not list((installed_app / "src").rglob("__pycache__"))
    for module in (
        "src/kiss/server/web_server",
        "src/kiss/agents/sorcar/agent_dispatch",
        "src/kiss/agents/sorcar/sea_commands",
        "src/kiss/agents/third_party_agents/slack/__init__",
        "src/kiss/__init__",
    ):
        assert module + ".pyc" in pyc, module
        assert module + ".py" not in py, module
    for script in (
        "src/kiss/agents/seas/sorcar/sorcar_sea.py",
        "src/kiss/agents/seas/__init__.py",
        "src/kiss/agents/third_party_agents/slack/slack_sea.py",
        "src/kiss/agents/sorcar/cron_agent.py",
        "src/kiss/agents/sorcar/cron_prompt_sea.py",
        "src/kiss/agents/sorcar/chat_summary.py",
        "src/kiss/agents/sorcar/history_columns.py",
        "src/kiss/scripts/sync_db.py",
        "src/kiss/scripts/carry_over_tables.py",
        "src/kiss/scripts/merge_memory_pages.py",
        "src/kiss/core/_version.py",
    ):
        assert script in py, script
        assert script[:-3] + ".pyc" not in pyc, script
    # Every remaining .py is one of the kept kinds, nothing else slipped through.
    for path in py:
        assert (
            path.startswith("src/kiss/agents/seas/")
            or path.endswith("_sea.py")
            or path
            in {
                "src/kiss/agents/sorcar/cron_agent.py",
                "src/kiss/agents/sorcar/chat_summary.py",
                "src/kiss/agents/sorcar/history_columns.py",
                "src/kiss/scripts/sync_db.py",
                "src/kiss/scripts/carry_over_tables.py",
                "src/kiss/scripts/merge_memory_pages.py",
                "src/kiss/core/_version.py",
            }
        ), path


def test_compiled_library_imports_and_resolves_seas(installed_app: Path) -> None:
    """The installed copy imports from ``.pyc`` and every bundled SEA loads from its ``.py``."""
    probe = """
from pathlib import Path
import kiss, kiss.server.web_server as w
from kiss.agents.sorcar.sea_commands import bundled_commands, load_sea, sea_settings
assert w.__file__.endswith("web_server.pyc"), w.__file__
assert kiss.__file__.endswith("__init__.pyc"), kiss.__file__
commands = bundled_commands()
assert len(commands) > 50, commands
for name, path in sorted(commands.items()):
    assert path.suffix == ".py" and path.is_file(), (name, path)
    load_sea(path)
    sea_settings(path)
assert commands["cron"].name == "cron_agent.py"
assert commands["slack"].name == "slack_sea.py"
# The Sorcar SEA itself is hidden from the command list but dispatched by path.
from kiss.agents.sorcar import agent_dispatch
agents = Path(agent_dispatch.__file__).resolve().parents[1]
load_sea(agents / "seas" / "sorcar" / "sorcar_sea.py")
# sync_db ships its own source and the SHIPPED_MODULES to the remote: all must be Python text.
from kiss.scripts import sync_db
shipped = [sync_db._shipped_module_path(name) for name in sync_db.SHIPPED_MODULES]
for source_path in (sync_db._script_path(), *shipped):
    compile(sync_db._read_source(source_path), source_path, "exec")
# /sync_history_and_memory ships memory_delta.py and merge_memory_pages.py the same way.
from kiss.agents.seas.sync_history_and_memory import memory_delta, peer_sync
for source_path in (peer_sync.MEMORY_DELTA_PATH, peer_sync.MERGE_PAGES_PATH):
    assert source_path.suffix == ".py", source_path
    compile(source_path.read_bytes(), str(source_path), "exec")
assert memory_delta.merge_module().TOMBSTONES_DIR == ".tombstones"
# The Settings header shows the version and, in parentheses, the installed commit.
commit = (Path(w.__file__).parent.parent / "core" / "_commit.txt").read_text().strip()
assert w._read_commit() == commit != ""
assert f"Configuration {w._read_version()} ({commit})<" in w._build_html()
# A saved chat page records both in its last line.
saved = f"version {w._read_version()}, commit {commit}</footer>"
assert saved in w._build_share_page("t", "<p>x</p>")
# Without the file (a tarball install, outside any git work tree) only the version shows,
# and a saved page records the commit as unknown.
# (git missing or hanging, the except branch of _read_commit, is not reproducible here.)
commit_file = Path(w.__file__).parent.parent / "core" / "_commit.txt"
commit_file.rename(commit_file.with_suffix(".bak"))
try:
    assert w._read_commit() == ""
    assert f"Configuration {w._read_version()}<" in w._build_html()
    saved = f"version {w._read_version()}, commit unknown</footer>"
    assert saved in w._build_share_page("t", "<p>x</p>")
finally:
    commit_file.with_suffix(".bak").rename(commit_file)
# An install whose _version.py is gone records the version as unknown too.
version = w._read_version()
version_file = Path(w.__file__).parent.parent / "core" / "_version.py"
version_file.rename(version_file.with_suffix(".bak"))
try:
    saved = f"version unknown, commit {commit}</footer>"
    assert saved in w._build_share_page("t", "<p>x</p>")
finally:
    version_file.with_suffix(".bak").rename(version_file)
assert w._read_version() == version != ""
print("ok", len(commands), w._read_version())
"""
    env = dict(os.environ, PYTHONPATH=str(installed_app / "src"))
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(installed_app),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok "), result.stdout


def fake_checkout(root: Path) -> Path:
    """A minimal checkout without git: the ``find`` fallback of list_app_files."""
    checkout = root / "checkout"
    files = {
        "pyproject.toml": "[project]\nname='x'\n",
        "src/kiss/__init__.py": "",
        "src/kiss/core/mod.py": "VALUE = 1\n",
        "src/kiss/core/_commit.txt": "abc1234\n",
        "src/kiss/core/__pycache__/mod.cpython-313.pyc": "stale",
        "src/kiss/core/stale.pyc": "stale",
        "src/kiss/core/.DS_Store": "junk",
        "src/kiss/agents/seas/demo/demo_sea.py": "SEA = True\n",
        "src/kiss/tests/test_mod.py": "def test(): pass\n",
        f"{MEDIA}/chat.html": "<html></html>",
        f"{MEDIA}/brand.css": "stock",
        "src/kiss/webapp/package.json": "{}",
        "src/kiss/webapp/test/chat.test.js": "export {}",
        "src/kiss/webapp/node_modules/x/index.js": "",
        ".brand/brand.css": "branded",
    }
    for rel, text in files.items():
        path = checkout / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return checkout


@pytest.mark.parametrize("inside_another_repository", [False, True])
def test_find_fallback_without_git(tmp_path: Path, inside_another_repository: bool) -> None:
    """A tree without git history is copied by ``find`` with the same exclusions.

    Also when it lies inside some other repository, as the Mac package's
    payload does in a home directory kept under version control: git
    answers there, and lists none of the tree's files.
    """
    if inside_another_repository:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    checkout = fake_checkout(tmp_path)
    home = tmp_path / "home"
    result = run_block(checkout, home, "copy_app_files\ncopy_brand_overlay\ncompile_app_library")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Copied 7 files to " in result.stdout, result.stdout
    assert "Byte-compiled 2 modules; 1 Python scripts kept as source" in result.stdout
    app = home / "kiss"
    installed = sorted(
        str(p.relative_to(app)) for p in app.rglob("*") if p.is_file() and ".venv" not in p.parts
    )
    assert installed == [
        "pyproject.toml",
        "src/kiss/__init__.pyc",
        "src/kiss/agents/seas/demo/demo_sea.py",
        "src/kiss/core/_commit.txt",
        "src/kiss/core/mod.pyc",
        f"{MEDIA}/brand.css",
        f"{MEDIA}/chat.html",
    ]
    assert (app / MEDIA / "brand.css").read_text() == "branded"
    assert "src/kiss/webapp/media/brand.css" in result.stdout
    # The commit such a tree ships (the Mac package's payload has one) is
    # the one the Settings panel shows.
    assert (app / "src/kiss/core/_commit.txt").read_text() == "abc1234\n"


def test_rerun_replaces_stale_files(tmp_path: Path) -> None:
    """A second copy drops files the checkout no longer has and keeps the venv."""
    checkout = fake_checkout(tmp_path)
    home = tmp_path / "home"
    assert run_block(checkout, home, "copy_app_files").returncode == 0
    (checkout / "src/kiss/core/mod.py").unlink()
    (home / "kiss" / ".venv" / "marker").write_text("keep")
    result = run_block(checkout, home, "copy_app_files")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (home / "kiss/src/kiss/core/mod.py").exists()
    assert (home / "kiss/.venv/marker").read_text() == "keep"


def test_empty_checkout_is_an_error(tmp_path: Path) -> None:
    """Nothing to install fails the step instead of producing an empty app."""
    checkout = tmp_path / "checkout"
    (checkout / "src/kiss").mkdir(parents=True)
    result = run_block(checkout, tmp_path / "home", "copy_app_files")
    assert result.returncode != 0
    assert "ERROR: no files to install found" in result.stdout


def test_compile_failure_is_reported(tmp_path: Path) -> None:
    """A module that does not compile fails the step (the source is left in place)."""
    checkout = fake_checkout(tmp_path)
    (checkout / "src/kiss/core/broken.py").write_text("def (:\n")
    home = tmp_path / "home"
    result = run_block(checkout, home, "copy_app_files\ncompile_app_library")
    assert result.returncode != 0
    assert (home / "kiss/src/kiss/core/broken.py").exists()


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_venv_is_built_on_a_python_line_of_the_range(tmp_path: Path) -> None:
    """A first build asks uv for the Python lines of the range; later builds keep the line it got.

    uv is the real one, started through a script that first appends the
    ``.python-version`` it is about to read to a file.  The environment
    holds no ``UV_*`` variable: one naming a project environment would
    make uv build this project's environment there.
    """
    app = tmp_path / "home" / "kiss"
    app.mkdir(parents=True)
    (app / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1"\nrequires-python = ">=3.13"\n'
    )
    requests = tmp_path / "requests.txt"
    uv = tmp_path / "uv"
    uv.write_text(f'#!/bin/bash\ncat "{app}/.python-version" >> "{requests}"\nexec uv "$@"\n')
    uv.chmod(0o755)
    harness = (
        "set -eo pipefail\n"
        f"{_extract_signal_helpers(INSTALL_APP)}\n"
        f'APP_DIR="{app}"\n'
        f"{extract_app_files_block()}\n"
        f'build_app_venv "{uv}"\n'
    )
    env = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "KISS_HEARTBEAT_INTERVAL": "60"}
    for _ in ("first install", "update"):
        result = subprocess.run(
            ["bash", "-c", harness], capture_output=True, text=True, env=env, cwd=tmp_path
        )
        assert result.returncode == 0, result.stdout + result.stderr
    line = subprocess.run(
        [str(app / ".venv/bin/python"), "-c", 'import sys; print("%d.%d" % sys.version_info[:2])'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert (app / ".python-version").read_text() == line
    first, update = requests.read_text().splitlines()
    assert update == line.strip()
    bounds = re.fullmatch(r">=3\.(\d+),<3\.(\d+)", first)
    assert bounds, first
    assert line.strip() in [f"3.{minor}" for minor in range(int(bounds[1]), int(bounds[2]))]
