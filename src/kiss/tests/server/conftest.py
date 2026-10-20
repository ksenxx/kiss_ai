# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared fixtures for the kiss.server test suite.

These tests exercise the real sqlite persistence layer
(:mod:`kiss.agents.sorcar.persistence`).  When they run in the same
process AFTER tests from other suites that swap or delete the database
file at ``persistence._DB_PATH``, a stale per-thread connection (or a
still-running background event-writer thread) left behind by those
tests can keep writing into the orphaned old file while these tests
read a fresh, empty database — producing order-dependent failures such
as ``sqlite3.OperationalError: no such table: task_history`` or rows
that silently never appear in ``_get_history()``.

Many tests relocated here from the former ``tests/vscode/`` directory
relied on its identical autouse reset, and the same order-dependence
protection is appropriate for every server test.

The orphan-sweep join guard for ``AgentServer``'s ``orphan-task-sweep``
threads lives in the ROOT ``tests/conftest.py`` and protects this
directory too; see its module docstring for the mechanism.

The module also holds the helpers that most server tests need (a free
port, a permissive TLS client context, a scratch history database, a
seeded git repository, ...); test files import them with
``from kiss.tests.server.conftest import ...``.
"""

from __future__ import annotations

import contextlib
import os
import ssl
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml

from kiss.agents.sorcar import persistence
from kiss.server import agent_state
from kiss.tests.conftest import find_free_port as find_free_port

if TYPE_CHECKING:
    from kiss.server.server import AgentServer

PersistenceState = tuple[Path, Any, Path]

# Upgrade-request headers that make a test client on 127.0.0.1 look like a
# visitor relayed by the local cloudflared tunnel: the daemon admits a DIRECT
# loopback peer without a password, so tests of the password handshake pass
# these as ``additional_headers`` to ``websockets.connect``.
FORWARDED_VISITOR = {"X-Forwarded-For": "203.0.113.9"}


def no_verify_ssl() -> ssl.SSLContext:
    """Return a TLS client context that accepts the daemon's self-signed cert."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def redirect_persistence(tmpdir: str | Path) -> PersistenceState:
    """Point the history database at ``<tmpdir>/.kiss``; return the prior state.

    ``persistence._get_db()`` invalidates cached per-thread connections
    whenever ``_DB_PATH`` changes, so reassigning the module globals is
    the supported test redirect.  Pass the result to
    :func:`restore_persistence` when the test is done.
    """
    saved = (persistence._DB_PATH, persistence._db_conn, persistence._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    persistence._KISS_DIR = kiss_dir
    persistence._DB_PATH = kiss_dir / "history.db"
    persistence._db_conn = None
    return saved


def restore_persistence(saved: PersistenceState) -> None:
    """Undo :func:`redirect_persistence`.

    ``_close_db`` drains and stops the event-writer thread before the
    scratch connection goes away: closing ``persistence._db_conn``
    directly while that thread is still flushing a finished task's
    events is a use-after-close inside the sqlite3 C module (a SIGSEGV
    of the whole pytest process, not a Python exception).  The
    process-owner marker minted under the scratch ``_KISS_DIR``
    (``task-owners/<token>.lock``, held open for the liveness probe) is
    released too: Windows refuses to delete an open file, so leaving it
    would make the caller's temporary-directory cleanup fail.  The next
    owner-token request re-mints it under the restored directory.
    """
    persistence._release_owner_marker()
    persistence._close_db()
    persistence._DB_PATH, persistence._db_conn, persistence._KISS_DIR = saved


def run_git(cwd: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``git *args`` in *cwd* and return the completed process (never raises)."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def init_repo(repo: str | Path) -> None:
    """Create a git repository at *repo*, on branch ``main``, with one committed ``seed.txt``."""
    Path(repo).mkdir(parents=True, exist_ok=True)
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test User")
    run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n", encoding="utf-8")
    run_git(repo, "add", "seed.txt")
    run_git(repo, "commit", "-q", "-m", "seed")


def kiss_wt_branches(repo: str | Path) -> list[str]:
    """Return the names of the ``kiss/wt-*`` branches of *repo*."""
    out = run_git(repo, "branch", "--list", "kiss/wt-*").stdout
    return [line.strip().lstrip("*+ ").strip() for line in out.splitlines() if line.strip()]


def patch_sorcar_parent_run(fake_run: Any) -> Any:
    """Replace ``SorcarAgent``'s parent-class ``run`` with *fake_run*; return the original.

    *fake_run* is called as ``fake_run(agent, **kwargs)``.  Pass the
    returned original to :func:`restore_sorcar_parent_run`.
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: object, **kwargs: object) -> str:
        return cast(str, fake_run(self_agent, **kwargs))

    parent.run = _run_proxy
    return original


def restore_sorcar_parent_run(original: Any) -> None:
    """Undo :func:`patch_sorcar_parent_run`."""
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    cast(Any, SorcarAgent.__mro__[1]).run = original


def clear_agent_states() -> None:
    """Unregister every agent state in the process-global registry."""
    for state in agent_state.snapshot():
        agent_state.unregister(state.task_id, state)


def clear_tab_states(tab_id: str) -> None:
    """Unregister every agent state registered for *tab_id*."""
    for state in agent_state.snapshot():
        if state.tab_id == tab_id:
            agent_state.unregister(state.task_id, state)


class EntryGate:
    """Hold a thread at the entry of one real function until released.

    A race test needs a thread parked INSIDE a real code path (not a
    sleep that hopes the other thread gets there in time).  Installed
    as a profile hook, the gate sets ``reached`` when a thread enters
    the function whose ``__code__`` it watches and blocks that thread
    until ``release`` is set.  Nothing is replaced: the function then
    runs as it is.  :meth:`start` installs the gate with
    ``threading.setprofile``, so every thread started afterwards
    carries it (the handler's thread and any worker it spawns); call
    ``threading.setprofile(None)`` in teardown.
    """

    def __init__(self, code: Any) -> None:
        self.code = code
        self.reached = threading.Event()
        self.release = threading.Event()

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is self.code:
            self.reached.set()
            self.release.wait(30)

    def start(self, target: Any, *args: Any) -> threading.Thread:
        """Install the gate and run ``target(*args)`` on a daemon thread."""
        threading.setprofile(self)
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        return thread


def make_capturing_server() -> tuple[AgentServer, list[dict[str, Any]]]:
    """Create a ``AgentServer`` whose broadcasts are appended to the returned list."""
    from kiss.server.server import AgentServer

    server = AgentServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def make_recording_server() -> tuple[AgentServer, list[dict[str, Any]], threading.Lock]:
    """Create a ``AgentServer`` whose broadcasts are recorded, persisted and captured.

    Every event gets its task id injected, is recorded into the printer's
    replay buffer and written to the history database, and is then
    appended to the returned list under the returned lock.
    """
    from kiss.server.server import AgentServer

    server = AgentServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        ev = printer._inject_task_id(event)
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        with lock:
            events.append(ev)

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


def docker_available() -> bool:
    """Return True when a Docker daemon answers a ping."""
    import docker

    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Run with an empty agent-state registry and leave it empty."""
    clear_agent_states()
    try:
        yield
    finally:
        clear_agent_states()


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``KISS_HOME`` to a fresh, empty directory for each test."""
    home = tmp_path / ".kiss"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    return home


@contextlib.contextmanager
def environ(**values: str) -> Iterator[None]:
    """Set environment variables for the block, then restore them."""
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def file_names(event: dict[str, Any]) -> list[str]:
    """Extract the file-name strings from a ``files`` event."""
    names: list[str] = []
    for entry in event.get("files", []):
        if isinstance(entry, dict):
            names.append(str(entry.get("text", "")))
        else:
            names.append(str(entry))
    return names


def touch(root: Path, *rel: str) -> None:
    """Create empty files at each *rel* path below *root*."""
    for r in rel:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


def write_trajectory(jobs_root: Path, job: str, name: str) -> None:
    """Create a minimal trajectory YAML under ``<jobs_root>/<job>/trajectories``."""
    traj_dir = jobs_root / job / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "id": 1,
        "run_start_timestamp": 100,
        "run_end_timestamp": 200,
        "model": "test-model",
        "command": "do something",
        "step_count": 3,
        "max_steps": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    (traj_dir / "trajectory_0.yaml").write_text(yaml.safe_dump(data))


def write_fake_player(tmp_path: Path, marker_dir: Path) -> Path:
    """Write a python script standing in for the daemon's audio player.

    The script receives the audio file path as its LAST argument and
    writes one unique JSON marker file into *marker_dir* recording the
    audio bytes (base64) it read.
    """
    marker_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_player.py"
    script.write_text(
        "import base64, json, os, sys, uuid\n"
        f"marker_dir = {str(marker_dir)!r}\n"
        "path = sys.argv[-1]\n"
        "with open(path, 'rb') as fh:\n"
        "    data = fh.read()\n"
        "marker = os.path.join(marker_dir, uuid.uuid4().hex + '.json')\n"
        "with open(marker, 'w') as fh:\n"
        "    json.dump({'audio_b64':"
        " base64.b64encode(data).decode('ascii')}, fh)\n"
    )
    return script


@pytest.fixture(autouse=True)
def _reset_persistence_connection_cache() -> Iterator[None]:
    """Reset the persistence layer's cached sqlite connections before each test.

    ``persistence._close_db()`` stops and drains the background event
    writer, clears the sequence/HasEvents caches, bumps the global
    connection generation counter (lazily invalidating EVERY thread's
    cached connection, not just this thread's), and closes the current
    thread's connection.  The next ``_get_db()`` call therefore
    reconnects against the CURRENT ``persistence._DB_PATH`` and
    re-creates the schema, eliminating the test-order-dependent
    ``task_history`` failures described in the module docstring.
    """
    persistence._close_db()
    yield
