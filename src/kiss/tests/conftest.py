# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest configuration and shared test utilities for KISS tests.

Orphan-sweep join guard
-----------------------

Every :class:`~kiss.server.server.AgentServer` constructor starts a
daemon thread named ``orphan-task-sweep`` that runs SQL on a per-thread
SQLite connection which ``persistence._get_db()`` also publishes in the
module-global ``_db_conn`` (see ``_run_orphan_sweep`` in
``kiss/server/server.py``).  About 150 test files construct a server in
``setUp`` and then, in ``tearDown``, close ``persistence._db_conn`` and
delete the temporary KISS_HOME.  If the sweep is still inside
``db.execute(...)`` at that point, the C-level
``pysqlite_connection_execute`` call dereferences a freed connection and
the whole interpreter dies with SIGSEGV, taking the entire pytest
process down with it.

:func:`pytest_runtest_call` below therefore joins every live sweep
thread right before each ``unittest.TestCase.tearDown`` body runs — the
last moment at which the connection is still valid.  Joining is done by
wrapping the test instance's ``tearDown`` rather than from the
hookwrapper's ``finally``: for unittest-style tests pytest runs
``setUp``/test/``tearDown`` inside a single ``runtest`` call, so the
``finally`` would only fire long after ``tearDown`` already closed the
connection.  The ``finally`` is still used as a backstop for sweeps
started by non-unittest tests.

``IsolatedAsyncioTestCase`` needs the same guard on ``asyncTearDown``:
CPython's ``_callTearDown`` awaits ``asyncTearDown`` **before** it runs
the sync ``tearDown``, and the ~60 async server test files close
``persistence._db_conn`` inside ``asyncTearDown`` — so wrapping only
the sync ``tearDown`` joins the sweep *after* the connection is already
closed, leaving the exact SIGSEGV this guard exists to prevent (seen as
the order-dependent crash in large ``tests/webapp`` runs, e.g.
``test_per_window_reply_isolation.py`` closing the connection while an
``orphan-task-sweep`` thread was inside ``_recover_orphaned_tasks``).
``asyncTearDown`` is therefore wrapped too, joining sweeps before its
body runs.
"""

import atexit
import contextlib
import functools
import os
import random
import shutil
import socket
import sys
import sysconfig
import tempfile
import threading
import time
import unittest
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as _th
from kiss.core import stop_signal, user_config
from kiss.core.kiss_error import KISSError
from kiss.core.processes import suppress_error_dialogs
from kiss.tests import subprocess_reaper

# Generous: a sweep only walks the sentinel rows of one temporary
# database, so it finishes in milliseconds unless the machine is badly
# overloaded.
_SWEEP_JOIN_TIMEOUT_SECONDS = 30.0

_subprocess_rc = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".coveragerc.subprocess")
if os.path.isfile(_subprocess_rc):
    os.environ.setdefault("COVERAGE_PROCESS_START", os.path.abspath(_subprocess_rc))

os.environ["BROWSER"] = "true"

# A pytest process launched from a hidden window (the Windows full-suite
# driver) inherits an error mode under which CreateProcess of a corrupt
# executable blocks on a dialog nobody can click; the daemon opts out
# at start-up and so does the test process.
suppress_error_dialogs()

# Muse-auth is on by default in production (see muse_auth._common.
# muse_auth_enabled), but the legacy connector suites emulate raw
# service HTTP endpoints and would otherwise route every request
# through a per-test authd daemon.  Pin the legacy transport for the
# whole session; the Muse suites opt back in per test with
# monkeypatch.setenv("KISS_MUSE_AUTH", "1").
os.environ["KISS_MUSE_AUTH"] = "0"


def _seed_voice_models(test_home: str, real_home: Path) -> None:
    """Link the developer's downloaded voice models into the test home.

    The voice tests (``kiss.server.speech`` and the browser
    wake-word pipeline) fetch 40-80 MB of Vosk models into
    ``$KISS_HOME/models`` on first use.  With the session-wide
    temporary ``KISS_HOME`` below, every pytest process re-downloaded
    them from alphacephei.com and ccoreilly.github.io, and under a
    parallel full run the TLS handshake regularly timed out.  Each
    entry of the real cache is symlinked individually, so a download
    that does happen lands in the temporary home and never touches
    the real cache (``_atomic_publish`` replaces the symlink, not its
    target).  Nothing is seeded when the real cache is absent.
    """
    # Absolute, or a relative KISS_HOME would yield links resolved
    # against the temporary models dir instead of the working directory.
    real_models = (real_home / "models").absolute()
    if not real_models.is_dir():
        return
    test_models = Path(test_home, "models")
    test_models.mkdir()
    for entry in real_models.iterdir():
        if not entry.name.startswith("."):
            (test_models / entry.name).symlink_to(entry)


def _remove_test_kiss_home(home: str, owner_pid: int) -> None:
    """Delete the session ``KISS_HOME`` when the process that made it exits.

    Every pytest process (each split of a parallel run, each
    ``uv run pytest file.py``, ``--collect-only``, IDE discovery)
    imports this module and makes its own directory; before this
    handler existed nothing removed it, and 1,700 ``kiss_test_*``
    trees (6 GB) piled up in one month.  Registered with :mod:`atexit`
    rather than ``pytest_sessionfinish`` so it also fires for a plain
    import of the module and for a run cut short by ``pytest.exit`` or
    Ctrl-C.

    atexit runs handlers newest-first, so the three import-time
    handlers that still write into the home — persistence's event
    drain (``_drain_events_at_exit`` reopens ``history.db`` and
    recreates the directory for a queued event) and owner-marker
    release, and the subprocess reaper's last-resort sweep (the test
    bucket of a run cut short by ``pytest.exit`` never reached
    teardown) — would run *after* this one.  They are idempotent, so
    this handler runs them first and their own atexit calls find
    nothing left to do.  The database connection is deliberately left
    open: closing it while an ``orphan-task-sweep`` thread is inside
    ``execute`` is the SIGSEGV described in the module docstring, and
    unlinking an open SQLite file is harmless on POSIX.

    A child forked from the pytest process inherits the atexit table;
    the pid check keeps such a child from deleting its parent's home
    when it exits through ``sys.exit`` instead of ``os._exit``.
    Errors are ignored: a still-open ``history.db`` on Windows is not
    worth failing the exit for.

    Args:
        home: The temporary ``KISS_HOME`` created at import.
        owner_pid: The pid of the process that created it.
    """
    if os.getpid() != owner_pid:
        return
    subprocess_reaper._sweep_at_exit()
    _th._drain_events_at_exit()
    _th._release_owner_marker()
    shutil.rmtree(home, ignore_errors=True)


_test_kiss_home = tempfile.mkdtemp(prefix="kiss_test_")
atexit.register(_remove_test_kiss_home, _test_kiss_home, os.getpid())
_seed_voice_models(_test_kiss_home, Path(os.environ.get("KISS_HOME") or Path.home() / ".kiss"))
os.environ["KISS_HOME"] = _test_kiss_home
# The chat page auto-opens the Tips window once per version in every
# fresh browser context (web_server._build_html sends ``show: true``).
# The modal would intercept the clicks of the Playwright suites, so the
# session opts out the way a user does; the tips tests that need the
# auto-open point KISS_HOME at a home of their own.
Path(_test_kiss_home, "TIPS_DISABLED").write_text("test session opt-out\n")
# The daemon polls PyPI at startup and raises a sticky "update available"
# toast over the top-right of every chat page as soon as a release newer
# than this checkout is published, so Playwright clicks there failed
# deterministically whenever the live index was ahead (2026-09-23,
# 2026-10-02).  The session snoozes it the way "Remind me later" does:
# ``web_server._is_update_snoozed`` treats an unparseable
# ``snoozedLatest`` as covering every release until ``snoozeUntilMs``.
# The passive settings-button badge still appears, so tests that wait
# for ``#cfg-update-btn.has-update`` keep their marker.  The snooze
# tests point KISS_HOME at a home of their own.
Path(_test_kiss_home, ".update-check.json").write_text(
    f'{{"snoozeUntilMs": {int(time.time() + 10 * 365 * 24 * 3600) * 1000}, "snoozedLatest": ""}}\n'
)
_th._db_conn = None
_th._KISS_DIR = Path(_test_kiss_home)
_th._DB_PATH = _th._KISS_DIR / "history.db"

DEFAULT_MODEL = "claude-opus-4-6"

# Platform gates.  A test that exercises a POSIX-only mechanism (pty,
# fcntl semantics, rlimits, signals, systemd, ``chmod``-based permission
# denial, ``bash`` scripts) skips on Windows with a reason naming the
# mechanism.  Use ``posix_only(...)`` for the mechanism itself and
# ``is_root()`` instead of ``os.geteuid() == 0``.
IS_WINDOWS = sys.platform == "win32"


def posix_only(reason: str) -> pytest.MarkDecorator:
    """Skip the decorated test or module on Windows.

    Args:
        reason: The POSIX-only mechanism the test depends on.

    Returns:
        A ``skipif`` mark that fires on Windows.
    """
    return pytest.mark.skipif(IS_WINDOWS, reason=f"POSIX-only: {reason}")


def is_root() -> bool:
    """Return whether the test process runs as root (never on Windows)."""
    return hasattr(os, "geteuid") and os.geteuid() == 0


def find_free_port() -> int:
    """Return a TCP port that is free on 127.0.0.1 right now.

    A random port below every OS's ephemeral range (Linux hands out
    32768-60999, macOS and Windows 49152-65535) is tried first: a test
    that closes its listener and binds the same port again (a login
    session replaced by its successor) would otherwise race the
    outgoing connections of the parallel runner's other processes,
    which the kernel assigns from that range, for the port it just
    gave back.  Port 0 (kernel's pick) is the fallback.
    """
    for _ in range(20):
        port = random.randrange(20000, 32768)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


AMBIENT_AWS_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_CREDENTIAL_EXPIRATION",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CREDENTIAL_FILE",
)
"""Environment variables through which boto3's default credential chain
finds AWS credentials without a config file."""


@pytest.fixture(autouse=True)
def no_ambient_aws_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the developer's ambient AWS credentials from every test.

    ``bedrock_credentials_configured()`` consults boto3's default chain, so
    a real ``~/.aws`` profile, an exported key pair or an IRSA token would
    make Bedrock models "configured" in any test that expects no provider
    at all (default-model selection, the ``No model`` picker state).  The
    chain's environment variables are unset and its three file locations
    are pointed at files that do not exist; a test that needs an ambient
    source sets the variables it wants on top of this.
    """
    for name in AMBIENT_AWS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("BOTO_CONFIG", str(tmp_path / "no-boto-config"))


@pytest.fixture
def mcp_redirect_port(monkeypatch: pytest.MonkeyPatch) -> int:
    """Run the MCP OAuth loopback redirect listener on a free ephemeral port.

    The product binds the fixed port 53683 (a dynamically registered
    client keeps working only while its redirect URI never changes), but
    every test registers a fresh client against its own in-process
    authorization server, so the port can differ per test.  Patching the
    module constants, which ``mcp_oauth`` reads at call time, lets the
    parallel runner's pytest processes run these tests at once instead
    of taking turns on one port.

    Returns:
        The port the redirect listener binds for this test.
    """
    from kiss.agents.sorcar import mcp_oauth

    port = find_free_port()
    monkeypatch.setattr(mcp_oauth, "MCP_REDIRECT_PORT", port)
    monkeypatch.setattr(mcp_oauth, "MCP_REDIRECT_URI", f"http://localhost:{port}/callback")
    return port


@pytest.fixture
def loopback_redirect_port(monkeypatch: pytest.MonkeyPatch) -> int:
    """Run the Slack/Discord PKCE loopback redirect listener on a free ephemeral port.

    Same idea as :func:`mcp_redirect_port` for the fixed port 53682 of
    ``_oauth_apps``: the tests' stand-in API servers accept any redirect
    URI, so each test gets its own port.  ``slack_sea`` and
    ``discord_sea`` import ``LOOPBACK_REDIRECT_URI`` by name, so their
    copies are patched too.

    Returns:
        The port the redirect listener binds for this test.
    """
    from kiss.agents.third_party_agents import _oauth_apps
    from kiss.agents.third_party_agents.discord import discord_sea
    from kiss.agents.third_party_agents.slack import slack_sea

    port = find_free_port()
    uri = f"http://localhost:{port}/callback"
    monkeypatch.setattr(_oauth_apps, "LOOPBACK_PORT", port)
    monkeypatch.setattr(_oauth_apps, "LOOPBACK_REDIRECT_URI", uri)
    monkeypatch.setattr(slack_sea, "LOOPBACK_REDIRECT_URI", uri)
    monkeypatch.setattr(discord_sea, "LOOPBACK_REDIRECT_URI", uri)
    return port


@contextlib.contextmanager
def occupy_loopback_port(port: int) -> Iterator[None]:
    """Listen on loopback *port* so that the code under test cannot bind it.

    ``SO_REUSEADDR`` on POSIX lets the blocker bind over TIME_WAIT leftovers
    of earlier tests while still denying a second listener.  On Windows
    that same flag *grants* the bind to any other ``SO_REUSEADDR`` socket
    (which every ``HTTPServer`` is), so the blocker needs
    ``SO_EXCLUSIVEADDRUSE`` there; Windows never refuses a TIME_WAIT port.

    Args:
        port: The loopback port to occupy.

    Yields:
        None while the port is held.
    """
    with socket.socket() as blocker:
        if sys.platform == "win32":
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", port))
        blocker.listen(1)
        yield


@contextlib.contextmanager
def nproc_limit_lowered_to_one() -> Iterator[None]:
    """Lower the soft ``RLIMIT_NPROC`` to 1 for the block, then restore it.

    Threads count against the per-user process limit for a non-root
    uid, so inside the block ``Thread.start()`` genuinely raises
    ``RuntimeError`` on hosts that enforce the limit (Linux).

    The restore re-reads the hard limit instead of reusing the value
    reported before lowering: macOS *reports* the system-wide
    ``kern.maxproc`` as the hard limit but silently clamps it to
    ``kern.maxprocperuid`` on the first ``setrlimit`` call, so putting
    the reported value back raises ``ValueError: not allowed to raise
    maximum limit`` and would leave the whole pytest process stuck at
    soft NPROC=1 -- every later ``subprocess`` spawn then fails with
    ``BlockingIOError``.

    Raises:
        ValueError, OSError: The limit could not be lowered (the block
            is not entered).
    """
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
    try:
        yield
    finally:
        _, current_hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (min(soft, current_hard), current_hard))


def thread_start_can_be_starved() -> bool:
    """Return whether lowering ``RLIMIT_NPROC`` makes ``Thread.start`` fail here.

    False for root (the limit is not enforced), on hosts where the limit
    cannot be lowered, and on hosts that do not count threads against it
    (macOS): tests that need real thread exhaustion skip in those cases.
    """
    if IS_WINDOWS or is_root():
        return False
    try:
        with nproc_limit_lowered_to_one():
            probe = threading.Thread(target=lambda: None)
            try:
                probe.start()
            except RuntimeError:
                return True
            probe.join()
            return False
    except (ValueError, OSError):
        return False


# Errors a hot reader polling a file that writers atomically replace
# must retry rather than report as a torn read.  Windows refuses to open
# the target for the instant ``os.replace`` swaps it in (sharing
# violation -> ``PermissionError``); the next read sees one complete
# file.  NTFS also implements a superseding rename as "unlink the old
# name, link the new one", so with several writers racing on one target
# a lookup can land between the two steps and see no file at all
# (``FileNotFoundError``): measured on Windows Server 2022, 0 in 25k
# reads with 1-4 idle writers, ~1 % with 8 writers or a loaded machine;
# the ``FILE_RENAME_FLAG_POSIX_SEMANTICS`` rename shows the same window.
# Neither error is a torn file.  POSIX renames never do either, so there
# nothing is tolerated.
TRANSIENT_REPLACE_READ_ERRORS: tuple[type[OSError], ...] = (
    (PermissionError, FileNotFoundError) if IS_WINDOWS else ()
)

# Pause (seconds) between two reads of a hot reader polling a file that
# writers atomically replace.  Python's ``open`` takes no
# ``FILE_SHARE_DELETE``, so on Windows a reader that reopens the file
# back to back holds a handle ~90 % of the time (measured: 87 us per
# read, 12 us between reads) and every ``os.replace`` of the writer
# fails until it lands in the gap -- on a loaded machine that starves
# the writer for seconds.  No real consumer (an editor, the VS Code
# extension, a poller) reopens a file thousands of times per second; a
# 2 ms pause keeps the reader hot enough to catch a torn write while
# letting the writer publish.
HOT_READER_PAUSE = 0.002


def install_cli_script(script: Path, source: str) -> Path:
    """Write the Python program *source* to *script* as a stand-in CLI.

    The stand-in must be found by ``shutil.which(script.name)`` and start
    from ``subprocess.Popen([path, ...])`` exactly like the real CLI.  On
    POSIX that is the shebang line in *source* plus the executable bit.
    Windows cannot execute a shebang script (``WinError 193``) and
    ``shutil.which`` never returns an extensionless file there, so a
    ``<name>.cmd`` shim that runs the script through this interpreter is
    written beside it -- the same shape as the ``claude.cmd`` /
    ``codex.cmd`` shims npm installs.  Git bash (the shell behind the
    agent's ``Bash`` tool and cron command jobs) finds the extensionless
    file first and honours its shebang, so on Windows the shebang names
    this interpreter outright: ``/usr/bin/env python3`` would look for a
    ``python3`` that a uv-managed Windows box does not have.

    Args:
        script: Where to write the program (its directory goes on PATH).
        source: The program text, starting with a shebang line.

    Returns:
        The file to launch as the CLI: *script* itself, or its ``.cmd``
        shim on Windows (what ``shutil.which`` finds there), for tests
        that name the executable instead of searching PATH.
    """
    if IS_WINDOWS and source.startswith("#!"):
        source = f"#!{Path(sys.executable).as_posix()}\n{source.partition(chr(10))[2]}"
    script.write_text(source, encoding="utf-8", newline="\n")
    script.chmod(0o755)
    if not IS_WINDOWS:
        return script
    shim = script.with_name(script.name + ".cmd")
    shim.write_text(f'@"{sys.executable}" "%~dp0{script.name}" %*\n', encoding="utf-8")
    return shim


_FAKE_CLOUDFLARED_WRAPPER = """\
import os
import runpy
import sys
import sysconfig

# A bare ``cloudflared`` start gives ``['']``; ``run_path`` fills argv[0].
sys.argv = [sys.executable, *(sys.argv if sys.argv != [""] else [])]
try:
    runpy.run_path({body_path!r}, run_name="__main__")
except SystemExit as exc:
    code = exc.code
    code = code if isinstance(code, int) else (0 if code is None else 1)
else:
    code = 0
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
"""


def install_fake_cloudflared(directory: Path, body: str) -> Path:
    """Install a fake ``cloudflared`` executable running the Python *body*.

    The product spawns ``cloudflared`` by bare name from ``PATH`` and
    verifies pidfile processes by executable basename
    (``web_server._looks_like_cloudflared``), so the fake must be a
    single process that *is* named ``cloudflared`` -- a ``#!python``
    script is not: macOS ``ps -o comm=`` reports the *interpreter* for
    a shebang script (Linux reports the script), and Windows cannot run
    one at all.  So the fake is the interpreter itself under the name
    ``cloudflared``: a symlink to it on POSIX (the kernel names the
    process after the exec'd path), and on Windows -- where the kernel
    reports a symlink's *target* as the image path and a ``.cmd`` shim
    would make the visible process ``cmd.exe`` -- a copy of the base
    interpreter as ``cloudflared.exe`` beside its runtime DLLs.

    *body* is run through the embeddable-distribution mechanism, which
    CPython's ``getpath`` applies on every platform: a
    ``cloudflared._pth`` file adjacent to the executable pins
    ``sys.path`` to this directory plus the real installation's standard
    library and turns on ``import site``, and *body* runs from a
    ``sitecustomize`` module here, which ``site`` imports at startup.
    ``sys.argv`` is ``[<script path>, <cloudflared args>...]``.

    Callers add *directory* to ``PATH`` themselves.

    Args:
        directory: Existing directory that receives the fake.
        body: Python source to run; it may call ``sys.exit(code)``.

    Returns:
        Path of the executable (``cloudflared`` or ``cloudflared.exe``).
    """
    base = Path(getattr(sys, "_base_executable", sys.executable))
    if IS_WINDOWS:
        fake = directory / "cloudflared.exe"
        shutil.copy2(base, fake)
        for dll in [*base.parent.glob("python3*.dll"), *base.parent.glob("vcruntime*.dll")]:
            shutil.copy2(dll, directory / dll.name)
        stdlib_dirs = [base.parent / "Lib", base.parent / "DLLs"]
    else:
        fake = directory / "cloudflared"
        fake.unlink(missing_ok=True)
        fake.symlink_to(base)
        base_vars = {"base": sys.base_prefix, "platbase": sys.base_exec_prefix}
        stdlib_dirs = [
            Path(sysconfig.get_path("stdlib", vars=base_vars)),
            Path(sysconfig.get_path("platstdlib", vars=base_vars)) / "lib-dynload",
        ]
    # This directory goes FIRST: Debian's Python ships its own
    # ``sitecustomize.py`` in the stdlib directory, which would shadow
    # the fake's if the stdlib were searched before it.
    (directory / "cloudflared._pth").write_text(
        ".\n" + "".join(f"{d}\n" for d in stdlib_dirs) + "import site\n",
        encoding="utf-8",
    )
    body_path = directory / "fake_cloudflared_body.py"
    body_path.write_text(body, encoding="utf-8")
    (directory / "sitecustomize.py").write_text(
        _FAKE_CLOUDFLARED_WRAPPER.format(body_path=str(body_path)),
        encoding="utf-8",
    )
    return fake


def goto_retrying_network_change(page, url: str, **goto_kwargs) -> None:
    """Navigate the Playwright *page* to *url*, retrying ``ERR_NETWORK_CHANGED``.

    Chromium aborts every in-flight request with ``net::ERR_NETWORK_CHANGED``
    when a host interface appears or disappears mid-request; on this box
    other jobs' Docker containers add and remove veth links all the time,
    so under full-suite load a first navigation sporadically dies that way.
    The navigation is retried up to three times for that error only; any
    other error, or a third failure, propagates unchanged.

    The same event can also strike the remote webapp's self-reload: when
    one of the page's scripts fails to load, the guard in
    ``media/assetLoadGuard.js`` reloads the page (up to three times
    per 30 s), and when a reload's own document request is aborted too,
    Chromium shows its error page
    (``chrome-error://chromewebdata/``, an empty DOM). ``page.goto`` tracked
    the first navigation and returns normally, so the caller's first
    ``wait_for_selector`` would sit out its full timeout. That case is
    retried under the same rule: only when the failed navigation reported
    ``net::ERR_NETWORK_CHANGED``; any other reason raises an
    ``AssertionError`` naming it, so a real server fault stays visible.

    Args:
        page: A ``playwright.sync_api.Page``.
        url: The address to open.
        **goto_kwargs: Passed through to ``page.goto`` (``wait_until`` ...).
    """
    _navigate_retrying_network_change(page, functools.partial(page.goto, url, **goto_kwargs), url)


def mark_version_seen(context_or_page) -> None:
    """Make a fresh Playwright profile one that already saw the running version.

    The first page load of a version (a fresh install, an update) leaves
    only the Recommendations section of the task-info panel expanded
    (``resetMetaSectionsForVersion`` in ``media/main.js``).  Tests about
    the panel's everyday layout want what every later load shows (every
    section expanded, the user's folds kept), so this stores the running
    version under the key main.js checks before the page's scripts run.

    Args:
        context_or_page: A Playwright ``BrowserContext`` or ``Page``;
            ``add_init_script`` is called on it.
    """
    from kiss.server.web_server import _read_version

    version = _read_version().replace("\\", "\\\\").replace("'", "\\'")
    context_or_page.add_init_script(
        f"localStorage.setItem('kiss-meta-sections-version', '{version}')"
    )


def reload_retrying_network_change(page, **reload_kwargs) -> None:
    """Reload the Playwright *page*, retrying ``ERR_NETWORK_CHANGED``.

    ``page.reload()`` is a navigation like ``page.goto`` and dies the same
    way when a host interface changes mid-request (full-suite run of
    2026-10-01: ``Page.reload: net::ERR_NETWORK_CHANGED`` in
    ``test_activity_bar``). Same retry rule as
    :func:`goto_retrying_network_change`: up to three attempts, for that
    error only.

    Args:
        page: A ``playwright.sync_api.Page``.
        **reload_kwargs: Passed through to ``page.reload`` (``wait_until`` ...).
    """
    _navigate_retrying_network_change(
        page, functools.partial(page.reload, **reload_kwargs), page.url
    )


def _navigate_retrying_network_change(page, navigate: Callable[[], object], label: str) -> None:
    """Run *navigate* on *page* up to three times while it fails with ``ERR_NETWORK_CHANGED``.

    Args:
        page: A ``playwright.sync_api.Page``.
        navigate: A no-argument callable performing the navigation (a bound
            ``page.goto``/``page.reload`` with its arguments).
        label: The address named in the error when the page ends on
            Chromium's error page for a reason other than a network change.
    """
    from playwright.sync_api import Error as PlaywrightError

    failures: list[str] = []

    def note_failed_navigation(request) -> None:
        # Every failed request, not only navigations: the webapp
        # reloads itself when a page script or stylesheet fails to
        # load, so the asset that died is the cause of a reload that
        # then ends on the error page (full-suite run of 2026-10-12:
        # an unexplained self-reload under a 97-way load).
        kind = "navigation" if request.is_navigation_request() else request.resource_type
        failures.append(f"{request.url} ({kind}): {request.failure}")

    page.on("requestfailed", note_failed_navigation)
    try:
        for attempt in range(3):
            failures.clear()
            try:
                navigate()
            except PlaywrightError as exc:
                if "net::ERR_NETWORK_CHANGED" not in str(exc) or attempt == 2:
                    raise
                time.sleep(1.0)
                continue
            if not page.url.startswith("chrome-error://"):
                return
            network_changed = any("net::ERR_NETWORK_CHANGED" in f for f in failures)
            if not network_changed or attempt == 2:
                raise AssertionError(f"{label} ended on {page.url}; failed requests: {failures}")
            time.sleep(1.0)
    finally:
        page.remove_listener("requestfailed", note_failed_navigation)


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=DEFAULT_MODEL,
        help=f"Model name to test (default: {DEFAULT_MODEL})",
    )


collect_ignore = ["run_all_models_test.py"]


def join_orphan_sweeps() -> None:
    """Wait for every live ``orphan-task-sweep`` thread to finish.

    Returns:
        None. Threads that outlive the timeout are left running: the
        caller cannot do better, and blocking forever would hang the
        whole session.
    """
    for thread in threading.enumerate():
        if thread.name == "orphan-task-sweep" and thread.is_alive():
            thread.join(timeout=_SWEEP_JOIN_TIMEOUT_SECONDS)


def _tear_down_after_orphan_sweeps(tear_down: Callable[[], None]) -> None:
    """Join lingering orphan sweeps, then run the test's own teardown.

    Args:
        tear_down: The ``unittest.TestCase.tearDown`` bound method this
            call replaces.

    Returns:
        None.
    """
    join_orphan_sweeps()
    tear_down()


async def _async_tear_down_after_orphan_sweeps(
    tear_down: Callable[[], Awaitable[None]],
) -> None:
    """Join lingering orphan sweeps, then run the async teardown body.

    ``IsolatedAsyncioTestCase._callTearDown`` awaits ``asyncTearDown``
    BEFORE the sync ``tearDown``, so this is the last moment at which
    ``persistence._db_conn`` — which many async server fixtures close
    inside ``asyncTearDown`` — is still guaranteed valid for a sweep.

    Args:
        tear_down: The ``asyncTearDown`` bound coroutine method this
            call replaces.

    Returns:
        None.
    """
    join_orphan_sweeps()
    await tear_down()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    """Guard every test's teardown against the orphan-sweep race.

    Also unbinds the runner thread's stop event afterwards.  A test that
    binds one (``printer._thread_local.stop_event = ...``) publishes it
    for the whole thread — that is the point of
    :mod:`kiss.core.stop_signal`, which lets model streams see a stop —
    so a test that leaves a *set* event behind would make the next
    test's first ``print()`` raise ``KeyboardInterrupt``.  Production
    unbinds per run; tests get it centrally here.

    Args:
        item: The test about to run. For ``unittest.TestCase`` items its
            ``tearDown`` — and for ``IsolatedAsyncioTestCase`` items its
            ``asyncTearDown``, which runs first — is wrapped so sweeps
            are joined before the teardown body closes the database.

    Returns:
        Generator required by the pytest hookwrapper protocol.
    """
    instance = getattr(item, "instance", None)
    if isinstance(instance, unittest.TestCase):
        instance.tearDown = functools.partial(  # type: ignore[method-assign]
            _tear_down_after_orphan_sweeps,
            instance.tearDown,
        )
    if isinstance(instance, unittest.IsolatedAsyncioTestCase):
        # functools.partial of a coroutine function still satisfies
        # the inspect.iscoroutinefunction assert in _callAsync.
        instance.asyncTearDown = functools.partial(  # type: ignore[method-assign]
            _async_tear_down_after_orphan_sweeps,
            instance.asyncTearDown,
        )
    try:
        yield
    finally:
        join_orphan_sweeps()
        stop_signal.set_thread_stop_event(None)


def simple_calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Args:
        expression: The arithmetic expression to evaluate (e.g., '2+2', '10*5')

    Returns:
        The result of the expression as a string
    """
    try:
        compiled = compile(expression, "<string>", "eval")
        return str(eval(compiled, {"__builtins__": {}}, {}))
    except Exception as e:
        raise KISSError(f"Error evaluating expression: {e}") from e


def has_openai_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_anthropic_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_gemini_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def has_together_api_key() -> bool:
    return bool(os.environ.get("TOGETHER_API_KEY"))


def has_openrouter_api_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def has_zai_api_key() -> bool:
    return bool(os.environ.get("ZAI_API_KEY"))


def has_moonshot_api_key() -> bool:
    return bool(os.environ.get("MOONSHOT_API_KEY"))


def get_required_api_key_for_model(model_name: str) -> str | None:
    """Return the environment variable a model needs, or ``None`` if none.

    Derived from the two routing tables in
    :mod:`kiss.core.models.model_info` — the same ones the ``model()``
    factory dispatches on — so a vendor added there is honoured here
    without a second, hand-maintained copy of its prefixes drifting out
    of sync.  ``None`` means the model needs no API key: either it is
    routed to a subscription CLI (``cc/``, ``codex/``), whose credential
    is a local executable, or nothing routes it at all.

    Args:
        model_name: A ``MODEL_INFO`` key.

    Returns:
        The environment variable name, or ``None``.
    """
    from kiss.core.models.model_info import (
        _NATIVE_PROVIDERS,
        _match_openai_compatible_provider,
    )

    provider = _match_openai_compatible_provider(model_name)
    if provider is not None:
        return provider.api_key_name
    for prefix, _label, api_key_name in _NATIVE_PROVIDERS:
        if model_name.startswith(prefix):
            return api_key_name
    return None


def has_api_key_for_model(model_name: str) -> bool:
    key_name = get_required_api_key_for_model(model_name)
    if key_name is None:
        return True
    return bool(os.environ.get(key_name))


def skip_if_no_api_key_for_model(model_name: str) -> None:
    key_name = get_required_api_key_for_model(model_name)
    if key_name and not os.environ.get(key_name):
        raise unittest.SkipTest(f"Skipping test: {key_name} is not set")


requires_openai_api_key = pytest.mark.skipif(
    not has_openai_api_key(), reason="OPENAI_API_KEY environment variable not set"
)
requires_anthropic_api_key = pytest.mark.skipif(
    not has_anthropic_api_key(), reason="ANTHROPIC_API_KEY environment variable not set"
)
requires_gemini_api_key = pytest.mark.skipif(
    not has_gemini_api_key(), reason="GEMINI_API_KEY environment variable not set"
)
requires_together_api_key = pytest.mark.skipif(
    not has_together_api_key(), reason="TOGETHER_API_KEY environment variable not set"
)
requires_openrouter_api_key = pytest.mark.skipif(
    not has_openrouter_api_key(), reason="OPENROUTER_API_KEY environment variable not set"
)
requires_zai_api_key = pytest.mark.skipif(
    not has_zai_api_key(), reason="ZAI_API_KEY environment variable not set"
)
requires_moonshot_api_key = pytest.mark.skipif(
    not has_moonshot_api_key(),
    reason="MOONSHOT_API_KEY environment variable not set",
)


@pytest.fixture(autouse=True)
def _isolated_default_workdir(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point ``AgentServer``'s default ``work_dir`` away from this repo.

    ``AgentServer.__init__`` falls back to ``os.getcwd()`` when
    ``KISS_WORKDIR`` is unset, which is the *developer repository* when
    pytest runs from the repo root.  Since the diff/merge review was
    removed, ``_run_task_inner`` auto-commits a dirty working tree at
    task end — so any test that drives ``_run_task`` on a server whose
    ``work_dir`` was never overridden would commit the developer's
    in-progress work.  Defaulting the variable to a per-test temporary
    directory makes that path harmless.

    The override is unconditional — it also replaces an ambient
    ``KISS_WORKDIR`` — because developer machines commonly run the
    daemon with the variable pointing at the repository itself, and
    honoring that value would disable the guard exactly where it
    matters.  Tests that need a specific work dir must set the
    variable (or assign ``server.work_dir``) inside their own
    setup/body, which runs after this fixture and therefore wins.
    The directory lives in the system temp dir rather than under
    pytest's ``basetemp``: it must be OUTSIDE the test's own
    ``tmp_path`` (so tests that scan their ``tmp_path`` do not see an
    extra entry) and OUTSIDE the repository even when pytest is run
    with an in-repo ``--basetemp`` (e.g. ``--basetemp=./tmp/...`` from
    a parallel split runner) — a default work dir nested anywhere
    inside the repo's git tree would still let the auto-commit reach
    the developer's checkout.

    Yields:
        None.
    """
    default_dir = tempfile.mkdtemp(prefix="kiss-default-workdir-")
    monkeypatch.setenv("KISS_WORKDIR", default_dir)
    yield
    shutil.rmtree(default_dir, ignore_errors=True)


@pytest.fixture
def claude_live_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a live ``cc/`` test bill whatever this host can pay with.

    The subscription switch defaults to on, so the real ``claude`` CLI is
    run without ``ANTHROPIC_API_KEY`` and needs a ``claude login``.  On a
    host without that login the test falls back to the API key by turning
    the switch off for its duration, and skips when there is no key either
    (the ``live_cli`` contract: never fail on the developer's credentials).
    """
    from kiss.core.config import DEFAULT_CONFIG
    from kiss.core.models.claude_code_model import claude_auth_status

    if claude_auth_status()["loggedIn"]:
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("claude is not logged in and ANTHROPIC_API_KEY is not set")
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)


@pytest.fixture(autouse=True)
def _isolated_codex_home(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``CODEX_HOME`` at a directory that does not exist.

    ``codex/`` turns read the ``codex`` CLI's ``auth.json`` (default
    ``~/.codex``) to refuse a saved API-key login while the subscription
    switch is on.  Tests with a fake ``codex`` must not see the
    developer's own login; tests marked ``live_cli`` spawn the real CLI
    and keep the real home.  A test of the login check sets its own.
    """
    if request.node.get_closest_marker("live_cli") is None:
        monkeypatch.setenv("CODEX_HOME", os.path.join(_test_kiss_home, "no-codex-home"))


def _drop_redundant_config_overrides() -> None:
    """Delete ``CONFIG_DIR``/``CONFIG_PATH`` overrides equal to the lazy value.

    ``user_config.CONFIG_DIR``/``CONFIG_PATH`` are lazy module
    attributes that follow ``$KISS_HOME``; a materialized override with
    the very same value resolves identically *now* but silently stops
    following later ``KISS_HOME`` swaps — the mechanism behind the
    ``remote_password`` shared-config leak.  Overrides pointing anywhere
    else belong to a live test or module-scoped harness redirection and
    are kept.

    Returns:
        None.
    """
    from kiss.core.config import kiss_home

    module_dict = vars(user_config)
    lazy_dir = kiss_home()
    for name, lazy_value in (
        ("CONFIG_DIR", lazy_dir),
        ("CONFIG_PATH", lazy_dir / "config.json"),
    ):
        if module_dict.get(name) == lazy_value:
            delattr(user_config, name)


@pytest.fixture(autouse=True)
def _isolated_shared_config() -> Iterator[None]:
    """Undo per-test damage to the session-shared ``config.json``.

    The session-wide ``KISS_HOME`` above means ``config.json`` is shared
    by every test in the run, and a leaked non-empty ``remote_password``
    turns every later websocket handshake into ``auth_required`` (the
    Playwright tests then hang behind ``#auth-modal``).  Two leak paths
    existed, both order-dependent:

    * **Pinned path overrides.**  ``user_config.CONFIG_DIR`` /
      ``CONFIG_PATH`` are *lazy* module attributes (``__getattr__``)
      that follow ``$KISS_HOME`` on every access.  The common test
      pattern ``self._orig = vc.CONFIG_DIR`` … ``vc.CONFIG_DIR =
      self._orig`` restores the right *value* but materializes it as a
      permanent module global — so a later test that swaps
      ``KISS_HOME`` and calls ``save_config`` (expecting the write to
      land in its isolated home, e.g.
      ``test_ntfy_topic_isolation.py``) silently writes into the
      session-shared file instead.  Overrides that did not exist before
      a test are therefore deleted after it, restoring laziness.

    * **Direct writes.**  Any test that writes the shared file without
      restoring it.  The file's byte content is snapshotted before each
      test and written back (or unlinked) afterwards.

    Tests that point ``CONFIG_DIR``/``CONFIG_PATH`` or ``KISS_HOME`` at
    their own temporary directory are unaffected: their writes never
    touch the session-shared path, and their overrides are cleaned up
    for them.  Overrides installed by a *module*- or *class*-scoped
    harness (e.g. ``test_content_tab_file_links.py``) exist before this
    fixture's setup and are preserved for the harness's remaining
    tests; only *redundant* overrides — ones equal to the lazy
    ``$KISS_HOME`` resolution, which such a harness leaves behind when
    it restores the values it read at setup — are dropped, because they
    resolve identically but silently stop following ``KISS_HOME``.

    Yields:
        None.
    """
    _drop_redundant_config_overrides()
    override_names = ("CONFIG_DIR", "CONFIG_PATH")
    module_dict = vars(user_config)
    saved_overrides = {name: module_dict[name] for name in override_names if name in module_dict}
    # api_keys.env is the canonical API-key store, written by
    # ``save_api_key`` and the legacy-RC key migration inside
    # ``load_api_keys``.  Like ``config.json`` it lives in the
    # session-shared ``KISS_HOME``, so a test that saves or migrates a
    # key without redirecting ``CONFIG_DIR`` would otherwise leak that
    # key into every later test's loads.
    shared_files = (
        Path(_test_kiss_home) / "config.json",
        Path(_test_kiss_home) / "api_keys.env",
    )
    saved_contents: dict[Path, bytes | None] = {}
    for shared in shared_files:
        try:
            saved_contents[shared] = shared.read_bytes()
        except OSError:
            saved_contents[shared] = None
    yield
    for name in override_names:
        if name in saved_overrides:
            setattr(user_config, name, saved_overrides[name])
        elif name in module_dict:
            delattr(user_config, name)
    _drop_redundant_config_overrides()
    for shared, saved_content in saved_contents.items():
        try:
            if saved_content is None:
                shared.unlink(missing_ok=True)
            else:
                shared.write_bytes(saved_content)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def _isolated_tab_registry() -> Iterator[None]:
    """Start every test with an empty shared tab registry.

    ``AgentServer`` persists the canonical tab registry to
    ``KISS_HOME/tabs.json``.  The session-wide ``KISS_HOME`` above is
    shared by every test in the run, so tabs registered by one test
    (a ``ready`` merge, an ``openTab``, a ``run``) would otherwise leak
    into the next test's registry — changing its ``ready`` replay
    fan-out and defeating merge-if-empty expectations.  Tests that
    redirect ``persistence._KISS_DIR`` themselves are unaffected.

    Yields:
        None.
    """
    for kiss_dir in {Path(_test_kiss_home), Path(_th._KISS_DIR)}:
        try:
            (kiss_dir / "tabs.json").unlink(missing_ok=True)
        except OSError:
            pass
    yield


@pytest.fixture
def temp_dir(tmp_path):
    original_dir = os.getcwd()
    resolved_path = tmp_path.resolve()
    os.chdir(resolved_path)
    yield resolved_path
    os.chdir(original_dir)


def simple_test_tool(message: str) -> str:
    """A simple test tool that echoes a message.

    Args:
        message: The message to echo back.

    Returns:
        The echoed message with a prefix.
    """
    return f"Echo: {message}"


def add_numbers(a: int, b: int) -> str:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum as a string.
    """
    return str(a + b)
