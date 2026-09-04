# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest configuration and shared test utilities for KISS tests.

Orphan-sweep join guard
-----------------------

Every :class:`~kiss.server.server.VSCodeServer` constructor starts a
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
the order-dependent crash in large ``tests/agents/vscode`` runs, e.g.
``test_per_window_reply_isolation.py`` closing the connection while an
``orphan-task-sweep`` thread was inside ``_recover_orphaned_tasks``).
``asyncTearDown`` is therefore wrapped too, joining sweeps before its
body runs.
"""

import functools
import os
import shutil
import tempfile
import threading
import unittest
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as _th
from kiss.core import stop_signal, vscode_config
from kiss.core.kiss_error import KISSError

# Generous: a sweep only walks the sentinel rows of one temporary
# database, so it finishes in milliseconds unless the machine is badly
# overloaded.
_SWEEP_JOIN_TIMEOUT_SECONDS = 30.0

_subprocess_rc = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".coveragerc.subprocess")
if os.path.isfile(_subprocess_rc):
    os.environ.setdefault("COVERAGE_PROCESS_START", os.path.abspath(_subprocess_rc))

os.environ["BROWSER"] = "true"

_test_kiss_home = tempfile.mkdtemp(prefix="kiss_test_")
os.environ["KISS_HOME"] = _test_kiss_home
_th._db_conn = None
_th._KISS_DIR = Path(_test_kiss_home)
_th._DB_PATH = _th._KISS_DIR / "sorcar.db"

DEFAULT_MODEL = "claude-opus-4-6"


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
            _tear_down_after_orphan_sweeps, instance.tearDown,
        )
    if isinstance(instance, unittest.IsolatedAsyncioTestCase):
        # functools.partial of a coroutine function still satisfies
        # the inspect.iscoroutinefunction assert in _callAsync.
        instance.asyncTearDown = functools.partial(  # type: ignore[method-assign]
            _async_tear_down_after_orphan_sweeps, instance.asyncTearDown,
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
    """Point ``VSCodeServer``'s default ``work_dir`` away from this repo.

    ``VSCodeServer.__init__`` falls back to ``os.getcwd()`` when
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


def _drop_redundant_config_overrides() -> None:
    """Delete ``CONFIG_DIR``/``CONFIG_PATH`` overrides equal to the lazy value.

    ``vscode_config.CONFIG_DIR``/``CONFIG_PATH`` are lazy module
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

    module_dict = vars(vscode_config)
    lazy_dir = kiss_home()
    for name, lazy_value in (
        ("CONFIG_DIR", lazy_dir),
        ("CONFIG_PATH", lazy_dir / "config.json"),
    ):
        if module_dict.get(name) == lazy_value:
            delattr(vscode_config, name)


@pytest.fixture(autouse=True)
def _isolated_shared_config() -> Iterator[None]:
    """Undo per-test damage to the session-shared ``config.json``.

    The session-wide ``KISS_HOME`` above means ``config.json`` is shared
    by every test in the run, and a leaked non-empty ``remote_password``
    turns every later websocket handshake into ``auth_required`` (the
    Playwright tests then hang behind ``#auth-modal``).  Two leak paths
    existed, both order-dependent:

    * **Pinned path overrides.**  ``vscode_config.CONFIG_DIR`` /
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
    module_dict = vars(vscode_config)
    saved_overrides = {
        name: module_dict[name]
        for name in override_names
        if name in module_dict
    }
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
            setattr(vscode_config, name, saved_overrides[name])
        elif name in module_dict:
            delattr(vscode_config, name)
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

    ``VSCodeServer`` persists the canonical tab registry to
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
