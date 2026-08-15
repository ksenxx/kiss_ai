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
"""

import functools
import os
import tempfile
import threading
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as _th
from kiss.core import stop_signal
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
            ``tearDown`` is wrapped so sweeps are joined before the
            teardown body closes the database.

    Returns:
        Generator required by the pytest hookwrapper protocol.
    """
    instance = getattr(item, "instance", None)
    if isinstance(instance, unittest.TestCase):
        instance.tearDown = functools.partial(  # type: ignore[method-assign]
            _tear_down_after_orphan_sweeps, instance.tearDown,
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
def _isolated_default_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
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
    The directory lives OUTSIDE the test's own ``tmp_path`` so tests
    that scan their ``tmp_path`` do not see an extra entry.

    Yields:
        None.
    """
    default_dir = tmp_path_factory.mktemp("kiss-default-workdir")
    monkeypatch.setenv("KISS_WORKDIR", str(default_dir))
    yield


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
