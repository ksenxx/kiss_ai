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


collect_ignore = ["test_openevolve.py", "run_all_models_test.py"]


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
    if model_name.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    elif model_name == "text-embedding-004":
        return "GEMINI_API_KEY"
    elif model_name.startswith(
        ("gpt", "text-embedding", "o1", "o3", "o4", "codex", "computer-use")
    ) and not model_name.startswith("openai/gpt-oss"):
        return "OPENAI_API_KEY"
    elif model_name.startswith(
        (
            "meta-llama/",
            "Qwen/",
            "mistralai/",
            "deepseek-ai/",
            "deepcogito/",
            "google/gemma",
            "moonshotai/",
            "nvidia/",
            "zai-org/",
            "openai/gpt-oss",
            "arcee-ai/",
            "refuel-ai/",
            "marin-community/",
            "essentialai/",
            "BAAI/",
            "togethercomputer/",
            "intfloat/",
            "Alibaba-NLP/",
        )
    ):
        return "TOGETHER_API_KEY"
    elif model_name.startswith("claude-"):
        return "ANTHROPIC_API_KEY"
    elif model_name.startswith("gemini-"):
        return "GEMINI_API_KEY"
    elif model_name.startswith("glm-"):
        return "ZAI_API_KEY"
    elif model_name.startswith("kimi-") or model_name.startswith("moonshot-"):
        return "MOONSHOT_API_KEY"
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
