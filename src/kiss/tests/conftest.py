"""Pytest configuration and shared test utilities for KISS tests."""

import os
import unittest

import pytest

from kiss.core.kiss_error import KISSError

# Enable coverage for subprocesses: the .pth file (a1_coverage.pth) checks this
# env var and calls coverage.process_startup() so spawned subprocesses write
# their branch-coverage data to .coverage.subprocess.* files.
_subprocess_rc = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".coveragerc.subprocess")
if os.path.isfile(_subprocess_rc):
    os.environ.setdefault("COVERAGE_PROCESS_START", os.path.abspath(_subprocess_rc))

DEFAULT_MODEL = "claude-opus-4-6"


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=DEFAULT_MODEL,
        help=f"Model name to test (default: {DEFAULT_MODEL})",
    )


collect_ignore = ["test_openevolve.py", "run_all_models_test.py"]


def _is_truthy_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def run_live_api_tests() -> bool:
    """Return True only when live, potentially billable API tests are explicitly enabled."""
    return _is_truthy_env("KISS_RUN_LIVE_API_TESTS")


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


def has_minimax_api_key() -> bool:
    return bool(os.environ.get("MINIMAX_API_KEY"))


def get_required_api_key_for_model(model_name: str) -> str | None:
    if model_name.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    elif model_name == "text-embedding-004":
        return "GEMINI_API_KEY"
    elif model_name.startswith(
        ("chatgpt", "gpt", "text-embedding", "o1", "o3", "o4", "codex", "computer-use")
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
    elif model_name.startswith("minimax-"):
        return "MINIMAX_API_KEY"
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


_LIVE_API_TESTS_REASON = (
    "Live API tests are disabled by default. "
    "Set KISS_RUN_LIVE_API_TESTS=1 to enable provider-backed tests."
)

requires_openai_api_key = pytest.mark.skipif(
    (not has_openai_api_key()) or (not run_live_api_tests()),
    reason=(
        "OPENAI_API_KEY environment variable not set"
        if not has_openai_api_key()
        else _LIVE_API_TESTS_REASON
    ),
)
requires_anthropic_api_key = pytest.mark.skipif(
    (not has_anthropic_api_key()) or (not run_live_api_tests()),
    reason=(
        "ANTHROPIC_API_KEY environment variable not set"
        if not has_anthropic_api_key()
        else _LIVE_API_TESTS_REASON
    ),
)
requires_gemini_api_key = pytest.mark.skipif(
    (not has_gemini_api_key()) or (not run_live_api_tests()),
    reason=(
        "GEMINI_API_KEY environment variable not set"
        if not has_gemini_api_key()
        else _LIVE_API_TESTS_REASON
    ),
)
requires_together_api_key = pytest.mark.skipif(
    (not has_together_api_key()) or (not run_live_api_tests()),
    reason=(
        "TOGETHER_API_KEY environment variable not set"
        if not has_together_api_key()
        else _LIVE_API_TESTS_REASON
    ),
)
requires_openrouter_api_key = pytest.mark.skipif(
    (not has_openrouter_api_key()) or (not run_live_api_tests()),
    reason=(
        "OPENROUTER_API_KEY environment variable not set"
        if not has_openrouter_api_key()
        else _LIVE_API_TESTS_REASON
    ),
)
requires_minimax_api_key = pytest.mark.skipif(
    (not has_minimax_api_key()) or (not run_live_api_tests()),
    reason=(
        "MINIMAX_API_KEY environment variable not set"
        if not has_minimax_api_key()
        else _LIVE_API_TESTS_REASON
    ),
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
