# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Pytest configuration and shared test utilities for KISS tests."""

import os
import unittest

import pytest

from kiss.core.kiss_error import KISSError

DEFAULT_MODEL = "gpt-5.2"


def pytest_addoption(parser):
    """Add --model option to pytest.

    Args:
        parser: The pytest argument parser to add options to.

    Returns:
        None
    """
    parser.addoption(
        "--model",
        action="store",
        default=DEFAULT_MODEL,
        help=f"Model name to test (default: {DEFAULT_MODEL})",
    )


# Ignore gepa and openevolve test files
collect_ignore = [
    "test_openevolve.py",
]


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


# =============================================================================
# API Key availability helpers
# =============================================================================


def has_openai_api_key() -> bool:
    """Check if OPENAI_API_KEY environment variable is set.

    Returns:
        bool: True if the API key is available and non-empty.
    """
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_anthropic_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY environment variable is set.

    Returns:
        bool: True if the API key is available and non-empty.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_gemini_api_key() -> bool:
    """Check if GEMINI_API_KEY environment variable is set.

    Returns:
        bool: True if the API key is available and non-empty.
    """
    return bool(os.environ.get("GEMINI_API_KEY"))


def has_together_api_key() -> bool:
    """Check if TOGETHER_API_KEY environment variable is set.

    Returns:
        bool: True if the API key is available and non-empty.
    """
    return bool(os.environ.get("TOGETHER_API_KEY"))


def has_openrouter_api_key() -> bool:
    """Check if OPENROUTER_API_KEY environment variable is set.

    Returns:
        bool: True if the API key is available and non-empty.
    """
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def get_required_api_key_for_model(model_name: str) -> str | None:
    """Determine which API key is required for a given model.

    Args:
        model_name: The name of the model.

    Returns:
        str | None: The name of the required environment variable, or None if unknown.
    """
    if model_name.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    elif model_name == "text-embedding-004":
        return "GEMINI_API_KEY"
    elif (
        model_name.startswith(("gpt", "text-embedding", "o1", "o3", "o4", "codex"))
        and not model_name.startswith("openai/gpt-oss")
    ):
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
    return None


def has_api_key_for_model(model_name: str) -> bool:
    """Check if the required API key for a model is available.

    Args:
        model_name: The name of the model.

    Returns:
        bool: True if the required API key is available.
    """
    key_name = get_required_api_key_for_model(model_name)
    if key_name is None:
        return True  # Unknown model, assume available
    return bool(os.environ.get(key_name))


def skip_if_no_api_key_for_model(model_name: str) -> None:
    """Skip the current test if the required API key for a model is not available.

    Args:
        model_name: The name of the model.

    Raises:
        unittest.SkipTest: If the required API key is not available.
    """
    key_name = get_required_api_key_for_model(model_name)
    if key_name and not os.environ.get(key_name):
        raise unittest.SkipTest(f"Skipping test: {key_name} is not set")


# =============================================================================
# Pytest skip markers
# =============================================================================

# Skip markers for tests requiring specific API keys
requires_openai_api_key = pytest.mark.skipif(
    not has_openai_api_key(),
    reason="OPENAI_API_KEY environment variable not set",
)

requires_anthropic_api_key = pytest.mark.skipif(
    not has_anthropic_api_key(),
    reason="ANTHROPIC_API_KEY environment variable not set",
)

requires_gemini_api_key = pytest.mark.skipif(
    not has_gemini_api_key(),
    reason="GEMINI_API_KEY environment variable not set",
)

requires_together_api_key = pytest.mark.skipif(
    not has_together_api_key(),
    reason="TOGETHER_API_KEY environment variable not set",
)

requires_openrouter_api_key = pytest.mark.skipif(
    not has_openrouter_api_key(),
    reason="OPENROUTER_API_KEY environment variable not set",
)


# =============================================================================
# Shared test fixtures
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory that's automatically cleaned up.

    The fixture also changes to the temp directory during the test and
    restores the original directory afterward.

    Yields:
        Path: The resolved path to the temporary directory.
    """
    original_dir = os.getcwd()
    resolved_path = tmp_path.resolve()
    os.chdir(resolved_path)
    yield resolved_path
    os.chdir(original_dir)


@pytest.fixture
def verbose_config():
    """Saves and restores the verbose config setting.

    Yields:
        None: Use config_module.DEFAULT_CONFIG.agent.verbose in the test.
    """
    from kiss.core import config as config_module

    original = config_module.DEFAULT_CONFIG.agent.verbose
    yield
    config_module.DEFAULT_CONFIG.agent.verbose = original


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
