# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Pytest configuration and shared test utilities for KISS tests."""

import pytest

from kiss.core.formatter import Formatter
from kiss.core.kiss_error import KISSError

DEFAULT_MODEL = "gpt-5.2"


def pytest_addoption(parser):
    """Add --model option to pytest."""
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


class CustomFormatter(Formatter):
    """A custom formatter for testing that captures messages."""

    def __init__(self):
        self.messages: list[dict[str, str]] = []
        self.status_messages: list[str] = []

    def format_message(self, message: dict[str, str]) -> str:
        return f"[{message.get('role', '')}]: {message.get('content', '')}"

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(self.format_message(m) for m in messages)

    def print_message(self, message: dict[str, str]) -> None:
        self.messages.append(message)

    def print_messages(self, messages: list[dict[str, str]]) -> None:
        self.messages.extend(messages)

    def print_status(self, message: str) -> None:
        self.status_messages.append(message)

    def print_error(self, message: str) -> None:
        self.status_messages.append(f"ERROR: {message}")

    def print_warning(self, message: str) -> None:
        self.status_messages.append(f"WARNING: {message}")


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


@pytest.fixture
def custom_formatter():
    """Fixture that provides a CustomFormatter instance."""
    return CustomFormatter()
