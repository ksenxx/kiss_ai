# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Pytest configuration and shared test utilities for KISS tests."""

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
