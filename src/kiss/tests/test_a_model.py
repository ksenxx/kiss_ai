"""Test suite for OpenAICompatibleModel with configurable model.

Usage:
    pytest src/kiss/tests/test_a_model.py --model=gpt-5.2
"""

import unittest


def pytest_configure(config) -> None:
    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
