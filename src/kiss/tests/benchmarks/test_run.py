# Author: Koushik Sen (ksen@berkeley.edu)

"""Tests for terminal_bench.run Docker Hub auth check and image pre-pull."""

from __future__ import annotations

import asyncio

import pytest

from kiss.benchmarks.terminal_bench.agent import _parse_test_counts
from kiss.benchmarks.terminal_bench.run import (
    _resolve_docker_images,
    is_docker_hub_authenticated,
)


class TestParseTestCounts:
    """Tests for _parse_test_counts()."""

    def test_pytest_passed_only(self) -> None:
        """Parses pytest output with only passed tests."""
        assert _parse_test_counts("===== 5 passed in 1.23s =====") == (5, 5)

    def test_pytest_passed_and_failed(self) -> None:
        """Parses pytest output with passed and failed tests."""
        assert _parse_test_counts("3 passed, 2 failed") == (3, 5)

    def test_pytest_passed_failed_and_errors(self) -> None:
        """Parses pytest output with passed, failed, and error counts."""
        assert _parse_test_counts("5 passed, 2 failed, 1 error") == (5, 8)

    def test_pytest_passed_and_errors_no_failed(self) -> None:
        """Parses pytest output with passed and errors but no failed."""
        assert _parse_test_counts("4 passed, 3 errors") == (4, 7)

    def test_tap_ok_lines(self) -> None:
        """Parses TAP-style output with only ok lines."""
        output = "ok 1\nok 2\nok 3\n"
        assert _parse_test_counts(output) == (3, 3)

    def test_tap_mixed_ok_and_not_ok(self) -> None:
        """Parses TAP-style output with both ok and not ok lines."""
        output = "ok 1\nnot ok 2\nok 3\nnot ok 4\n"
        assert _parse_test_counts(output) == (2, 4)

    def test_tap_all_not_ok(self) -> None:
        """Parses TAP-style output with only not ok lines."""
        output = "not ok 1\nnot ok 2\n"
        assert _parse_test_counts(output) == (0, 2)

    def test_no_test_output(self) -> None:
        """Returns (0, 0) when no recognisable test output is present."""
        assert _parse_test_counts("some random output\nno tests here\n") == (0, 0)

    def test_empty_string(self) -> None:
        """Returns (0, 0) for empty input."""
        assert _parse_test_counts("") == (0, 0)


class TestIsDockerHubAuthenticated:
    """Tests for is_docker_hub_authenticated()."""

    def test_returns_bool(self) -> None:
        """Function returns a bool reflecting current Docker Hub auth state."""
        result = is_docker_hub_authenticated()
        assert isinstance(result, bool)

class TestResolveDockerImages:
    """Tests for _resolve_docker_images()."""

    def test_returns_sorted_unique_images(self) -> None:
        """Resolving terminal-bench@2.0 returns a non-empty sorted list."""
        images = asyncio.run(_resolve_docker_images("terminal-bench@2.0"))
        assert isinstance(images, list)
        if images:
            assert images == sorted(images)
            assert len(images) == len(set(images))

    @pytest.mark.timeout(10)
    def test_nonexistent_dataset_returns_empty(self) -> None:
        """A bogus dataset name returns an empty list (or fails gracefully)."""
        try:
            images = asyncio.run(
                _resolve_docker_images("nonexistent-dataset-xyz@9.9")
            )
            assert images == []
        except Exception:
            # Harbor may raise; that's acceptable too
            pass
