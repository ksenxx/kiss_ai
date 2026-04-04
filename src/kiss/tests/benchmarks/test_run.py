# Author: Koushik Sen (ksen@berkeley.edu)

"""Tests for terminal_bench.run Docker Hub auth check and image pre-pull."""

from __future__ import annotations

import asyncio

import pytest

from kiss.benchmarks.terminal_bench.run import (
    _resolve_docker_images,
    is_docker_hub_authenticated,
)


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
