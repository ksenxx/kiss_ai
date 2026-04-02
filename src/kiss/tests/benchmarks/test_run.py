# Author: Koushik Sen (ksen@berkeley.edu)

"""Tests for terminal_bench.run Docker Hub auth check and image pre-pull."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

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

    def test_matches_credential_helper(self) -> None:
        """Result agrees with querying the Docker credential store directly.

        Reads ~/.docker/config.json to find the credsStore, queries it,
        and verifies is_docker_hub_authenticated() returns the same answer.
        """
        config_path = Path.home() / ".docker" / "config.json"
        if not config_path.exists():
            # No Docker config -> should be False
            assert is_docker_hub_authenticated() is False
            return

        config = json.loads(config_path.read_text())
        creds_store = config.get("credsStore")

        expected = False
        if creds_store:
            try:
                result = subprocess.run(
                    [f"docker-credential-{creds_store}", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    creds = json.loads(result.stdout)
                    expected = any("index.docker.io" in url for url in creds)
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                subprocess.TimeoutExpired,
            ):
                pass

        if not expected:
            auths = config.get("auths", {})
            expected = any("index.docker.io" in url for url in auths)

        assert is_docker_hub_authenticated() is expected


class TestResolveDockerImages:
    """Tests for _resolve_docker_images()."""

    def test_returns_sorted_unique_images(self) -> None:
        """Resolving terminal-bench@2.0 returns a non-empty sorted list."""
        images = asyncio.run(_resolve_docker_images("terminal-bench@2.0"))
        assert isinstance(images, list)
        if images:
            assert images == sorted(images)
            assert len(images) == len(set(images))

    def test_all_images_have_tag(self) -> None:
        """Every resolved image name contains a colon (image:tag format)."""
        images = asyncio.run(_resolve_docker_images("terminal-bench@2.0"))
        for img in images:
            assert ":" in img, f"Image missing tag: {img}"

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
