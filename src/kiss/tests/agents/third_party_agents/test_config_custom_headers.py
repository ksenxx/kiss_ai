# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that custom HTTP headers can be configured via the settings panel.

The settings panel has a textarea (after the custom endpoint field) where
users can enter custom HTTP headers in ``Key:Value`` format, one per line.
These headers flow through to the model via ``model_config["extra_headers"]``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


class TestCLIHeadersFlow(unittest.TestCase):
    """CLI --header option flows into model_config["extra_headers"]."""

    def test_build_run_kwargs_with_headers(self) -> None:
        import argparse

        from kiss.agents.third_party_agents._channel_cli import _build_run_kwargs

        args = argparse.Namespace(
            model_name="gpt-4o",
            endpoint="http://localhost:8080/v1",
            header=["X-Custom:value1", "Authorization:Bearer tok"],
            max_budget=100,
            work_dir=None,
            verbose=True,
            no_web=False,
            parallel=False,
            task="test task",
            file=None,
        )
        kwargs = _build_run_kwargs(args)
        assert kwargs["model_config"]["base_url"] == "http://localhost:8080/v1"
        assert kwargs["model_config"]["extra_headers"] == {
            "X-Custom": "value1",
            "Authorization": "Bearer tok",
        }

    def test_build_run_kwargs_without_headers(self) -> None:
        import argparse

        from kiss.agents.third_party_agents._channel_cli import _build_run_kwargs

        args = argparse.Namespace(
            model_name="gpt-4o",
            endpoint=None,
            header=None,
            max_budget=100,
            work_dir=None,
            verbose=True,
            no_web=False,
            parallel=False,
            task="test task",
            file=None,
        )
        kwargs = _build_run_kwargs(args)
        assert "extra_headers" not in kwargs.get("model_config", {})
