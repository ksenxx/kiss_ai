# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The channel CLI ``--header`` option flows into ``model_config["extra_headers"]``.

``_channel_cli._build_run_kwargs`` parses repeated ``Key:Value`` headers
and leaves ``extra_headers`` unset when none are given.
"""

from __future__ import annotations

import unittest


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
