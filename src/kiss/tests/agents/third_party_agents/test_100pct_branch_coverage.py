# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for 100% branch coverage of the channel-agent CLI.

The server-only json_printer/server.py coverage moved to
``kiss.tests.server.test_100pct_branch_coverage``; this file keeps the
``_channel_cli`` coverage.

Targets remaining uncovered branches in:
  _channel_cli.py (channel-agent CLI helpers)

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import tempfile

from kiss.agents.third_party_agents._channel_cli import (
    _build_arg_parser,
    _build_run_kwargs,
)


class TestCliHelpers:
    """Cover uncovered branches in _channel_cli.py."""

    def test_build_run_kwargs(self) -> None:
        """_build_run_kwargs builds kwargs from parsed args."""
        with tempfile.TemporaryDirectory() as d:
            parser = _build_arg_parser()
            args = parser.parse_args(["-t", "do something", "-w", d, "-e", "http://localhost:1234"])
            kwargs = _build_run_kwargs(args)
            assert kwargs["prompt_template"] == "do something"
            assert kwargs["work_dir"] == d
            assert kwargs["model_config"]["base_url"] == "http://localhost:1234"
            assert kwargs["web_tools"] is True
