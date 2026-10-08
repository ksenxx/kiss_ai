# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Hermes-style cron automations (cron_agent).

Everything runs against the real JSON job store under an isolated
``KISS_HOME`` — no mocks or test doubles (``monkeypatch`` is used
only to isolate environment variables, ``sys.argv``, and the
module-level daemon endpoint-file default between tests).  The only
branches not exercised here are ``_run_prompt_job``'s successful /
silent / timed-out LLM paths: they submit a task to the kiss-web
daemon and require a live LLM endpoint, which is unavailable (and
non-deterministic) in unit tests; the failure path is covered via
``_execute_job``'s exception handling.

The daemon/SEA scenarios (pure kiss.agents.sorcar +
kiss.server closure) moved to ``kiss.tests.server.test_cron_agent``;
this file keeps the delivery test that imports real
``kiss.agents.third_party_agents`` channel modules and the SEA
contract checks.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from kiss.agents.seas.base.base_sea import ChannelSea
from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import (
    cron_job,
    load_jobs,
    tick,
)
from kiss.core.base import SYSTEM_PROMPT
from kiss.tests.agents.sorcar.test_cron_agent import (  # noqa: F401
    _create,
    _isolated_kiss_home,
    _set_job_fields,
)


def test_delivery_error_notes() -> None:
    job = _create(cron_job(
        "create", name="multi", command="echo payload",
        schedule="every 1m",
        deliver="local,nosuchchannel:1,homeassistant:x,telegram:123",
    ))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    notes = load_jobs()[0]["last_delivery"]
    assert len(notes) == 3
    assert "unknown channel 'nosuchchannel'" in notes[0]
    assert "does not support delivery" in notes[1]
    # Telegram module exists and has _make_backend, but no credentials
    # exist under the isolated KISS_HOME, so its factory sys.exit(1)s.
    assert "not authenticated" in notes[2]


def test_get_tools_and_sorcar_wiring() -> None:
    sea = cron_agent.CronAgentSea()
    assert sea.tools([]) == [cron_job, cron_agent.gateway_command]
    # The dispatch preamble reaches the session through the SEA
    # contract (the SEA's ``system_prompt`` method), not through a prompt prefix.
    assert sea.system_prompt("S") == "S\n\n" + cron_agent.CRON_DISPATCH_PREAMBLE
    assert isinstance(sea, ChannelSea)
    # Sorcar's system prompt, as the product loads it, sends scheduling
    # requests to this agent.
    assert 'run_agent tool with "cron"' in SYSTEM_PROMPT
    # The installed console script runs this module's ``main``.
    (script,) = entry_points(group="console_scripts", name="kiss-cron")
    assert script.value == "kiss.agents.sorcar.cron_agent:main"
    assert script.load() is cron_agent.main


_IDLE_SIGNAL_CLI = """#!/usr/bin/env python3
import sys
sys.exit(0)
"""


def test_gateway_command_tick_is_silent_when_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """The command ``gateway_command`` builds is a real, silent channel tick.

    Runs the Signal CLI against a stand-in ``signal-cli`` that receives
    nothing: with ``--quiet`` the tick prints nothing, so scheduled as a
    cron command job it is recorded as ``silent`` and delivers nothing;
    without the flag the CLI keeps its progress output.

    Not covered here: ``--quiet`` also suppresses the ``Resolved user``
    line printed for ``--allow-users`` names.  Slack is the only channel
    whose ``find_user`` resolves names, and ``channel_main`` resolves
    them on the ``_make_backend()`` client, which talks to slack.com
    before ``connect()`` installs the test-injectable base URL — so that
    branch is reachable only with network access or a test double.
    """
    from kiss.agents.third_party_agents.signal import signal_sea
    from kiss.tests.conftest import install_cli_script

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_cli_script(bin_dir / "signal-cli", _IDLE_SIGNAL_CLI)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    signal_sea._config.save({"phone_number": "+1BOT"})

    command = cron_agent.gateway_command("signal", "+1AAA", pairing=False)
    assert command == "kiss-signal --channel=+1AAA --quiet"
    monkeypatch.setattr(sys, "argv", shlex.split(command))
    signal_sea.main()
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(sys, "argv", ["kiss-signal", "--channel=+1AAA"])
    signal_sea.main()
    out = capsys.readouterr().out
    assert "Checking Signal channel for pending messages..." in out
    assert "Processed 0 message(s)." in out

    # Scheduled as a command job, the same tick runs through the console
    # script in its own process and is silent.
    assert shutil.which("kiss-signal") is not None
    job = _create(cron_job("create", name="signal gw", command=command, schedule="every 2m"))
    _set_job_fields(job["id"], next_run_at=1.0)
    assert tick(2.0) == 1
    stored = load_jobs()[0]
    assert (stored["last_status"], stored["last_summary"]) == ("silent", "")
    assert not (tmp_path / "cron" / "output" / f"{job['id']}.md").exists()
