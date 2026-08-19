# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Hermes-style cron automations (cron_agent).

Everything runs against the real JSON job store under an isolated
``KISS_HOME`` — no mocks or test doubles (``monkeypatch`` is used
only to isolate environment variables, ``sys.argv``, and the
module-level daemon-socket default between tests).  The only
branches not exercised here are ``_run_prompt_job``'s successful /
silent / timed-out LLM paths: they submit a task to the kiss-web
daemon and require a live LLM endpoint, which is unavailable (and
non-deterministic) in unit tests; the failure path is covered via
``_execute_job``'s exception handling.

The daemon/tools-file scenarios (pure kiss.agents.sorcar +
kiss.server closure) moved to ``kiss.tests.server.test_cron_agent``;
this file keeps the delivery test that imports real
``kiss.agents.third_party_agents`` channel modules and the
source/configuration wiring checks.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import (
    cron_job,
    load_jobs,
    tick,
)
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
    assert cron_agent.get_tools() == [cron_job]
    # The module lives in the sorcar package and never imports from
    # kiss.agents.third_party_agents at module scope.
    source_text = Path(cron_agent.__file__).read_text(encoding="utf-8")
    assert "/agents/sorcar/" in cron_agent.__file__
    assert "from kiss.agents.third_party_agents" not in source_text
    assert "import kiss.agents.third_party_agents" not in source_text
    # cron_job is NOT a built-in tool of the default Sorcar toolset:
    # scheduling requests go through run_agent("cron", ...), which
    # dispatches this module as an agent script.
    agent_source = Path(cron_agent.__file__).parent / "sorcar_agent.py"
    agent_text = agent_source.read_text(encoding="utf-8")
    assert "tools.append(cron_job)" not in agent_text
    assert "from kiss.agents.sorcar.cron_agent import cron_job" not in agent_text
    dispatch_source = Path(cron_agent.__file__).parent / "agent_dispatch.py"
    assert (
        "cron_agent.CRON_DISPATCH_PREAMBLE + task"
        in dispatch_source.read_text(encoding="utf-8")
    )
    # The system prompt directs scheduling requests to run_agent("cron").
    system_md = Path(cron_agent.__file__).parents[2] / "SYSTEM.md"
    assert 'run_agent tool with "cron"' in system_md.read_text(encoding="utf-8")
    # The kiss-cron CLI entry point is wired in pyproject.toml.
    pyproject = Path(cron_agent.__file__).parents[4] / "pyproject.toml"
    assert (
        'kiss-cron = "kiss.agents.sorcar.cron_agent:main"'
        in pyproject.read_text(encoding="utf-8")
    )
