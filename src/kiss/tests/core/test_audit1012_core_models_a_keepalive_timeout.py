# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (core-models-a): a keep-alive ping under a ``timeout`` model_config.

``timeout`` is a request parameter the Anthropic SDK accepts, so a
``model_config`` carrying it survives into the ping's kwargs.  The ping
used to pass its own bound as a second ``timeout=`` keyword, which raised
``TypeError`` on every ping; the keep-alive thread swallowed that as a
failed ping and the prompt cache silently went cold.  Same harness as
:mod:`kiss.tests.core.test_prompt_cache_keepalive`: a real
:class:`~kiss.core.kiss_agent.KISSAgent` on the native Anthropic adapter
against the scripted local Messages server.
"""

from __future__ import annotations

import pytest

import kiss.core.prompt_cache_keepalive as keepalive
from kiss.core.models.anthropic_model import KEEP_ALIVE_TIMEOUT_SECONDS
from kiss.tests.core import local_anthropic_server as anthropic_server
from kiss.tests.core.test_prompt_cache_keepalive import _run_anthropic_agent, _script


def test_ping_keeps_its_own_timeout_when_model_config_sets_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.3)
    script = _script(600)
    with anthropic_server.serve(script) as (url, requests):
        agent = _run_anthropic_agent(monkeypatch, url, len(script), model_config={"timeout": 60})
    pings = [r for r in requests if not r.get("stream")]
    # The 1 s tool with a 0.3 s interval gets pings (none were sent before
    # the fix), and the ping's own bound is what the SDK applied: a
    # per-request timeout is sent as a client option, never in the body.
    assert 1 <= len(pings) <= 4
    assert all("timeout" not in ping for ping in pings)
    assert KEEP_ALIVE_TIMEOUT_SECONDS != 60
    assert "slept with timeout 600" in str(agent.model.conversation)
