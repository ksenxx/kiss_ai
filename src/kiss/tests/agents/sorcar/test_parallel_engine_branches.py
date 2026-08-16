# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Parentless-abandon branch of the parallel fan-out engine.

Moved from ``kiss.tests.sorcar.test_parallel_engine_branches`` because
this test calls :func:`_register_abandoned` directly and depends only
on ``kiss.core`` and ``kiss.agents.sorcar``.  The engine's other branch
tests remain in the original module: they run real agents against the
stand-in model server from ``kiss.tests.sorcar.parallel_agent_harness``,
which depends on ``kiss.server``.
"""

from __future__ import annotations

from kiss.agents.sorcar.sorcar_agent import _register_abandoned


def test_register_abandoned_ignores_a_parentless_fanout() -> None:
    """A bare functional fan-out has no agent to hand children to."""
    _register_abandoned(None, [], [], [])
