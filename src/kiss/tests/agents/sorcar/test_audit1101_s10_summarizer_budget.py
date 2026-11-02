# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The failed-session summarizer must respect the task's cap like a session does.

``RelentlessAgent._summarize_failed_session`` gave its helper
:class:`KISSAgent` a fixed allowance (the task's cap minus what was
banked when it started) and nothing else.  A sub-agent's spend
attributed to the task meanwhile (``_attribute_sub_usage`` from a
dispatch thread) was invisible to the helper, so it kept sending model
requests after the task's combined spend had crossed the user's cap.
The helper now carries the task-wide budget check and is the task's
live session while it runs, so the check sees both the attribution
and the helper's own spend.

Real scripted model server: while it answers the summarizer's first
request, it attributes $0.95 of a finished sub-agent to the $1 task
through the real production path; nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage
from kiss.core.kiss_error import KISSError
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    ScriptedHandler,
    serve,
    tool_call_body,
)


def note() -> str:
    """Do harmless work without finishing the session."""
    return "work completed"


class ChildReportsDuringSummary(ScriptedHandler):
    """Attribute a finished sub-agent's spend while the summarizer's first request is served."""

    agent: ClassVar[RelentlessAgent]

    def do_POST(self) -> None:  # noqa: N802
        # The first two requests exhaust the session; the third is the
        # summarizer's, whose allowance was computed before this lands.
        if len(type(self).requests) == 2:
            _attribute_sub_usage(type(self).agent, 0.95, 1000, 1)
        super().do_POST()


def test_the_summarizer_stops_once_the_task_cap_is_crossed(tmp_path: Path) -> None:
    """No helper request goes out after a concurrent attribution exceeded the cap."""
    agent = RelentlessAgent("summarizer-budget")
    ChildReportsDuringSummary.agent = agent
    script = [
        tool_call_body("note", {}, 1000),
        tool_call_body("note", {}, 1000),
        # The summarizer's first answer costs $0.1001 and calls no finish:
        # a helper blind to the task total would ask again.
        tool_call_body("unknown_tool", {}, 1000, completion_tokens=250_000),
        tool_call_body("finish", {"result": "Progress saved."}, 1000),
    ]
    with serve(script, handler_base=ChildReportsDuringSummary) as (url, requests):
        with pytest.raises(KISSError, match="Task failed after 1 sub-sessions"):
            agent.run(
                model_name=MODEL,
                prompt_template="Work on the task.",
                tools=[note],
                work_dir=str(tmp_path),
                max_steps=2,
                max_sub_sessions=1,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local", "plain_io": True},
                verbose=False,
            )
        count = len(requests)
    spent = agent.usage_snapshot()[0]
    assert spent > 1.0, spent
    assert count == 3, (
        f"{count} model requests: the summarizer kept going after the task's "
        f"${spent:.4f} crossed its $1.00 cap"
    )
    # The helper's one request is banked with the rest of the task.
    assert agent.usage_snapshot()[2] == 4, agent.usage_snapshot()
