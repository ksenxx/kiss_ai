# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A shell rewrite is redirected to ``Write``/``Edit`` only in a run that has those tools.

``UsefulTools.Bash`` raises a ``ToolRedirect`` for the ``Write``/``Edit``
a here-document copy or a ``sed -i`` of a shown file stands for, and the
agent makes that call through its own tool path.  Every host toolset got
the redirect, including the ``bash``, ``shell`` and ``review`` profiles
and a SEA whose ``tools()`` drops the editing tools: there the agent had
no ``Write``/``Edit`` to make, answered ``Function Write is not a
registered tool`` and the command had no effect (audit 2026-11-01, S1
F2).  The redirect now follows the run's effective toolset, decided after
the profile and the SEA's ``tools()`` hook: with both targets the rewrite
goes through the agent (its hook sees the ``Write``); without them the
tool applies the rewrite itself, as it did before the redirect existed.

Real ``SorcarAgent`` runs against the scripted local model server, over
real files; nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.tool_verdict import ALLOW, Verdict
from kiss.tests.agents.seas.sea_contract import tool_results
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash here-documents")

HEREDOC = "cat > notes.md <<'EOF'\nnew text\nEOF"


def _applied(work: Path, redirected: bool) -> str:
    """What the rewrite's result opens with: the ``Write`` it was made as, then its result.

    Args:
        work: The run's work dir, where ``notes.md`` is.
        redirected: Whether the agent made the ``Write`` (the result
            follows on its own line) or the tool applied it itself.
    """
    notes = work / "notes.md"
    call = f'Write(file_path="{notes}", content="new text\\n")'
    joint = ", whose result follows:\n" if redirected else ": "
    return f"Applied as {call}{joint}Successfully wrote 9 characters to {notes}"


class _CallLog:
    """Records every tool call the agent's ``tool_call_hook`` sees."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def hook(self, name: str, args: dict[str, Any]) -> Verdict:
        """Record *name* and allow the call."""
        self.calls.append(name)
        return ALLOW


def _without_write_and_edit(tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
    """A SEA ``tools()`` that keeps every tool but the editing pair."""
    return [tool for tool in tools if tool.__name__ not in ("Write", "Edit")]


def _run_rewrite(
    work: Path, log: _CallLog, show: str, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run an agent that views ``notes.md`` with *show*, rewrites it by heredoc, then finishes.

    Returns:
        The requests the scripted model server received.
    """
    (work / "notes.md").write_text("old text\n")
    script = [
        tool_call_body("Bash", {"command": show, "description": "show"}, 1000)
        if show.startswith("cat") else
        tool_call_body("Read", {"file_path": str(work / "notes.md")}, 1000),
        tool_call_body("Bash", {"command": HEREDOC, "description": "rewrite"}, 1100),
        finish_body("<p>done</p>", 1200),
    ]
    with serve(script) as (url, requests):
        SorcarAgent("redirect-profile").run(
            model_name=MODEL,
            prompt_template="Rewrite notes.md.",
            work_dir=str(work),
            max_steps=3,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            tool_call_hook=log.hook,
            web_tools=False,
            use_memory=False,
            verbose=False,
            **kwargs,
        )
    return requests


@pytest.mark.parametrize("profile", ["bash", "shell", "review"])
def test_a_profile_without_the_editing_tools_applies_the_rewrite_in_the_tool(
    profile: str, tmp_path: Path
) -> None:
    """The heredoc copy of a shown file is written, and its result says what it was made as."""
    log = _CallLog()
    requests = _run_rewrite(tmp_path, log, "cat notes.md", tool_profile=profile)
    (_, rewrite) = tool_results(requests[2])
    assert rewrite.startswith(_applied(tmp_path, redirected=False)), rewrite
    assert (tmp_path / "notes.md").read_text() == "new text\n"
    assert log.calls == ["Bash", "Bash", "finish"], "no Write the run does not have"


def test_a_tools_hook_that_drops_the_editing_tools_keeps_the_rewrite_in_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full profile cut down by a SEA's ``tools()`` has the same missing targets."""
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    log = _CallLog()
    requests = _run_rewrite(tmp_path, log, "Read", tools_hook=_without_write_and_edit)
    (_, rewrite) = tool_results(requests[2])
    assert rewrite.startswith(_applied(tmp_path, redirected=False)), rewrite
    assert (tmp_path / "notes.md").read_text() == "new text\n"
    assert "Write" not in log.calls


def test_the_full_profile_still_makes_the_rewrite_through_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``Write`` in the toolset the rewrite is redirected: the hook sees the ``Write``."""
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    log = _CallLog()
    requests = _run_rewrite(tmp_path, log, "Read", tool_profile="shell+edit")
    (_, rewrite) = tool_results(requests[2])
    assert rewrite.startswith(_applied(tmp_path, redirected=True)), rewrite
    assert (tmp_path / "notes.md").read_text() == "new text\n"
    assert log.calls[:3] == ["Read", "Bash", "Write"]
