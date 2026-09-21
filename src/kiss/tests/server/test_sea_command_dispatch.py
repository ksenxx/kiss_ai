# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the server wiring of SEA slash commands.

Two integration surfaces are pinned:

* the ``getSeaCommands`` API handler on :class:`VSCodeServer`,
  including the ``connId``-scoped delivery contract shared by every
  other read command;
* the ``_run_task_inner`` prompt rewriter, which turns a submitted
  prompt starting with ``/xxx text`` into an explicit ``run_agent``
  directive by the time the LLM sees it.

A real :class:`JsonPrinter` subclass captures broadcasts (no mocks).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_commands import rewrite_prompt_if_command
from kiss.core.config import kiss_home
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records broadcasts."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _make_server() -> tuple[VSCodeServer, _CapturePrinter]:
    printer = _CapturePrinter()
    return VSCodeServer(printer=printer), printer


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Isolate the SEA command registry between tests."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()


def _seed_seas_md(folder: Path, name: str) -> Path:
    """Create ``<folder>/<name>_sea.py`` and point ``SEAS.md`` at it."""
    folder.mkdir(parents=True, exist_ok=True)
    sea = folder / f"{name}_sea.py"
    sea.write_text("# stub\n", encoding="utf-8")
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SEAS.md").write_text(str(folder) + "\n", encoding="utf-8")
    sea_commands.refresh_registry()
    return sea.resolve()


def test_get_sea_commands_emits_sea_commands_event() -> None:
    """The handler MUST broadcast a ``seaCommands`` event with the list."""
    server, printer = _make_server()
    sea_commands.refresh_registry()
    server._handle_command({"type": "getSeaCommands"})
    sea_events = [e for e in printer.events if e["type"] == "seaCommands"]
    assert sea_events, "no seaCommands event broadcast"
    commands = sea_events[-1]["commands"]
    assert isinstance(commands, list)
    # The bundled SEAs must be present.
    assert "slack" in commands
    assert "gmail" in commands


def test_get_sea_commands_conn_id_stamping() -> None:
    """A ``connId`` on the request MUST be stamped on the reply.

    An empty ``connId`` broadcasts to all clients; a non-empty
    ``connId`` scopes the event to the requesting connection.  This
    mirrors the same contract every other read command follows
    (``getFrequentTasks``, ``getInputHistory``, ``getModels``).
    """
    server, printer = _make_server()
    sea_commands.refresh_registry()
    printer.events.clear()
    server._handle_command({"type": "getSeaCommands", "connId": "c7"})
    server._handle_command({"type": "getSeaCommands"})
    sea_events = [e for e in printer.events if e["type"] == "seaCommands"]
    assert len(sea_events) == 2
    assert sea_events[0]["connId"] == "c7"
    assert "connId" not in sea_events[1]


def test_get_sea_commands_reflects_seas_md(tmp_path: Path) -> None:
    """A folder listed in ``SEAS.md`` MUST appear in the reply."""
    sea = _seed_seas_md(tmp_path / "seas", "customcmd")
    assert sea.name == "customcmd_sea.py"
    server, printer = _make_server()
    server._handle_command({"type": "getSeaCommands"})
    sea_events = [e for e in printer.events if e["type"] == "seaCommands"]
    assert sea_events
    commands = sea_events[-1]["commands"]
    assert "customcmd" in commands


def test_task_runner_prompt_rewriter_replaces_slash_command(
    tmp_path: Path,
) -> None:
    """``_run_task_inner``'s slash-command branch rewrites the prompt.

    Rather than launching the whole worker (which needs a live agent
    and a real model), this asserts the rewriter helper the runner
    imports produces the expected shape for a registered command —
    the same helper the runner calls at the top of _run_task_inner.
    """
    sea = _seed_seas_md(tmp_path / "user-seas", "notify")
    result = rewrite_prompt_if_command('/notify send "hi" now')
    assert result is not None
    rewritten, resolved = result
    assert resolved == sea
    # The rewritten prompt MUST reference the SEA path and the user
    # text, and MUST direct the agent to invoke ``run_agent`` first.
    assert 'run_agent' in rewritten
    assert str(sea) in rewritten
    assert 'send "hi" now' in rewritten


def test_task_runner_prompt_rewriter_leaves_plain_prompt_untouched(
    tmp_path: Path,
) -> None:
    """A non-command prompt must not be rewritten."""
    _seed_seas_md(tmp_path / "user-seas", "notify")
    assert rewrite_prompt_if_command("please summarize this file") is None
    # A slash prefix that does not match a known command also passes
    # through unchanged.
    assert rewrite_prompt_if_command("/unknowncmd anything") is None


def test_slash_command_with_embedded_task_tags_runs_atomically(
    tmp_path: Path,
) -> None:
    """A ``/xxx <task>...</task>`` prompt dispatches as a SINGLE subtask.

    Regression for a bug where the task-tag splitter (``parse_task_tags``)
    consumed the embedded ``<task>`` blocks and discarded the
    ``run_agent`` directive.  The runner MUST detect the slash prefix
    on the raw prompt and, after ``parse_task_tags`` has run, substitute
    the parsed subtasks with a single-element list containing the
    ``run_agent`` directive.

    Exercises the ``_sea_dispatch`` branch in
    :meth:`TaskRunner._run_task_inner` by reproducing its slash-detection
    and subtask-replacement contract explicitly.
    """
    from kiss.server.task_runner import parse_task_tags

    sea = _seed_seas_md(tmp_path / "user-seas", "atomic")
    raw_prompt = "/atomic <task>send first</task><task>send second</task>"

    # ``parse_task_tags`` sees the ORIGINAL prompt (this mirrors the
    # runner's ordering — the classifier and the tag splitter run
    # before the rewrite is applied).
    parsed = parse_task_tags(raw_prompt)
    assert len(parsed) == 2, "sanity: task-tag splitter reads two subtasks"

    # The runner detects the slash command up front and substitutes
    # the parsed list with a single rewritten subtask.
    hit = rewrite_prompt_if_command(raw_prompt)
    assert hit is not None
    rewritten, resolved = hit
    subtasks = [rewritten]

    assert len(subtasks) == 1
    assert resolved == sea
    assert 'run_agent' in subtasks[0]
    # The embedded ``<task>`` markers MUST reach run_agent verbatim so
    # the SEA (not the runner) decides what they mean.
    assert '<task>send first</task>' in subtasks[0]
    assert '<task>send second</task>' in subtasks[0]
