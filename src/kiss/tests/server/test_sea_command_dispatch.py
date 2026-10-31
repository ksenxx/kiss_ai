# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the server wiring of SEA slash commands.

Two integration surfaces are pinned:

* the ``getSeaCommands`` API handler on :class:`AgentServer`,
  including the ``connId``-scoped delivery contract shared by every
  other read command;
* the slash-command run: ``/xxx text`` runs the SEA ``xxx`` DIRECTLY
  in the tab's own run (``slash_command_task`` splits the prompt, and
  :meth:`TaskRunner._run_task` makes the SEA the run's ``seaPath``
  and ``text`` its prompt), so the LLM sees ``text`` as its task while
  the tab, ``state.last_user_prompt`` and the history row keep the raw
  ``/xxx text``.  There is no ``run_agent`` relay turn and no nested
  sub-agent any more.

A real :class:`JsonPrinter` subclass captures broadcasts (no mocks).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.persistence import _get_db, _rw_lock
from kiss.agents.sorcar.sea_commands import slash_command_task
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.server import agent_state, sorcar
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records broadcasts."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _make_server() -> tuple[AgentServer, _CapturePrinter]:
    printer = _CapturePrinter()
    return AgentServer(printer=printer), printer


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Isolate the SEA command registry between tests.

    Also removes the ``SEAS.md`` :func:`_seed_seas_md` wrote into the
    session-wide ``$KISS_HOME`` so it cannot shadow bundled commands
    for later tests in the same process.
    """
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


def _seed_seas_md(folder: Path, name: str, body: str = "# stub\n") -> Path:
    """Create ``<folder>/<name>/<name>_sea.py`` holding *body* and point ``SEAS.md`` at it."""
    sea = folder / name / f"{name}_sea.py"
    sea.parent.mkdir(parents=True, exist_ok=True)
    sea.write_text(body, encoding="utf-8")
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
    (``getInputHistory``, ``getModels``).
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


def test_slash_command_task_splits_a_registered_command(tmp_path: Path) -> None:
    """``slash_command_task`` yields the SEA to run and the trailing text as its task.

    This is the helper :meth:`TaskRunner._run_task` calls first: the
    SEA becomes the run's ``seaPath`` and the text its ``prompt``.
    No ``run_agent`` directive is built any more.
    """
    sea = _seed_seas_md(tmp_path / "user-seas", "notify")
    result = slash_command_task('/notify send "hi" now')
    assert result is not None
    task_text, resolved = result
    assert resolved == sea
    assert task_text == 'send "hi" now'


def test_slash_command_task_leaves_plain_prompt_untouched(tmp_path: Path) -> None:
    """A non-command prompt, an unknown command, a bare command and ``help`` run as usual."""
    _seed_seas_md(
        tmp_path / "user-seas",
        "notify",
        "from kiss.agents.seas.base.base_sea import BaseSea\n\nclass NotifySea(BaseSea):\n"
        "    pass\n",
    )
    assert slash_command_task("please summarize this file") is None
    # A slash prefix that does not match a known command also passes
    # through unchanged.
    assert slash_command_task("/unknowncmd anything") is None
    # A bare command has no task text to run a stock SEA on (one whose
    # ``prompt()`` supplies a default task does run; see the registry tests).
    assert slash_command_task("/notify") is None
    assert slash_command_task("/notify   ") is None
    # ``/xxx help`` is answered by ``description()`` without running the SEA.
    assert slash_command_task("/notify help") is None
    assert slash_command_task("/notify HELP") is None


def test_slash_command_with_embedded_task_tags_runs_atomically(
    tmp_path: Path,
) -> None:
    """A ``/xxx <task>...</task>`` prompt runs the SEA on the WHOLE trailing text.

    Regression for a bug where the task-tag splitter (``parse_task_tags``)
    consumed the embedded ``<task>`` blocks before the slash command
    was recognised.  The runner detects the slash prefix on the raw
    prompt FIRST (:meth:`TaskRunner._run_task`), so the embedded
    ``<task>`` markers reach the SEA verbatim and the SEA (not the
    runner) decides what they mean.
    """
    from kiss.server.task_runner import parse_task_tags

    sea = _seed_seas_md(tmp_path / "user-seas", "atomic")
    raw_prompt = "/atomic <task>send first</task><task>send second</task>"

    # Sanity: the tag splitter alone would read two subtasks out of it.
    parsed = parse_task_tags(raw_prompt)
    assert len(parsed) == 2, "sanity: task-tag splitter reads two subtasks"

    hit = slash_command_task(raw_prompt)
    assert hit is not None
    task_text, resolved = hit
    assert resolved == sea
    assert task_text == "<task>send first</task><task>send second</task>"


def _history_rows() -> list[dict[str, Any]]:
    """Return ``(id, task, parent_task_id)`` of every persisted task row."""
    with _rw_lock.read_lock():
        rows = _get_db().execute(
            "SELECT id, task, parent_task_id FROM task_history "
            "ORDER BY timestamp ASC, rowid ASC",
        ).fetchall()
    return [dict(r) for r in rows]


def _marker_tool(note: str) -> str:
    """Record *note*; the SEA under test adds this tool to the run."""
    return note


class SlashCommandRunTest(DaemonRunApiHarness):
    """``/xxx text`` runs the SEA directly in the tab's own run, against a real daemon."""

    def setUp(self) -> None:
        super().setUp()
        sea_commands._reset_for_tests()
        self.addCleanup(sea_commands._reset_for_tests)
        self.addCleanup((kiss_home() / "SEAS.md").unlink, missing_ok=True)

    def _record_runs(self, runs: list[dict[str, Any]]) -> None:
        """Replace the executor LLM loop with a stub recording what the LLM is given."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""  # follow-up proposer etc.: silent, unrecorded
            arguments = dict(kwargs.get("arguments") or {})
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.step_count = 1
            if "task_description" not in arguments:
                return "result: prior progress\n"
            runs.append({
                "system_prompt": str(kwargs.get("system_prompt") or ""),
                "prompt": str(arguments["task_description"]),
                "tool_names": [t.__name__ for t in (kwargs.get("tools") or [])],
                # What the live tab shows while the LLM works.
                "live_user_prompts": [
                    s.last_user_prompt for s in agent_state.snapshot() if s.last_user_prompt
                ],
            })
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1, total_tokens=1, cost="$0.0001",
                )
            return raw

        KISSAgent.run = stub_run  # type: ignore[assignment,method-assign]

    def test_slash_command_runs_the_sea_directly_in_the_tabs_run(self) -> None:
        """The LLM's task is the trailing text; tab, state and history keep ``/xxx text``."""
        _seed_seas_md(
            Path(self.tmpdir) / "user-seas", "notify",
            """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return 'notify'

    def system_prompt(self, system_prompt):
        return system_prompt + '\\n\\nNOTIFY-PROTOCOL'

    def tools(self, tools):
        from kiss.tests.server.test_sea_command_dispatch import _marker_tool
        return tools + [_marker_tool]
""",
        )
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        raw_prompt = '/notify send "hi" now'
        result = sorcar.run(
            raw_prompt, work_dir=self.repo,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        # ONE LLM run — no relay turn that would then spawn a sub-agent.
        assert len(runs) == 1, runs
        run = runs[0]
        assert run["prompt"] == '# Task\nsend "hi" now', run["prompt"]
        # The SEA's settings apply to this very run: its protocol is in
        # the system prompt and its tool sits beside the built-in ones.
        assert "NOTIFY-PROTOCOL" in run["system_prompt"]
        assert "_marker_tool" in run["tool_names"]
        assert "finish" in run["tool_names"]
        # The tab's state and the single history row show what the user typed.
        assert run["live_user_prompts"] == [raw_prompt], run
        rows = _history_rows()
        assert [r["task"] for r in rows] == [raw_prompt], rows
        assert rows[0]["parent_task_id"] in ("", None), rows

    def _seed_counting_sea(self, name: str, prompt_method: str) -> Path:
        """Register ``/<name>`` as a SEA that appends a line to a counter file each
        time its file is executed; return the counter file."""
        counter = Path(self.tmpdir) / f"{name}-executions.txt"
        _seed_seas_md(
            Path(self.tmpdir) / "user-seas", name,
            f"""
from pathlib import Path
from kiss.agents.seas.base.base_sea import BaseSea

with Path({str(counter)!r}).open("a") as fh:
    fh.write("executed\\n")

class Sea(BaseSea):
{prompt_method}
""",
        )
        return counter

    def test_bare_command_with_a_default_task_executes_the_sea_file_once(self) -> None:
        """A bare ``/xxx`` whose ``prompt()`` supplies the task runs that task,
        and the SEA file is executed once, not once to probe for the default
        and again to load the run's layers."""
        counter = self._seed_counting_sea(
            "counted",
            "    def prompt(self, task):\n"
            "        return task + 'Summarize the day.'\n",
        )
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            "/counted", work_dir=self.repo,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert len(runs) == 1, runs
        assert runs[0]["prompt"] == "# Task\nSummarize the day.", runs[0]["prompt"]
        assert counter.read_text().count("executed") == 1, counter.read_text()

    def test_bare_command_without_a_default_task_is_a_plain_prompt(self) -> None:
        """A bare ``/xxx`` whose SEA passes the empty task through runs as an
        ordinary prompt (the SEA is not applied), its file executed once."""
        counter = self._seed_counting_sea(
            "plain",
            "    def system_prompt(self, system_prompt):\n"
            "        return system_prompt + '\\\\n\\\\nPLAIN-PROTOCOL'\n",
        )
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            "/plain", work_dir=self.repo,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert len(runs) == 1, runs
        assert runs[0]["prompt"] == "# Task\n/plain", runs[0]["prompt"]
        assert "PLAIN-PROTOCOL" not in runs[0]["system_prompt"]
        assert counter.read_text().count("executed") == 1, counter.read_text()

    def test_bare_command_fallback_keeps_the_picker_sea_loaded_once(self) -> None:
        """Under a model-picker SEA, a bare ``/xxx`` without a default task runs
        as a plain prompt through the picker, whose file is executed once for
        the run: the fallback reuses the layers already loaded instead of
        loading the picker again."""
        picker_counter = self._seed_counting_sea(
            "picker",
            "    def register_as_model(self):\n"
            "        return True\n\n"
            "    def system_prompt(self, system_prompt):\n"
            "        return system_prompt + '\\\\n\\\\nPICKER-PROTOCOL'\n",
        )
        plain_counter = self._seed_counting_sea(
            "plain",
            "    def system_prompt(self, system_prompt):\n"
            "        return system_prompt + '\\\\n\\\\nPLAIN-PROTOCOL'\n",
        )
        # The picker verdict (``register_as_model``) executes the file
        # once, cached by file stamp; the run must add exactly one more.
        assert sea_commands.model_sea("picker") is not None
        executed_before = picker_counter.read_text().count("executed")
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            "/plain", model="picker", work_dir=self.repo,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert len(runs) == 1, runs
        assert runs[0]["prompt"] == "# Task\n/plain", runs[0]["prompt"]
        assert "PLAIN-PROTOCOL" not in runs[0]["system_prompt"]
        assert "PICKER-PROTOCOL" in runs[0]["system_prompt"]
        assert plain_counter.read_text().count("executed") == 1, plain_counter.read_text()
        assert picker_counter.read_text().count("executed") == executed_before + 1, (
            picker_counter.read_text()
        )

    def test_a_named_command_gets_its_help_but_a_bare_fallback_does_not(self) -> None:
        """A typed task naming ``/xxx`` mid-text gets its ``description()``
        after the task; a later bare ``/xxx`` in that chat, which falls back
        to a plain prompt, gets no help block, so the follow-up executes the
        SEA file once although the chat's earlier task names the command."""
        counter = self._seed_counting_sea(
            "plain",
            "    def description(self):\n"
            "        return 'Plain; /plain <text>.'\n",
        )
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        first = sorcar.run(
            "Later, use /plain on the notes.", work_dir=self.repo,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert first.success is True, first
        assert runs[0]["prompt"].endswith(
            "\n\n" + sea_commands.NAMED_COMMANDS_HEADING + "\n"
            + 'Each is an agent; run one with run_agent(agent="<name>", task="...") when the '
            "task calls for it.  What each does:\n- `/plain`: Plain; /plain <text>."
        ), runs[0]["prompt"]
        executed_before = counter.read_text().count("executed")
        follow_up = sorcar.run(
            "/plain", work_dir=self.repo, chat_id=first.chat_id,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert follow_up.success is True, follow_up
        assert len(runs) == 2, runs
        assert "Later, use /plain on the notes." in runs[1]["prompt"]
        assert runs[1]["prompt"].endswith("# Task (work on it now)\n\n/plain"), runs[1]["prompt"]
        assert counter.read_text().count("executed") == executed_before + 1, counter.read_text()
