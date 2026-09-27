# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``autorouter`` model-picker entry runs every task through the autoroute SEA.

``autorouter`` is not a model.  The daemon lists it in the picker
(``VSCodeServer._get_models``) and, when a run's model resolves to it,
``_resolve_autorouter`` in the task runner rewrites the run into an
agent-script run of ``src/kiss/agents/seas/autoroute_sea.py`` on the
SEA's orchestrator model.  Runs that already name their agent (an
explicit ``agentPath``, a ``/xxx`` slash command) keep it and only take
the orchestrator model from the pick.

Everything runs on a real UDS daemon (:class:`DaemonRunApiHarness`); only
the executor LLM loop is a stub recording the model and system prompt it
was handed.

Branches not covered here: which of ``orchestrator_model``'s three
returns (frontier candidate, best ranked runnable model, keyless default)
is taken depends on the provider keys and CLIs of the machine, so the
module-level test asserts the invariant that holds for whichever branch
applies instead of forcing each one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.seas import autoroute_sea
from kiss.agents.seas.autoroute_sea import orchestrator_model
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import (
    AUTOROUTER,
    MODEL_INFO,
    get_available_models,
    get_default_model,
)
from kiss.core.vscode_config import save_config
from kiss.server import sorcar
from kiss.server.autocomplete import ranked_function_calling_models
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

pytestmark = requires_unix_sockets

PROTOCOL_MARKER = "You are the autoroute agent."
"""First sentence of ``autoroute_sea.SYSTEM_PROMPT``."""


def test_orchestrator_model_is_runnable_whenever_the_picker_offers_autorouter() -> None:
    """First runnable frontier candidate, else the picker's best runnable model."""
    name = orchestrator_model()
    assert name in MODEL_INFO, name
    runnable = set(get_available_models())
    frontier = [n for n, _note in autoroute_sea.TIERS["frontier"] if n in runnable]
    ranked = ranked_function_calling_models()
    if frontier:
        assert name == frontier[0]
    elif ranked:
        assert name == ranked[0]
    if ranked:  # the picker offers ``autorouter`` exactly when this list is non-empty
        assert name in runnable, name


class AutorouterModelEntryTest(DaemonRunApiHarness):
    """Picker entry and run rewriting of the ``autorouter`` pick."""

    def _record_runs(self, runs: list[dict[str, Any]]) -> None:
        """Replace the executor LLM loop with a stub recording model and prompts."""

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
                "model_name": kwargs.get("model_name"),
                "system_prompt": str(kwargs.get("system_prompt") or ""),
                "prompt": str(arguments["task_description"]),
            })
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1, total_tokens=1, cost="$0.0001",
                )
            return raw

        KISSAgent.run = stub_run  # type: ignore[assignment,method-assign]

    def _run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            prompt, work_dir=self.repo, use_worktree=False, auto_commit=False,
            sock_path=self.sock_path, timeout=60, **kwargs,
        )
        assert result.success is True, result
        assert len(runs) == 1, runs
        return runs[0]

    def _models_event(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        vs = self.server._vscode_server
        original = vs.printer.broadcast

        def capture(event: dict[str, Any]) -> None:
            events.append(event)

        vs.printer.broadcast = capture  # type: ignore[method-assign]
        try:
            vs._get_models()
        finally:
            vs.printer.broadcast = original  # type: ignore[method-assign]
        models = [e for e in events if e.get("type") == "models"]
        assert len(models) == 1, events
        return models[0]

    def test_picker_lists_autorouter_first_with_a_cost_label(self) -> None:
        """The entry heads the list, in its own vendor group, with a label not a price."""
        event = self._models_event()
        assert event["models"], "precondition: at least one runnable model"
        entry = event["models"][0]
        assert entry["name"] == AUTOROUTER
        assert entry["vendor"] == "Autoroute"
        assert entry["inp"] == 0 and entry["out"] == 0
        assert isinstance(entry["cost_label"], str) and entry["cost_label"]
        assert [m["name"] for m in event["models"]].count(AUTOROUTER) == 1

    def test_picker_keeps_autorouter_as_the_selected_model(self) -> None:
        """A persisted ``autorouter`` pick survives the availability check."""
        vs = self.server._vscode_server
        with vs._state_lock:
            vs._default_model = AUTOROUTER
        event = self._models_event()
        assert event["selected"] == AUTOROUTER
        assert vs._default_model == AUTOROUTER

    def test_stale_pick_falls_back_to_a_real_model_not_autorouter(self) -> None:
        """An unavailable cached pick lands on the first real model, never the router."""
        vs = self.server._vscode_server
        with vs._state_lock:
            vs._default_model = "stale-unavailable-model"
        event = self._models_event()
        assert event["selected"] != AUTOROUTER
        assert event["selected"] in MODEL_INFO, event["selected"]
        assert event["models"][0]["name"] == AUTOROUTER

    def test_autorouter_model_runs_the_task_through_the_autoroute_sea(self) -> None:
        """The wire ``model`` field ``autorouter`` becomes an autoroute SEA run."""
        run = self._run("say hello", model=AUTOROUTER)
        assert run["model_name"] == orchestrator_model()
        assert run["model_name"] in MODEL_INFO
        assert run["system_prompt"].startswith(PROTOCOL_MARKER), run["system_prompt"][:200]
        assert "say hello" in run["prompt"]

    def test_tab_pick_autorouter_runs_the_task_through_the_autoroute_sea(self) -> None:
        """A run with no wire ``model`` takes the tab's pick, here ``autorouter``."""
        vs = self.server._vscode_server
        with vs._state_lock:
            vs._default_model = AUTOROUTER
        run = self._run("say hello")
        assert run["model_name"] == orchestrator_model()
        assert run["system_prompt"].startswith(PROTOCOL_MARKER)

    def test_explicit_agent_script_keeps_its_agent_on_the_orchestrator_model(self) -> None:
        """An ``agentPath`` run only takes the model from the ``autorouter`` pick."""
        sea = Path(self.tmpdir) / "plain_sea.py"
        sea.write_text(
            'def system_prompt() -> str:\n    return "PLAIN SEA PROMPT"\n', encoding="utf-8"
        )
        run = self._run("say hello", model=AUTOROUTER, extension_agent_path=str(sea))
        assert run["model_name"] == orchestrator_model()
        assert run["system_prompt"].startswith("PLAIN SEA PROMPT")
        assert PROTOCOL_MARKER not in run["system_prompt"]

    def test_agent_script_model_getter_returning_blank_still_gets_a_real_model(self) -> None:
        """A ``model()`` getter returning ``""`` means "the tab's pick" — never the router."""
        vs = self.server._vscode_server
        with vs._state_lock:
            vs._default_model = AUTOROUTER
        sea = Path(self.tmpdir) / "blankmodel_sea.py"
        sea.write_text(
            'def model() -> str:\n    return ""\n'
            'def system_prompt() -> str:\n    return "BLANK MODEL SEA"\n',
            encoding="utf-8",
        )
        run = self._run("say hello", extension_agent_path=str(sea))
        assert run["model_name"] == orchestrator_model()
        assert run["system_prompt"].startswith("BLANK MODEL SEA")

    def test_malformed_agent_path_is_still_rejected(self) -> None:
        """A blank ``agentPath`` is not "no agent": it fails as it always did.

        The Python client validates the path itself, so the malformed
        field is sent raw, as an arbitrary client would.
        """
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        events: list[dict[str, Any]] = []
        self._raw_daemon_run({"model": AUTOROUTER, "agentPath": "   "}, events)
        results = [e for e in events if e.get("type") == "result"]
        assert results, events
        assert "AgentFileError" in str(results[-1]), results[-1]
        assert runs == []

    def test_persisted_autorouter_pick_is_not_a_model_for_direct_runs(self) -> None:
        """Outside the daemon, a persisted ``autorouter`` pick counts as no pick."""
        save_config({"last_model": AUTOROUTER})
        assert SorcarAgent._resolve_model_name(None) == get_default_model()
        assert SorcarAgent._resolve_model_name("gpt-6-astra") == "gpt-6-astra"
        real = orchestrator_model()
        save_config({"last_model": real})
        assert SorcarAgent._resolve_model_name(None) == real

    def test_slash_command_keeps_its_own_agent(self) -> None:
        """``/sh ...`` names its agent; the pick only supplies the model."""
        run = self._run("/sh echo hi", model=AUTOROUTER)
        assert run["model_name"] == orchestrator_model()
        assert PROTOCOL_MARKER not in run["system_prompt"]
        assert "run_agent" in run["prompt"], run["prompt"]

    def test_other_models_are_left_untouched(self) -> None:
        """A real model name is neither rewritten nor given an agent script."""
        model = orchestrator_model()
        run = self._run("say hello", model=model)
        assert run["model_name"] == model
        assert PROTOCOL_MARKER not in run["system_prompt"]
