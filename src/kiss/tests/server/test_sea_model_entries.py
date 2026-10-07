# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SEAs registered as models (``register_as_model()``) run every task of the tab.

``autorouter`` and ``bestrouter`` are not models.  The daemon lists every
SEA whose ``register_as_model()`` returns ``True`` in the picker
(``VSCodeServer._get_models`` through ``sea_commands.model_seas``) and,
when a run's model resolves to such an entry, ``_resolve_sea_model`` in
the task runner rewrites the run into an agent-script run of the SEA on
the model its ``settings()["model"]`` names (else the default model), with the
SEA's ``add_to_system_prompt()`` protocol added to the system prompt.
Runs that already name their agent (an explicit ``agentPath``, a ``/xxx``
slash command) keep it and only take the model from the pick.

Everything runs on a real local daemon (:class:`DaemonRunApiHarness`); only
the executor LLM loop is a stub recording the model and system prompt it
was handed.

Branches not covered here: which of ``orchestrator_model``'s three
returns (frontier candidate, best ranked runnable model, keyless default)
is taken depends on the provider keys and CLIs of the machine, so the
module-level test asserts the invariant that holds for whichever branch
applies instead of forcing each one.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.seas.autorouter import autorouter_sea
from kiss.agents.seas.autorouter.autorouter_sea import orchestrator_model
from kiss.agents.seas.bestrouter import bestrouter_sea
from kiss.agents.seas.sh import sh_sea
from kiss.agents.sorcar import local_endpoint, sea_commands
from kiss.agents.sorcar.cron_agent import load_jobs, save_jobs
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_available_models,
    get_default_model,
)
from kiss.core.vscode_config import save_config
from kiss.server import sorcar
from kiss.server.autocomplete import ranked_function_calling_models
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

AUTOROUTER = "autorouter"
BESTROUTER = "bestrouter"
AUTOROUTER_MARKER = "## Model routing protocol (autorouter)"
"""Heading of ``autorouter_sea.SYSTEM_PROMPT``."""
BESTROUTER_MARKER = "## Model routing protocol (bestrouter)"
"""Heading of ``bestrouter_sea.SYSTEM_PROMPT``."""


def test_orchestrator_model_is_runnable_whenever_the_picker_offers_autorouter() -> None:
    """First runnable frontier candidate, else the picker's best runnable model."""
    name = orchestrator_model()
    runnable = set(get_available_models())
    if not runnable:
        assert name == "No model"
        assert ranked_function_calling_models() == []
        return
    assert name in MODEL_INFO, name
    frontier = [n for n, _note in autorouter_sea.TIERS["frontier"] if n in runnable]
    ranked = ranked_function_calling_models()
    if frontier:
        assert name == frontier[0]
    elif ranked:
        assert name == ranked[0]
    if ranked:  # the picker offers the routers exactly when this list is non-empty
        assert name in runnable, name


def test_bundled_routers_are_the_model_picker_seas() -> None:
    """Both bundled routers register; no other bundled or third-party SEA does."""
    sea_commands._reset_for_tests()
    seas = sea_commands.model_seas()
    assert seas == {
        AUTOROUTER: Path(autorouter_sea.__file__).resolve(),
        BESTROUTER: Path(bestrouter_sea.__file__).resolve(),
    }
    assert sea_commands.model_sea("sh") is None
    assert sea_commands.model_sea("gpt-6-astra") is None
    assert sea_commands.model_sea("") is None


def test_bestrouter_protocol_names_its_models_literally() -> None:
    """The protocol fixes the primary and the review model by name and the 75% cap."""
    sea = bestrouter_sea.BestrouterSea()
    assert sea.register_as_model() is True
    assert sea.settings({"kind": "worker"}) == {
        "kind": "worker", "model": bestrouter_sea.PRIMARY_MODEL,
    }
    assert bestrouter_sea.PRIMARY_MODEL == "claude-fable-5-1"
    protocol = bestrouter_sea.SYSTEM_PROMPT
    assert sea.system_prompt("BASE") == "BASE\n\n" + protocol
    assert protocol.startswith(BESTROUTER_MARKER)
    flat = " ".join(protocol.split())
    for phrase in (
        "Use 'claude-fable-5-1' model for all tasks, including software development.",
        "Use gpt-6-astra (not codex) using `run_parallel` tool for a thorough read-only "
        "review and debugging of the other model's work.",
        "Use at most 75% of the task budget in gpt-6-astra",
        "ask the model not to invent new problems",
        "Use the model names literally without hallucinating new model names.",
    ):
        assert phrase in flat, phrase
    for name in (bestrouter_sea.PRIMARY_MODEL, bestrouter_sea.REVIEW_MODEL):
        assert name in MODEL_INFO, name
    assert "codex" not in bestrouter_sea.REVIEW_MODEL
    assert "bestrouter" in sea.description()
    # The protocol relies on run_parallel, so the SEA must not withhold it:
    # it appends to the system prompt, keeps every tool and leaves the
    # fan-out and prompt untouched.
    assert sea.settings({"allow_fan_out": True})["allow_fan_out"] is True
    assert sea.tools([print]) == [print]
    assert sea.prompt("the task") == "the task"


class SeaModelEntriesTest(DaemonRunApiHarness):
    """Picker entries and run rewriting of the ``autorouter`` / ``bestrouter`` picks."""

    def setUp(self) -> None:
        super().setUp()
        sea_commands._reset_for_tests()
        # Every ``autorouter`` pick schedules a cron job in the test ``$KISS_HOME``:
        # start each test from an empty store and leave none behind.
        save_jobs([])
        self.addCleanup(save_jobs, [])

    def tearDown(self) -> None:
        sea_commands._reset_for_tests()
        super().tearDown()

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
                "tool_names": sorted(t.__name__ for t in (kwargs.get("tools") or [])),
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
            endpoint_file=self.endpoint_file, timeout=60, **kwargs,
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

    def _write_user_sea(self, name: str, body: str) -> Path:
        """Create ``<tmpdir>/user_seas/<name>/<name>_sea.py``: a ``Sea`` class with *body*.

        *body* holds the class's methods (indented by four spaces); the
        class describes itself by *name*.
        """
        path = Path(self.tmpdir) / "user_seas" / name / f"{name}_sea.py"
        path.parent.mkdir(parents=True)
        path.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
            "class Sea(BaseSea):\n"
            f"    def description(self):\n        return {name!r}\n{body}",
            encoding="utf-8",
        )
        return path

    def _register_user_seas(self) -> None:
        """Point ``$KISS_HOME/SEAS.md`` at the user folder so its SEAs join the registry."""
        home = kiss_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "SEAS.md").write_text(f"{Path(self.tmpdir) / 'user_seas'}\n", encoding="utf-8")
        self.addCleanup((home / "SEAS.md").unlink, missing_ok=True)
        sea_commands.refresh_registry()

    def test_picker_lists_the_routers_first_with_a_cost_label(self) -> None:
        """The entries head the list, in one vendor group, with a label not a price."""
        event = self._models_event()
        assert event["models"], "precondition: at least one runnable model"
        names = [m["name"] for m in event["models"]]
        assert names[:2] == [AUTOROUTER, BESTROUTER], names[:4]
        for entry in event["models"][:2]:
            assert entry["vendor"] == "Router"
            assert entry["inp"] == 0 and entry["out"] == 0
            assert isinstance(entry["cost_label"], str) and entry["cost_label"]
        assert names.count(AUTOROUTER) == 1 and names.count(BESTROUTER) == 1

    def test_picker_lists_a_user_sea_that_registers_as_model(self) -> None:
        """A ``SEAS.md`` SEA joins the picker when ``register_as_model()`` is True, only then."""
        router = self._write_user_sea(
            "myrouter",
            "    def register_as_model(self):\n        return True\n",
        )
        self._write_user_sea(
            "notamodel",
            "    def register_as_model(self):\n        return False\n",
        )
        self._write_user_sea(
            "broken",
            "    def register_as_model(self):\n        raise RuntimeError('boom')\n",
        )
        self._write_user_sea(
            "notabool",
            "    def register_as_model(self):\n        return 'yes'\n",
        )
        self._write_user_sea("plain", "")
        self._register_user_seas()
        names = [m["name"] for m in self._models_event()["models"]]
        assert "myrouter" in names
        assert not {"notamodel", "broken", "notabool", "plain"} & set(names)
        assert sea_commands.model_sea("myrouter") == router
        # An edit that drops the registration is seen on the next lookup.
        router.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
            "class Sea(BaseSea):\n"
            "    def description(self):\n        return 'x'\n\n"
            "    def register_as_model(self):\n        return False\n",
            encoding="utf-8",
        )
        assert sea_commands.model_sea("myrouter") is None

    def test_picker_keeps_a_router_as_the_selected_model(self) -> None:
        """A persisted router pick survives the availability check."""
        vs = self.server._vscode_server
        for pick in (AUTOROUTER, BESTROUTER):
            with vs._state_lock:
                vs._default_model = pick
            event = self._models_event()
            assert event["selected"] == pick
            assert vs._default_model == pick

    def test_stale_pick_falls_back_to_a_real_model_not_a_router(self) -> None:
        """An unavailable cached pick lands on the first real model, never a router."""
        vs = self.server._vscode_server
        with vs._state_lock:
            vs._default_model = "stale-unavailable-model"
        event = self._models_event()
        assert event["selected"] not in (AUTOROUTER, BESTROUTER)
        assert event["selected"] in MODEL_INFO, event["selected"]
        assert event["models"][0]["name"] == AUTOROUTER

    def test_autorouter_model_runs_the_task_through_the_autorouter_sea(self) -> None:
        """The wire ``model`` field ``autorouter`` becomes an autorouter SEA run."""
        run = self._run("say hello", model=AUTOROUTER)
        assert run["model_name"] == orchestrator_model()
        assert run["model_name"] in MODEL_INFO
        assert run["system_prompt"].startswith("<identity>"), run["system_prompt"][:200]
        assert AUTOROUTER_MARKER in run["system_prompt"]
        assert BESTROUTER_MARKER not in run["system_prompt"]
        assert "say hello" in run["prompt"]

    def test_bestrouter_model_runs_the_task_on_its_primary_model_with_the_protocol(
        self,
    ) -> None:
        """The wire ``model`` field ``bestrouter`` runs on claude-fable-5-1 plus protocol."""
        run = self._run("say hello", model=BESTROUTER)
        assert run["model_name"] == "claude-fable-5-1"
        assert run["system_prompt"].startswith("<identity>"), run["system_prompt"][:200]
        assert BESTROUTER_MARKER in run["system_prompt"]
        assert "Use 'claude-fable-5-1' model for all tasks" in run["system_prompt"]
        assert AUTOROUTER_MARKER not in run["system_prompt"]
        assert "say hello" in run["prompt"]

    def test_protocol_is_added_after_the_callers_system_prompt_suffix(self) -> None:
        """``add_to_system_prompt()`` keeps the caller's ``append_to_system_prompt``."""
        run = self._run(
            "say hello", model=BESTROUTER, append_to_system_prompt="CALLER SUFFIX",
        )
        suffix = run["system_prompt"]
        assert suffix.index("CALLER SUFFIX") < suffix.index(BESTROUTER_MARKER)

    def test_tab_pick_router_runs_the_task_through_the_sea(self) -> None:
        """A run with no wire ``model`` takes the tab's pick, here ``bestrouter``."""
        vs = self.server._vscode_server
        with vs._state_lock:
            vs._default_model = BESTROUTER
        run = self._run("say hello")
        assert run["model_name"] == "claude-fable-5-1"
        assert BESTROUTER_MARKER in run["system_prompt"]

    def test_user_sea_without_model_setting_runs_on_the_default_model(self) -> None:
        """A registered SEA whose settings name no ``model`` runs on the default model."""
        self._write_user_sea(
            "myrouter",
            "    def register_as_model(self):\n        return True\n\n"
            "    def system_prompt(self, system_prompt):\n"
            "        return system_prompt + '\\n\\nMYROUTER PROTOCOL'\n",
        )
        self._register_user_seas()
        run = self._run("say hello", model="myrouter")
        assert run["model_name"] == get_default_model()
        assert "MYROUTER PROTOCOL" in run["system_prompt"]

    def test_explicit_agent_script_runs_on_top_of_the_router(self) -> None:
        """An ``agentPath`` run keeps the router pick as its base layer.

        The router's model is the model and its ``system_prompt`` runs
        first, so the script's own ``system_prompt`` receives the
        assembled prompt with the routing protocol already appended: a
        sub-agent or a slash command on a router tab follows the same
        routing rules.
        """
        sea = Path(self.tmpdir) / "plain_sea.py"
        sea.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
            "class Sea(BaseSea):\n"
            "    def system_prompt(self, system_prompt):\n"
            "        return system_prompt + '\\n\\nPLAIN SEA PROTOCOL'\n",
            encoding="utf-8",
        )
        run = self._run("say hello", model=AUTOROUTER, extension_agent_path=str(sea))
        assert run["model_name"] == orchestrator_model()
        assert run["system_prompt"].startswith("<identity>"), run["system_prompt"][:200]
        assert run["system_prompt"].index(AUTOROUTER_MARKER) < run["system_prompt"].index(
            "PLAIN SEA PROTOCOL"
        )
        run = self._run("say hello", model=BESTROUTER, extension_agent_path=str(sea))
        assert run["model_name"] == "claude-fable-5-1"
        assert run["system_prompt"].index(BESTROUTER_MARKER) < run["system_prompt"].index(
            "PLAIN SEA PROTOCOL"
        )

    def test_agent_script_blank_model_setting_still_gets_a_real_model(self) -> None:
        """A ``settings()["model"]`` of ``""`` means "the tab's pick" — never the router."""
        vs = self.server._vscode_server
        with vs._state_lock:
            vs._default_model = AUTOROUTER
        sea = Path(self.tmpdir) / "blankmodel_sea.py"
        sea.write_text(
            """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"model": ""}

    def system_prompt(self, system_prompt):
        return "BLANK MODEL SEA"
""",
            encoding="utf-8",
        )
        run = self._run("say hello", extension_agent_path=str(sea))
        assert run["model_name"] == orchestrator_model()
        assert run["system_prompt"].startswith("BLANK MODEL SEA")

    def test_agent_script_naming_a_picker_as_its_model_runs_on_a_real_model(self) -> None:
        """``settings()["model"]`` may name a picker entry: it is resolved, never forwarded.

        The tab's own picker keeps its model; another picker is executed
        once and contributes its model and its ``on_picked_as_model``
        hook — a run never names a picker SEA as its LLM.
        """
        sea = Path(self.tmpdir) / "pickbest_sea.py"
        sea.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
            "class Sea(BaseSea):\n"
            "    def settings(self, settings):\n"
            f"        return settings | {{'model': '{BESTROUTER}'}}\n\n"
            "    def system_prompt(self, system_prompt):\n"
            "        return system_prompt + '\\n\\nPICKS BESTROUTER'\n",
            encoding="utf-8",
        )
        run = self._run("say hello", model="gpt-6-astra", extension_agent_path=str(sea))
        assert run["model_name"] == "claude-fable-5-1"
        assert "PICKS BESTROUTER" in run["system_prompt"]
        # The same name under the bestrouter tab: the picker's model, once.
        run = self._run("say hello", model=BESTROUTER, extension_agent_path=str(sea))
        assert run["model_name"] == "claude-fable-5-1"
        assert run["system_prompt"].count(BESTROUTER_MARKER) == 1
        assert "PICKS BESTROUTER" in run["system_prompt"]

    def test_agent_script_extending_the_tab_picker_runs_the_picker_once(self) -> None:
        """A script that subclasses the tab's picker gets ONE picker layer, not two.

        The base class's ``system_prompt`` runs first (base first), so the
        router's protocol precedes the subclass's own.
        """
        sea = Path(self.tmpdir) / "onbest_sea.py"
        sea.write_text(
            "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
            f"class Sea(sea_class('{BESTROUTER}')):\n"
            "    def system_prompt(self, system_prompt):\n"
            "        return system_prompt + '\\n\\nONBEST PROTOCOL'\n",
            encoding="utf-8",
        )
        run = self._run("say hello", model=BESTROUTER, extension_agent_path=str(sea))
        assert run["model_name"] == "claude-fable-5-1"
        assert run["system_prompt"].count(BESTROUTER_MARKER) == 1
        assert run["system_prompt"].index(BESTROUTER_MARKER) < run["system_prompt"].index(
            "ONBEST PROTOCOL"
        )

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

    def test_raising_settings_of_a_picked_sea_fails_the_run(self) -> None:
        """A picked SEA whose ``settings()`` raises stops the task with the diagnostic."""
        self._write_user_sea(
            "badmodel",
            "    def register_as_model(self):\n        return True\n\n"
            "    def settings(self, settings):\n        raise RuntimeError('no model today')\n",
        )
        self._register_user_seas()
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        events: list[dict[str, Any]] = []
        self._raw_daemon_run({"model": "badmodel", "prompt": "say hello"}, events)
        results = [e for e in events if e.get("type") == "result"]
        assert results, events
        assert "settings() of agent script" in str(results[-1]), results[-1]
        assert "badmodel_sea.py' raised: RuntimeError: no model today" in str(results[-1])
        assert runs == []

    def test_persisted_router_pick_is_not_a_model_for_direct_runs(self) -> None:
        """Outside the daemon, a persisted router pick counts as no pick."""
        for pick in (AUTOROUTER, BESTROUTER):
            save_config({"last_model": pick})
            assert SorcarAgent._resolve_model_name(None) == get_default_model()
        assert SorcarAgent._resolve_model_name("gpt-6-astra") == "gpt-6-astra"
        real = orchestrator_model()
        save_config({"last_model": real})
        assert SorcarAgent._resolve_model_name(None) == real

    def test_slash_command_keeps_its_own_agent(self) -> None:
        """``/sh ...`` runs the sh SEA directly as the tab's agent, on top of the pick.

        The run's ``agentPath`` is the ``/sh`` SEA (its ``bash`` tool
        profile shapes the run), the LLM's task is the trailing ``echo
        hi`` (no ``run_agent`` relay), and the bestrouter pick is the
        base layer: its model.  The sh SEA's ``system_prompt`` replaces
        the whole assembled prompt, the router's appended protocol
        included, so the Bash-only worker runs on its own prompt.
        """
        run = self._run("/sh echo hi", model=BESTROUTER)
        assert run["model_name"] == "claude-fable-5-1"
        assert BESTROUTER_MARKER not in run["system_prompt"]
        assert run["prompt"] == "# Task\necho hi", run["prompt"]
        assert run["system_prompt"].startswith(sh_sea.SYSTEM_PROMPT), run["system_prompt"]
        assert run["tool_names"] == ["Bash", "finish"], run["tool_names"]

    def test_other_models_are_left_untouched(self) -> None:
        """A real model name is neither rewritten nor given an agent script."""
        model = orchestrator_model()
        run = self._run("say hello", model=model)
        assert run["model_name"] == model
        assert AUTOROUTER_MARKER not in run["system_prompt"]
        assert BESTROUTER_MARKER not in run["system_prompt"]

    def test_autorouter_pick_schedules_the_weekly_rsi7d_job_once(self) -> None:
        """Every run picked on ``autorouter`` makes sure one enabled weekly job exists.

        The first pick creates it in the real cron store of ``$KISS_HOME``;
        the second finds it enabled and leaves the store alone.  The test
        repo is not a KISS checkout, so the job runs in a scratch directory.
        """
        with self.assertLogs("kiss.sea_commands", level="INFO") as logs:
            self._run("say hello", model=AUTOROUTER)
        assert any("picked as model: created:" in line for line in logs.output), logs.output
        (job,) = load_jobs()
        assert job["name"] == autorouter_sea.RSI7D_JOB_NAME
        assert job["enabled"] is True
        assert job["schedule"] == autorouter_sea.RSI7D_JOB_SCHEDULE
        assert job["work_dir"] == "" and job["use_worktree"] is False
        assert f"task         = {autorouter_sea.RSI7D_TASK!r}" in job["prompt"]
        with self.assertLogs("kiss.sea_commands", level="INFO") as logs:
            self._run("say hello again", model=AUTOROUTER)
        assert any(
            "picked as model: exists:" in line and job["id"] in line for line in logs.output
        ), logs.output
        assert load_jobs() == [job]

    def test_picked_sea_hook_runs_once_with_the_work_dir_and_never_fails_the_run(self) -> None:
        """``on_picked_as_model(work_dir)`` fires once per picked run; a raising hook is logged.

        The hook does not run for a SEA that is dispatched as an explicit
        agent script on a real model, only for the SEA picked as the model.
        """
        calls = Path(self.tmpdir) / "hook_calls.txt"
        hooked = self._write_user_sea(
            "hooked",
            "    def register_as_model(self):\n        return True\n\n"
            "    def on_picked_as_model(self, work_dir):\n"
            f"        with open({str(calls)!r}, 'a') as f:\n"
            "            f.write(work_dir + '\\n')\n"
            "        return 'hook ran in ' + work_dir\n",
        )
        self._write_user_sea(
            "badhook",
            "    def register_as_model(self):\n        return True\n\n"
            "    def on_picked_as_model(self, work_dir):\n"
            "        raise RuntimeError('hook exploded')\n",
        )
        self._register_user_seas()
        with self.assertLogs("kiss.sea_commands", level="INFO") as logs:
            run = self._run("say hello", model="hooked")
        assert run["model_name"] == get_default_model()
        assert calls.read_text(encoding="utf-8") == f"{self.repo}\n"
        assert any(f"picked as model: hook ran in {self.repo}" in line for line in logs.output)
        # Dispatched as an explicit agent script on a real model: not a pick, no hook.
        self._run("say hello", model=orchestrator_model(), extension_agent_path=str(hooked))
        assert calls.read_text(encoding="utf-8") == f"{self.repo}\n"
        with self.assertLogs("kiss.sea_commands", level="WARNING") as logs:
            run = self._run("say hello", model="badhook")
        assert run["model_name"] == get_default_model()
        assert any("on_picked_as_model() failed" in line for line in logs.output), logs.output
        assert any("hook exploded" in line for line in logs.output), logs.output

    def _send(self, cmd: dict[str, Any], done: Callable[[], bool]) -> None:
        """Send one raw command over the local endpoint and wait until *done()* holds."""
        ws = local_endpoint.connect(Path(self.endpoint_file), open_timeout=10)
        try:
            ws.send(json.dumps(cmd))
            deadline = time.monotonic() + 10
            while not done() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert done(), cmd
        finally:
            ws.close()

    def test_selecting_autorouter_in_the_picker_schedules_the_weekly_job_without_a_task(
        self,
    ) -> None:
        """A ``selectModel`` pick alone fires the hook with the tab's work dir; a real model
        does not.

        The test repo is not a KISS checkout, so the scheduled job has no work dir.
        """
        vs = self.server._vscode_server
        tab_id = f"pick-{uuid.uuid4().hex}"
        real = orchestrator_model()
        self._send(
            {"type": "openTab", "tabId": tab_id, "title": "t", "workDir": self.repo},
            lambda: vs.tab_registry.has_tab(tab_id),
        )
        self._send(
            {"type": "selectModel", "tabId": tab_id, "model": real},
            lambda: vs._tab_models.get(tab_id) == real,
        )
        time.sleep(0.5)
        assert load_jobs() == []
        self._send(
            {"type": "selectModel", "tabId": tab_id, "model": AUTOROUTER},
            lambda: bool(load_jobs()),
        )
        (job,) = load_jobs()
        assert job["name"] == autorouter_sea.RSI7D_JOB_NAME and job["enabled"] is True
        assert job["work_dir"] == autorouter_sea.kiss_checkout(self.repo) == ""
        assert vs._tab_models[tab_id] == AUTOROUTER

    def test_blocking_hook_is_abandoned_and_the_run_goes_on(self) -> None:
        """A hook that never returns holds the run for the timeout only, then is logged."""
        self._write_user_sea(
            "stuckhook",
            "    def register_as_model(self):\n        return True\n\n"
            "    def on_picked_as_model(self, work_dir):\n"
            "        import threading\n"
            "        threading.Event().wait()\n"
            "        return 'never'\n",
        )
        self._register_user_seas()
        original = sea_commands.PICKED_HOOK_TIMEOUT_SECONDS
        sea_commands.PICKED_HOOK_TIMEOUT_SECONDS = 0.5
        self.addCleanup(setattr, sea_commands, "PICKED_HOOK_TIMEOUT_SECONDS", original)
        started = time.monotonic()
        with self.assertLogs("kiss.sea_commands", level="WARNING") as logs:
            run = self._run("say hello", model="stuckhook")
        assert run["model_name"] == get_default_model()
        assert time.monotonic() - started < 30
        assert any("still running after 0 s" in line for line in logs.output), logs.output
