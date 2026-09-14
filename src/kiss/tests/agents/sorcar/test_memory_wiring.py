# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the SorcarAgent persistent-memory wiring.

The ``use_memory`` config flag (on by default, overridable by the
``KISS_USE_MEMORY`` environment variable) gives every SorcarAgent run the
seven ``memory_*`` tools and the ``MEMORY_PROTOCOL`` system-prompt block,
with pages under ``$KISS_HOME/memories``.  Offline tests cover the settings
resolution; live tests run a real agent on ``claude-haiku-4-5`` and
inspect the actual system prompt the run installed on the live model
(``model_config["system_instruction"]``, where KISSAgent.run puts it).
"""

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _memory_root_for_run,
    _memory_settings,
    run_tasks_parallel,
)
from kiss.core.memoryfield.tools import MEMORY_PROTOCOL

live_api = pytest.mark.live_api
requires_keys = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY")),
    reason="ANTHROPIC_API_KEY and OPENAI_API_KEY needed for live memory wiring tests",
)


def _home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point $KISS_HOME at an isolated per-test directory."""
    home = tmp_path / "kiss-home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    monkeypatch.delenv("KISS_USE_MEMORY", raising=False)
    return home


def _write_config(home: Path, cfg: dict[str, Any]) -> None:
    (home / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


class TestMemorySettings:
    def test_default_is_on_with_kiss_home_memories(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        enabled, root = _memory_settings()
        assert enabled is True
        assert root == home / "memories"

    def test_config_flag_disables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": False})
        enabled, root = _memory_settings()
        assert enabled is False
        assert root == home / "memories"

    def test_config_memory_dir_overrides_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(
            home, {"use_memory": True, "memory_dir": str(tmp_path / "elsewhere")}
        )
        enabled, root = _memory_settings()
        assert enabled is True
        assert root == tmp_path / "elsewhere"

    def test_memory_dir_expands_tilde(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"memory_dir": "~/my-memories"})
        _, root = _memory_settings()
        assert root == Path.home() / "my-memories"

    def test_env_var_enables_over_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": False})
        monkeypatch.setenv("KISS_USE_MEMORY", "1")
        assert _memory_settings()[0] is True

    @pytest.mark.parametrize("value", ["0", "false", "No", "OFF"])
    def test_env_var_disables_over_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": True})
        monkeypatch.setenv("KISS_USE_MEMORY", value)
        assert _memory_settings()[0] is False

    def test_blank_env_var_falls_back_to_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": True})
        monkeypatch.setenv("KISS_USE_MEMORY", "  ")
        assert _memory_settings()[0] is True


class TestMemoryRootForRun:
    """The per-run gates that precede the user's use_memory setting."""

    def _enable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": True})
        return home

    def test_enabled_api_model_gets_memory_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = self._enable(monkeypatch, tmp_path)
        root = _memory_root_for_run(True, None, "claude-haiku-4-5")
        assert root == home / "memories"

    def test_config_off_yields_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": False})
        assert _memory_root_for_run(True, None, "claude-haiku-4-5") is None

    def test_default_on_yields_memory_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        root = _memory_root_for_run(True, None, "claude-haiku-4-5")
        assert root == home / "memories"

    def test_append_basic_tools_false_gates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._enable(monkeypatch, tmp_path)
        assert _memory_root_for_run(False, None, "claude-haiku-4-5") is None

    def test_docker_image_gates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._enable(monkeypatch, tmp_path)
        assert _memory_root_for_run(True, "python:3.12", "claude-haiku-4-5") is None

    @pytest.mark.parametrize("model", ["cc/claude-fable-5", "codex/gpt-5.3-codex"])
    def test_run_to_completion_models_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: str
    ) -> None:
        self._enable(monkeypatch, tmp_path)
        assert _memory_root_for_run(True, None, model) is None

    def test_caller_system_instruction_gates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A caller-supplied system_instruction replaces the composed prompt
        (KISSAgent.run only setdefault-s it), so MEMORY_PROTOCOL would never
        reach the model and the tools must not be registered without it."""
        self._enable(monkeypatch, tmp_path)
        root = _memory_root_for_run(
            True, None, "claude-haiku-4-5", caller_system_instruction=True
        )
        assert root is None


class TestUseMemoryOverride:
    """The per-run ``use_memory`` override (``SorcarAgent.run``'s new
    parameter, ``kiss.server.sorcar.run``'s ``useMemory`` wire field).

    A boolean override is the caller's per-run choice: it wins over
    both the stored ``use_memory`` setting and the ``KISS_USE_MEMORY``
    environment variable, but never bypasses the hard safety gates.
    ``None`` keeps the ``_memory_settings`` resolution.
    """

    def test_true_overrides_config_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": False})
        root = _memory_root_for_run(
            True, None, "claude-haiku-4-5", use_memory_override=True
        )
        assert root == home / "memories"

    def test_false_overrides_default_on(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _home(monkeypatch, tmp_path)
        root = _memory_root_for_run(
            True, None, "claude-haiku-4-5", use_memory_override=False
        )
        assert root is None

    def test_true_overrides_env_disable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        monkeypatch.setenv("KISS_USE_MEMORY", "0")
        root = _memory_root_for_run(
            True, None, "claude-haiku-4-5", use_memory_override=True
        )
        assert root == home / "memories"

    def test_false_overrides_env_enable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": False})
        monkeypatch.setenv("KISS_USE_MEMORY", "1")
        root = _memory_root_for_run(
            True, None, "claude-haiku-4-5", use_memory_override=False
        )
        assert root is None

    def test_none_falls_back_to_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": False})
        assert (
            _memory_root_for_run(
                True, None, "claude-haiku-4-5", use_memory_override=None
            )
            is None
        )
        _write_config(home, {"use_memory": True})
        assert (
            _memory_root_for_run(
                True, None, "claude-haiku-4-5", use_memory_override=None
            )
            == home / "memories"
        )

    def test_true_never_bypasses_hard_gates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An explicit True stays gated: stripped basic tools, Docker,
        run-to-completion CLI models, and a caller system_instruction
        all keep the run memory-free."""
        _home(monkeypatch, tmp_path)
        gated = [
            _memory_root_for_run(
                False, None, "claude-haiku-4-5", use_memory_override=True
            ),
            _memory_root_for_run(
                True, "python:3.12", "claude-haiku-4-5",
                use_memory_override=True,
            ),
            _memory_root_for_run(
                True, None, "cc/claude-fable-5", use_memory_override=True
            ),
            _memory_root_for_run(
                True, None, "claude-haiku-4-5",
                caller_system_instruction=True, use_memory_override=True,
            ),
        ]
        assert gated == [None, None, None, None]


class TestFanOutForwardsUseMemory:
    """``run_tasks_parallel`` forwards the parent's ``use_memory`` override
    to every sub-agent, so one explicit override governs the whole task
    tree while ``None`` lets each child fall back to the config default.

    Like the daemon suites in ``tests/server``, the only replaced
    boundary is the LLM itself: ``SorcarAgent``'s parent ``run`` is
    swapped for a recorder while the real fan-out pipeline (chat
    allocation, task persistence, the memory decision in
    ``SorcarAgent.run``) executes against an isolated ``KISS_HOME``.
    """

    @pytest.fixture()
    def recorded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> list[dict[str, Any]]:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": True})
        monkeypatch.setattr(_persistence, "_KISS_DIR", home)
        monkeypatch.setattr(_persistence, "_DB_PATH", home / "sorcar.db")
        monkeypatch.setattr(_persistence, "_db_conn", None)
        (tmp_path / "work").mkdir(exist_ok=True)
        seen: list[dict[str, Any]] = []
        parent_class = cast(Any, SorcarAgent.__mro__[1])

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            seen.append({
                "override": self_agent._use_memory_override,
                "memory_tools": self_agent._memory_tools,
            })
            return "success: true\nis_continue: false\nsummary: ok\n"

        monkeypatch.setattr(parent_class, "run", stub_run)
        return seen

    def test_false_reaches_every_sub_agent(
        self, recorded: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        results = run_tasks_parallel(
            ["record a", "record b"],
            max_workers=2,
            model_name="claude-haiku-4-5",
            work_dir=str(tmp_path / "work"),
            use_memory=False,
        )
        assert len(results) == 2
        assert [entry["override"] for entry in recorded] == [False, False]
        assert all(entry["memory_tools"] is None for entry in recorded)

    def test_default_none_lets_children_use_config(
        self, recorded: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        results = run_tasks_parallel(
            ["record"],
            max_workers=1,
            model_name="claude-haiku-4-5",
            work_dir=str(tmp_path / "work"),
        )
        assert len(results) == 1
        assert recorded[0]["override"] is None
        assert recorded[0]["memory_tools"] is not None

    def test_run_parallel_tool_forwards_parent_override(
        self, recorded: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """The ``run_parallel`` tool closure passes the PARENT's stored
        override on, so an LLM-triggered fan-out inherits it too."""
        agent = SorcarAgent("memory-fanout-parent")
        agent._use_web_tools = False
        agent._use_memory_override = False
        agent.work_dir = str(tmp_path / "work")
        run_parallel = next(
            t
            for t in agent._get_tools()
            if getattr(t, "__name__", "") == "run_parallel"
        )
        run_parallel('["record via tool"]', max_workers="1")
        assert [entry["override"] for entry in recorded] == [False]
        assert recorded[0]["memory_tools"] is None


def _run_capturing_system_prompt(
    agent: SorcarAgent,
    prompt: str,
    work_dir: Path,
    append_basic_tools: bool = True,
) -> tuple[str, str]:
    """Run *agent* on *prompt* with claude-haiku-4-5, returning (result, system prompt).

    The composed system prompt is captured from the run's live executor
    session through the documented ``llm_call_hook`` extension point: the
    hook fires on the executor while it is current, and
    ``RelentlessAgent`` exposes that executor as ``_current_executor``,
    whose model carries the system prompt out-of-band in
    ``model_config["system_instruction"]``.
    """
    seen_prompts: list[str] = []

    def hook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        executor = getattr(agent, "_current_executor", None)
        model = getattr(executor, "model", None)
        config = getattr(model, "model_config", None) or {}
        seen_prompts.append(str(config.get("system_instruction", "")))
        return messages

    result = agent.run(
        model_name="claude-haiku-4-5",
        prompt_template=prompt,
        work_dir=str(work_dir),
        web_tools=False,
        is_parallel=False,
        max_steps=10,
        verbose=False,
        append_basic_tools=append_basic_tools,
        llm_call_hook=hook,
    )
    return result, "\n".join(seen_prompts)


@live_api
@requires_keys
class TestMemoryWiringLive:
    def test_memory_on_stores_page_and_recalls_in_new_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": True, "classify_tasks": False})
        work = tmp_path / "work"
        work.mkdir()

        writer = SorcarAgent("memory-writer")
        _, system = _run_capturing_system_prompt(
            writer,
            "Store this durable fact in your persistent memory (one page, name it "
            "yourself), then finish: the internal artifact registry for project "
            "Heron is at registry.heron.internal:7481. Do not do anything else.",
            work,
        )
        assert MEMORY_PROTOCOL in system
        pages = list((home / "memories").glob("*.md"))
        assert pages, "agent wrote no memory page"
        assert any("7481" in p.read_text(encoding="utf-8") for p in pages)

        reader = SorcarAgent("memory-reader")
        result, _ = _run_capturing_system_prompt(
            reader,
            "Answer from your persistent memory: on which host and port is the "
            "project Heron artifact registry? Search memory first, then finish "
            "with the answer in the summary.",
            work,
        )
        assert "7481" in result

    def test_memory_off_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": False, "classify_tasks": False})
        work = tmp_path / "work"
        work.mkdir()
        agent = SorcarAgent("memory-off")
        result, system = _run_capturing_system_prompt(
            agent,
            "If a tool named memory_search is available to you, finish with the "
            "summary HAVE-MEMORY; otherwise finish with the summary NO-MEMORY. "
            "Do not call any other tool.",
            work,
        )
        assert MEMORY_PROTOCOL not in system
        assert "NO-MEMORY" in result
        assert not (home / "memories").exists()

    def test_caller_system_instruction_disables_memory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A run whose model_config carries its own system_instruction gets
        neither MEMORY_PROTOCOL (which that instruction suppresses) nor the
        memory_* tools, even though memory is on by default."""
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"classify_tasks": False})
        work = tmp_path / "work"
        work.mkdir()
        agent = SorcarAgent("memory-caller-prompt")
        result = agent.run(
            model_name="claude-haiku-4-5",
            prompt_template=(
                "If a tool named memory_search is available to you, finish with "
                "the summary HAVE-MEMORY; otherwise finish with the summary "
                "NO-MEMORY. Do not call any other tool."
            ),
            model_config={
                "system_instruction": "Follow the user's instructions exactly."
            },
            work_dir=str(work),
            web_tools=False,
            is_parallel=False,
            max_steps=10,
            verbose=False,
        )
        assert "NO-MEMORY" in result
        assert not (home / "memories").exists()

    def test_append_basic_tools_false_disables_memory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = _home(monkeypatch, tmp_path)
        _write_config(home, {"use_memory": True, "classify_tasks": False})
        work = tmp_path / "work"
        work.mkdir()
        agent = SorcarAgent("memory-stripped")
        _, system = _run_capturing_system_prompt(
            agent,
            "Call finish immediately with success=True and summary 'done'.",
            work,
            append_basic_tools=False,
        )
        assert MEMORY_PROTOCOL not in system
        assert not (home / "memories").exists()
