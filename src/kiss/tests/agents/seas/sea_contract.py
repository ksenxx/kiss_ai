# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared assertions for the bundled SEAs' class contract.

A SEA is one :class:`~kiss.agents.seas.base.base_sea.BaseSea` subclass
that configures its run through ``settings()`` and shapes it through
the other contract methods (``prompt``, ``system_prompt``, ``tools``,
the hooks).  The module-level getters of the old contract
(``settings()``, ``system_prompt()``, ``add_to_tools()``, ...) and the
per-field getters before them are no longer read by anything: a
bundled SEA that still defined one would ship a silently ignored
function, so every SEA contract test asserts none of those names
exists, on the module and on the class.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea, WorkerSea
from kiss.agents.sorcar import sea_commands, sea_lint
from kiss.agents.sorcar.sea_settings import WORKER_DEFAULTS, resolve_settings
from kiss.tests.agents.seas.conftest import run_sea

REMOVED_GETTERS: tuple[str, ...] = sea_lint.REMOVED_GETTERS + (
    "chat_id",
    "model_config",
    "auto_classify",
    "allow_fan_out",
    "use_web_tools",
    "use_memory",
    "if_append_basic_tools",
)
"""Names the old contracts read as getters; none is read any more, on a module or a class.

The lint's own list (the getters its ``stale-docstring`` rule looks for)
plus the per-field getters it does not look for in docstrings.
"""

MODULE_GETTERS: tuple[str, ...] = (
    "description",
    "settings",
    "prompt",
    "system_prompt",
    "tools",
    "tool_call_hook",
    "llm_call_hook",
    "register_as_model",
    "on_picked_as_model",
)
"""The contract methods: defined on the SEA class, never as module-level functions."""


def sea_class_of(module: ModuleType) -> type[BaseSea]:
    """Return the one :class:`BaseSea` subclass *module* defines itself."""
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, BaseSea)
        and value is not BaseSea
        and value.__module__ == module.__name__
    ]
    assert len(classes) == 1, f"{module.__name__} defines {len(classes)} SEA classes"
    return classes[0]


def assert_no_removed_getters(sea: ModuleType) -> None:
    """Fail when the SEA module *sea* or its class defines a name no launcher reads.

    The module must not define any :data:`REMOVED_GETTERS` name nor a
    module-level function named like a contract method
    (:data:`MODULE_GETTERS`); the class must not define any
    :data:`REMOVED_GETTERS` name.
    """
    for name in REMOVED_GETTERS:
        assert not hasattr(sea, name), f"{sea.__name__}.{name} is a removed getter"
    for name in MODULE_GETTERS:
        value: Any = getattr(sea, name, None)
        assert not callable(value) or isinstance(value, type), (
            f"{sea.__name__}.{name} is a module-level getter; the contract is the class's method"
        )
    cls = sea_class_of(sea)
    for name in REMOVED_GETTERS:
        assert name not in vars(cls), f"{cls.__name__}.{name} is a removed getter"


def assert_worker_contract(
    module: ModuleType, sea: WorkerSea, own_settings: dict[str, Any]
) -> None:
    """Fail unless *sea* (the :class:`WorkerSea` of *module*) declares exactly *own_settings*.

    ``settings()`` lays *own_settings* over what it is given and nothing
    more; the declared settings are already resolved; the effective
    settings (``base_settings``) are :data:`WORKER_DEFAULTS` (classifier,
    browser and memory off) under *own_settings* and nothing else, so a
    hook such as ``system_prompt`` never leaks into them; and neither the
    module nor the class keeps a getter of the old contract.
    """
    assert isinstance(sea, WorkerSea)
    assert sea.settings({}) == own_settings
    assert sea.settings({"model": "m"}) == {"model": "m", **own_settings}
    assert resolve_settings(own_settings) == own_settings
    assert sea_commands.base_settings([sea]) == {**WORKER_DEFAULTS, **own_settings}
    assert_no_removed_getters(module)


def system_message(request: dict[str, Any]) -> str:
    """Return the system message text of one chat-completions *request*."""
    return str(next(m for m in request["messages"] if m["role"] == "system")["content"])


def tool_names(request: dict[str, Any]) -> set[str]:
    """Return the names of the tools offered in one chat-completions *request*."""
    return {t["function"]["name"] for t in request["tools"]}


def tool_results(request: dict[str, Any]) -> list[str]:
    """Return the tool-result message texts of one chat-completions *request*, in order."""
    return [str(m["content"]) for m in request["messages"] if m["role"] == "tool"]


def tool_result_text(result: str) -> str:
    """Return a tool *result* without the ``\\n\\nSteps: ...`` status footer the agent appends."""
    return result.split("\n\nSteps:")[0]


def assert_appending_sea_run(
    sea: BaseSea,
    task: str,
    script: list[bytes],
    work_dir: Path,
    *,
    tools: set[str],
    system_prompt: str,
    anchors: tuple[str, ...],
    summary: str,
) -> list[str]:
    """Run *sea* on *task* against the scripted model and check an appending SEA's run shape.

    Every agentic request offers at least *tools* and carries a system
    message that keeps the default prompt (it does not start with the
    SEA's *system_prompt*, which the SEA appends) and contains every
    anchor; the run succeeds with *summary* in its summary.

    Returns:
        The tool-result texts of the last agentic request, in order.
    """
    run = sea_commands.evaluate_sea([sea], task)
    parsed, agentic = run_sea(run, script, work_dir, max_steps=len(script) + 2)
    assert parsed["success"] is True
    assert summary in parsed["summary"]
    assert len(agentic) == len(script)
    for request in agentic:
        assert tools <= tool_names(request), tool_names(request)
        system = system_message(request)
        assert not system.startswith(system_prompt)
        for anchor in anchors:
            assert anchor in system, anchor
    return tool_results(agentic[-1])
