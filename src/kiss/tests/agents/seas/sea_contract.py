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

from types import ModuleType
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea

REMOVED_GETTERS: tuple[str, ...] = (
    "model", "work_dir", "chat_id", "docker_image", "model_config",
    "max_budget", "tool_profile", "use_worktree", "auto_commit", "auto_classify",
    "allow_fan_out", "use_web_tools", "use_memory", "dispatch_timeout",
    "append_to_prompt", "append_to_system_prompt", "if_append_basic_tools",
    "add_to_prompt", "add_to_system_prompt", "add_to_tools",
)
"""Names the old contracts read as getters; none is read any more, on a module or a class."""

MODULE_GETTERS: tuple[str, ...] = (
    "description", "settings", "prompt", "system_prompt", "tools",
    "tool_call_hook", "llm_call_hook", "register_as_model", "on_picked_as_model",
)
"""The contract methods: defined on the SEA class, never as module-level functions."""


def sea_class_of(module: ModuleType) -> type[BaseSea]:
    """Return the one :class:`BaseSea` subclass *module* defines itself."""
    classes = [
        value for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, BaseSea) and value is not BaseSea
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
