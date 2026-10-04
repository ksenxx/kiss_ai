# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared assertions for the bundled SEAs' ``settings()`` contract.

An agent script configures its run through ``settings()`` alone (see
:mod:`kiss.agents.sorcar.sea_settings`).  The per-field getters of the
old contract are no longer read by anything: a bundled SEA that still
defined one would ship a silently ignored function, so every SEA
contract test asserts none of those names exists.
"""

from __future__ import annotations

from typing import Any

REMOVED_GETTERS: tuple[str, ...] = (
    "model", "work_dir", "chat_id", "docker_image", "model_config",
    "max_budget", "tool_profile", "use_worktree", "auto_commit", "classify_tasks",
    "is_parallel", "use_web_tools", "use_memory", "dispatch_timeout",
    "append_to_prompt", "append_to_system_prompt", "tools", "if_append_basic_tools",
)
"""Module-level names the old per-field getter contract used; none is read any more.

``prompt(task)`` is not among them: it is the one getter that shapes
the task prompt (the former ``add_to_prompt`` setting folded into it).
"""


def assert_no_removed_getters(sea: Any) -> None:
    """Fail when *sea* (a SEA module) defines any name in :data:`REMOVED_GETTERS`."""
    for name in REMOVED_GETTERS:
        assert not hasattr(sea, name), f"{sea.__name__}.{name} is a removed getter"
