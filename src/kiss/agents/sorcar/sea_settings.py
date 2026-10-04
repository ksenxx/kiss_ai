# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``settings()`` contract of an agent script (SEA).

An agent script configures the session that runs it with ONE optional
module-level function::

    def settings() -> dict:
        return {"preset": "worker", "tool_profile": "bash", "max_budget": 1.0}

The dict holds a ``preset`` (a named bundle of defaults, see
:data:`PRESETS`) and any of the per-run parameters of
:func:`kiss.server.sorcar.run` listed in :data:`SETTING_TYPES`, plus two
read by the dispatcher and the daemon respectively: ``timeout`` (seconds
a ``run_agent`` call waits for this script's sub-task) and
``add_to_prompt`` (text appended to the task prompt; ``{task_id}`` in it
is replaced by the calling task's id).  Explicit keys override the
preset; the result overrides whatever the caller (a ``run_agent``
argument, a ``run()`` keyword, the chat panel) sent for the same field.

Besides ``settings()`` a script may define ``add_to_system_prompt()``
(text appended to the system prompt), ``add_to_tools()`` (extra tool
callables), ``description()`` (``/xxx help`` text), the hooks
``llm_call_hook`` / ``tool_call_hook`` and the model-picker pair
``register_as_model()`` / ``on_picked_as_model()``; those are read by
:mod:`kiss.server.agent_file` and :mod:`kiss.agents.sorcar.sea_commands`.

A long base system prompt reads better as a function, so a script may
also define ``system_prompt()``; :mod:`kiss.server.agent_file` applies
its text to the run like a ``system_prompt`` key.  Only ``settings()``
is evaluated here, so the dispatcher can read a script's preset,
timeout and work directory without running its prompt code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home

SETTING_TYPES: dict[str, type | tuple[type, ...]] = {
    "preset": str,
    "prompt": str,
    "work_dir": str,
    "model": str,
    "chat_id": str,
    "system_prompt": str,
    "use_worktree": bool,
    "auto_commit": bool,
    "max_budget": (int, float),
    "model_config": dict,
    "use_web_tools": bool,
    "classify_tasks": bool,
    "use_memory": bool,
    "is_parallel": bool,
    "tool_profile": str,
    "docker_image": str,
    "add_to_prompt": str,
    "timeout": (int, float),
}
"""Every key ``settings()`` may return, with the type its value must have."""

WORKER_PRESET: dict[str, Any] = {
    "use_worktree": False,
    "auto_commit": False,
    "classify_tasks": False,
    "is_parallel": False,
    "use_web_tools": False,
    "use_memory": False,
}

PRESETS: dict[str, dict[str, Any]] = {
    "session": {},
    "worker": WORKER_PRESET,
    "channel": WORKER_PRESET,
}
"""Named bundles of defaults a script picks with ``settings()["preset"]``.

``session`` (the default when no preset is named) changes nothing: the
sub-task is an ordinary Sorcar session with the caller's or the user's
settings.  ``worker`` is a focused tool-bound run on the caller's tree:
no worktree, no auto-commit, no classifier, no fan-out, no browser, no
memory.  ``channel`` is a worker for an external service: the dispatcher runs
it in the shared ``~/.kiss/channel_work`` scratch directory instead of
the caller's project (unless the script's own ``work_dir`` says
otherwise) and inherits nothing from the caller into it, and the daemon
prepends the channel preamble to its system prompt (see
:mod:`kiss.server.agent_file`).
"""

def script_name(path: str) -> str:
    """Return an agent script's display name: its file stem without a ``_sea`` suffix.

    The name ``run_agent`` reports the sub-task under (``"the write_paper
    agent task ..."``) and the ``{name}`` of the channel preamble.
    """
    return Path(path).stem.removesuffix("_sea")


class SettingsError(ValueError):
    """An agent script's settings are malformed (wrong type, unknown key or preset)."""


def resolve_settings(namespace: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective settings of the agent script executed into *namespace*.

    Evaluates the script's ``settings()`` (when defined) and merges the
    named ``preset``'s defaults under its keys.  Every value is
    type-checked against :data:`SETTING_TYPES`.  ``prompt`` and
    ``work_dir`` are kept as the script returned them; the daemon
    resolves them.

    Args:
        namespace: The script's module namespace (``module.__dict__``).

    Returns:
        A new dict: ``{"preset": name, <key>: value, ...}`` with the
        preset's defaults already merged in under the explicit keys.
        A key whose value is ``None`` is dropped: it means "no
        override", so the caller's or the persisted value stands.

    Raises:
        SettingsError: When ``settings()`` is not a function returning a
            dict, names an unknown key or preset, a value has the wrong
            type, or ``settings()`` raises (whatever it raises).
    """
    declared: dict[str, Any] = {}
    if "settings" in namespace:
        settings_fn = namespace["settings"]
        if not callable(settings_fn):
            raise SettingsError(
                f"settings must be a function returning a dict, got {type(settings_fn).__name__}"
            )
        declared = _call(settings_fn, "settings()")
        if not isinstance(declared, dict):
            raise SettingsError(
                f"settings() must return a dict, got {type(declared).__name__}"
            )
        declared = dict(declared)
    sources = {key: f"settings()[{key!r}]" for key in declared}
    for key in declared:
        if key not in SETTING_TYPES:
            raise SettingsError(
                f"settings() has an unknown key {key!r}; "
                f"known keys: {', '.join(SETTING_TYPES)}"
            )
    # ``None`` means "no override": the caller's or persisted value stands.
    declared = {key: value for key, value in declared.items() if value is not None}
    for key, value in declared.items():
        expected = SETTING_TYPES[key]
        wrong_bool = isinstance(value, bool) and expected is not bool
        if wrong_bool or not isinstance(value, expected):
            names = (
                " or ".join(t.__name__ for t in expected)
                if isinstance(expected, tuple) else expected.__name__
            )
            raise SettingsError(
                f"{sources[key]} must be {names}, got {type(value).__name__}"
            )
        declared[key] = _check_value(sources[key], key, value)
    preset = declared.get("preset", "session")
    if preset not in PRESETS:
        raise SettingsError(
            f"unknown preset {preset!r}; known presets: {', '.join(PRESETS)}"
        )
    return {"preset": preset, **PRESETS[preset], **declared}


def _check_value(source: str, key: str, value: Any) -> Any:
    """Validate one type-checked setting beyond its type.

    ``prompt`` must be non-empty; ``max_budget`` and ``timeout`` must be
    finite (``coerce_budget_override`` would otherwise SILENTLY discard a
    NaN/infinite budget downstream) and are returned as floats.  A
    value whose own methods raise (an untrusted ``str`` subclass) is
    reported as broken.

    Raises:
        SettingsError: Naming *source* (``prompt()`` / ``settings()['prompt']``).
    """
    if key == "prompt":
        try:
            empty = not value.strip()
        except BaseException as exc:  # noqa: BLE001 — an untrusted str subclass may raise
            raise SettingsError(
                f"{source} returned a broken value: {type(exc).__name__}: {exc}"
            ) from exc
        if empty:
            raise SettingsError(f"{source} must return a non-empty string")
    if key in ("max_budget", "timeout"):
        try:
            value = float(value)
        except OverflowError:
            value = math.inf
        except BaseException as exc:  # noqa: BLE001 — an untrusted number subclass may raise
            raise SettingsError(
                f"{source} returned a broken value: {type(exc).__name__}: {exc}"
            ) from exc
        if not math.isfinite(value):
            raise SettingsError(f"{source} must return a finite number or None")
    return value


def default_work_dir(settings: Mapping[str, Any], fallback: str) -> str:
    """Return the work directory a run of a script with *settings* defaults to.

    The script's own ``work_dir`` when it names one; else, for the
    ``channel`` preset, the channel agents' shared scratch directory
    ``~/.kiss/channel_work`` (an external-service worker must not act
    in the calling project, whose git lifecycle it does not join);
    else *fallback* (the caller's directory).
    """
    declared = str(settings.get("work_dir") or "")
    if declared:
        return declared
    if settings.get("preset") == "channel":
        return str(kiss_home() / "channel_work")
    return fallback


def _call(fn: Any, label: str) -> Any:
    """Call *fn*; anything it raises becomes a :exc:`SettingsError` naming *label*."""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        try:
            detail = f"{type(exc).__name__}: {exc}"
        except BaseException:  # noqa: BLE001 — an untrusted __str__ may raise too
            detail = type(exc).__name__
        raise SettingsError(f"{label} raised: {detail}") from exc
