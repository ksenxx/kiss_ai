# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The contract of an agent script (SEA) and the loader that executes one.

An agent script configures the session that runs it with optional
module-level functions.  The run parameters come from ``settings()``::

    def settings() -> dict:
        return {"preset": "worker", "tool_profile": "bash", "max_budget": 1.0}

``settings()`` is data: a ``preset`` (a named dict of defaults, see
:func:`presets`), optionally ``extends`` (the command name or path of
a base script whose configuration this one refines, see
:func:`kiss.agents.sorcar.sea_commands.sea_layers`), any of the
per-run parameters of :func:`kiss.server.sorcar.run` listed in
:data:`SETTING_TYPES`, and three keys read by the dispatcher:
``timeout`` (seconds a ``run_agent`` call waits for this script's
sub-task), ``inherit`` (whether a ``run_agent`` sub-task takes the
calling task's model, budget share, chat, prompt suffixes, tools and
container; default ``True``) and ``kind`` (``"agent"``, the default,
or ``"channel"``: a worker for an external service, see below).
Explicit keys override the preset; the result overrides whatever the
caller (a ``run_agent`` argument, a ``run()`` keyword, the chat panel)
sent for the same field.

Getters are text and code: ``system_prompt()`` replaces the base
system prompt, ``add_to_system_prompt()`` appends to it, and
``prompt(task)`` receives the task text and returns the prompt body
(``{task_id}`` in the result is replaced by the calling task's id).  A
script may further define ``add_to_tools()`` (extra tool callables),
``description()`` (``/xxx help`` text), the hooks ``llm_call_hook`` /
``tool_call_hook`` and the model-picker pair ``register_as_model()`` /
``on_picked_as_model()``; :mod:`kiss.agents.sorcar.sea_commands`
evaluates them.

A script runs in one of three ways; what differs is only where it runs
and which settings apply:

==================  ====================  ======================  ========================
                    ``/<name> task``      ``run_agent(agent=)``   ``run_parallel(agent=)``
==================  ====================  ======================  ========================
Where               the tab's own run     a daemon sub-task in    a thread of the caller
                                          its own tab
Settings honoured   all but ``timeout``,  all                     all but ``timeout``,
                    ``inherit``                                   ``inherit`` (always
                                                                  inherits); a pinned
                                                                  ``use_worktree`` /
                                                                  ``auto_commit`` /
                                                                  ``classify_tasks: True``
                                                                  or ``chat_id`` is
                                                                  refused with an error
Parent inheritance  none (tab settings)   yes, unless             yes; budget =
                                          ``inherit`` is          remaining / (N+1)
                                          ``False``
``timeout``         none                  argument > setting      none
                                          > 3600
``kind: "channel"`` allowed               allowed                 refused
==================  ====================  ======================  ========================

``kind: "channel"`` is the one key the daemon acts on beyond passing a
value through: the run holds its channel workspace
(``run_agent(options='{"workspace": ...}')``, default ``"default"``)
for its lifetime, the channel preamble is added to its system prompt,
and it can be neither a ``run_parallel`` child nor an ``extends``
base.

:func:`execute_python_file` is the ONE loader every reader of a script
uses: it compiles and executes the file into a throw-away module, so
every run observes the file's current contents.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)

SETTING_TYPES: dict[str, type | tuple[type, ...]] = {
    "preset": str,
    "extends": str,
    "work_dir": str,
    "model": str,
    "chat_id": str,
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
    "timeout": (int, float),
    "inherit": bool,
    "kind": str,
}
"""Every key ``settings()`` may return, with the type its value must have."""

KINDS = ("agent", "channel")
"""The values of the ``kind`` setting (see the module docstring)."""

REMOVED_SETTINGS: dict[str, str] = {
    "prompt": (
        "define `def prompt(task: str) -> str` instead; it receives the "
        "task text and returns the prompt body"
    ),
    "system_prompt": "define `def system_prompt() -> str` instead",
    "add_to_prompt": (
        "return the extra text from `def prompt(task: str) -> str` instead "
        "(`{task_id}` in its result is replaced by the calling task's id)"
    ),
}
"""Former ``settings()`` keys, each with the function that replaced it.

Kept for one release so a script still using the old spelling fails
with a pointed message instead of an "unknown key" one.
"""

WORKER_PRESET: dict[str, Any] = {
    "use_worktree": False,
    "auto_commit": False,
    "classify_tasks": False,
    "is_parallel": False,
    "use_web_tools": False,
    "use_memory": False,
}

PRESET_NAMES = ("session", "worker", "channel")
"""The presets a script may name (see :func:`presets`)."""


def presets() -> dict[str, dict[str, Any]]:
    """Return the named dicts of defaults a script picks with ``settings()["preset"]``.

    A preset is nothing but defaults laid under the script's explicit
    keys.  ``session`` (the default when no preset is named) is empty:
    the run is an ordinary Sorcar session with the caller's or the
    user's settings.  ``worker`` is a focused tool-bound run on the
    caller's tree: no worktree, no auto-commit, no classifier, no
    fan-out, no browser, no memory.  ``channel`` is a worker for an
    external service: ``kind: "channel"``, ``inherit: False`` and a
    ``work_dir`` of the shared ``~/.kiss/channel_work`` scratch
    directory (never the caller's project, whose git lifecycle it does
    not join).  Computed on every call so a redirected ``$KISS_HOME``
    is honoured.
    """
    return {
        "session": {},
        "worker": dict(WORKER_PRESET),
        "channel": {
            **WORKER_PRESET,
            "kind": "channel",
            "inherit": False,
            "work_dir": str(kiss_home() / "channel_work"),
        },
    }


class SeaError(Exception):
    """Base of every "this agent script is broken" error.

    :exc:`kiss.agents.sorcar.sea_commands.SeaScriptError` (raised by the
    registry and the dispatcher) and
    :exc:`kiss.server.agent_file.AgentFileError` (raised by the daemon's
    task runner) both derive from it, so a caller that only wants to
    know "the script failed" catches one class.
    """


def script_name(path: str) -> str:
    """Return an agent script's display name: its file stem without a ``_sea`` suffix.

    The name ``run_agent`` reports the sub-task under (``"the write_paper
    agent task ..."``) and the ``{name}`` of the channel preamble.
    """
    return Path(path).stem.removesuffix("_sea")


def safe_message(exc: BaseException) -> str:
    """Format an untrusted exception without trusting its ``__str__``.

    ``str(exc)`` runs the exception's ``__str__``, which — for an
    exception minted by an untrusted agent script — may itself raise
    anything.  A diagnostic built here must never leak such a secondary
    raise, so the conversion is guarded and falls back to the type name.

    Args:
        exc: The exception raised by untrusted agent-script code.

    Returns:
        ``"TypeName: message"`` when the message renders, else ``"TypeName"``.
    """
    name = type(exc).__name__
    try:
        return f"{name}: {exc}"
    except BaseException:  # noqa: BLE001 — untrusted __str__ may raise anything
        return name


def execute_python_file(
    raw_path: Any,
    error_cls: type[Exception] = SeaError,
    label: str = "agent script",
) -> dict[str, Any]:
    """Execute a caller-supplied Python file and return its namespace.

    The one loader of agent scripts (the daemon's ``agentPath``, the
    slash-command registry, the dispatcher, SEAs that load other
    scripts such as ``skillopt``).  The source is compiled and executed
    directly (no ``__pycache__`` read or write), so every call observes
    the file's CURRENT contents and the caller's directory is never
    littered with bytecode.  The module is registered in ``sys.modules``
    exactly as ``import`` does, under a name derived from the file's
    path, and stays registered: ``@dataclass`` under ``from __future__
    import annotations`` and ``typing.get_type_hints`` resolve string
    annotations through ``sys.modules[cls.__module__]`` whenever a
    getter or a tool of the script runs later.  A re-execution of the
    same file replaces the entry (a failed one restores the previous
    entry), so a long-lived daemon holds one module per script file,
    not one per run, and an earlier execution's classes resolve their
    annotations through the latest execution of the same source; two
    files with the same stem in different folders get different names.

    Args:
        raw_path: The path of the file — expected to be an absolute path
            string, but treated as untrusted.
        error_cls: The exception class to raise on any failure, so each
            caller keeps its own diagnostic type.
        label: Human-readable name of the file kind (``"agent script"``,
            ``"SEA"``), used in diagnostic messages.

    Returns:
        The executed module's namespace dict.

    Raises:
        Exception: An *error_cls* instance when *raw_path* is not a
            string, is not the path of an existing ``.py`` file, or
            names a module that raises at import time (``BaseException``
            included: a file raising ``KeyboardInterrupt`` or
            ``SystemExit`` at import time is a broken file, not a
            cancelled task; the original raise stays reachable as
            ``__cause__``).
    """
    # Type-check FIRST: comparing or repr-ing an untrusted non-string
    # object could run arbitrary code (raising ``__eq__``/``__repr__``),
    # so nothing touches *raw_path* beyond isinstance until it is known
    # to be a plain string.
    if not isinstance(raw_path, str):
        raise error_cls(
            f"{label} field must be a path string, got {type(raw_path).__name__}"
        )
    path = Path(raw_path)
    try:
        is_py_file = path.suffix == ".py" and path.is_file()
    except (OSError, ValueError):
        # e.g. an embedded NUL byte makes ``is_file`` raise ValueError.
        is_py_file = False
    if not is_py_file:
        raise error_cls(f"{label} {raw_path!r} is not an existing Python (.py) file")
    module_name = f"_kiss_sea_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    # A failed re-execution must not unregister the module of an earlier,
    # successful execution of the same file whose classes still resolve
    # their annotations through this name: the previous entry is put back.
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        source = path.read_text(encoding="utf-8")
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 — the script is the user's own code
    except BaseException as exc:  # noqa: BLE001 — untrusted module code may raise anything
        logger.warning("Failed to import %s %r", label, raw_path, exc_info=True)
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise error_cls(
            f"{label} {raw_path!r} failed to import: {safe_message(exc)}"
        ) from exc
    return module.__dict__


class SettingsError(SeaError, ValueError):
    """An agent script's settings are malformed (wrong type, unknown key or preset)."""


def resolve_settings(namespace: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective settings of the agent script executed into *namespace*.

    Evaluates the script's ``settings()`` (when defined) and merges the
    named ``preset``'s defaults under its keys.  Every value is
    type-checked against :data:`SETTING_TYPES`.  ``work_dir`` and
    ``extends`` are kept as the script returned them; the daemon and
    :func:`kiss.agents.sorcar.sea_commands.sea_layers` resolve them.

    Args:
        namespace: The script's module namespace (``module.__dict__``).

    Returns:
        A new dict: ``{"preset": name, <key>: value, ...}`` with the
        preset's defaults already merged in under the explicit keys.
        A key whose value is ``None`` is dropped — as is a ``model`` of
        ``""`` — it means "no override", so the caller's or the
        persisted value stands.

    Raises:
        SettingsError: When ``settings()`` is not a function returning a
            dict, names an unknown or removed key, an unknown preset or
            an unknown ``kind``, a value has the wrong type, or
            ``settings()`` raises (whatever it raises).
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
        if key in REMOVED_SETTINGS:
            raise SettingsError(
                f"settings()[{key!r}] is no longer a setting: {REMOVED_SETTINGS[key]}"
            )
        if key not in SETTING_TYPES:
            raise SettingsError(
                f"settings() has an unknown key {key!r}; "
                f"known keys: {', '.join(SETTING_TYPES)}"
            )
    # ``None`` means "no override": the caller's or persisted value
    # stands.  So does an empty ``model`` (the spelling of "no model" a
    # script computing its model may produce).
    declared = {
        key: value for key, value in declared.items()
        if value is not None and not (key == "model" and value == "")
    }
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
    if preset not in PRESET_NAMES:
        raise SettingsError(
            f"unknown preset {preset!r}; known presets: {', '.join(PRESET_NAMES)}"
        )
    return {"preset": preset, **presets()[preset], **declared}


def merge_settings(chain: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the effective settings of a script and the bases it extends.

    *chain* lists resolved settings (:func:`resolve_settings`) from the
    outermost base to the script itself; a later entry's key wins.  The
    effective ``preset`` is the last one that names a preset other than
    ``session`` (``session`` changes nothing, so it never masks a
    base's preset).  ``extends`` is dropped: the chain has resolved it.

    Args:
        chain: The resolved settings, base first.

    Returns:
        The merged settings dict; ``{"preset": "session"}`` for an
        empty chain.
    """
    merged: dict[str, Any] = {}
    for settings in chain:
        merged.update(settings)
    merged["preset"] = next(
        (s["preset"] for s in reversed(chain) if s["preset"] != "session"), "session",
    )
    merged.pop("extends", None)
    return merged


def _check_value(source: str, key: str, value: Any) -> Any:
    """Validate one type-checked setting beyond its type.

    ``max_budget`` and ``timeout`` must be finite
    (``coerce_budget_override`` would otherwise SILENTLY discard a
    NaN/infinite budget downstream) and are returned as floats.  A
    value whose own methods raise (an untrusted number subclass) is
    reported as broken.  ``kind`` must be one of :data:`KINDS`.

    Raises:
        SettingsError: Naming *source* (``settings()['timeout']``).
    """
    if key == "kind" and value not in KINDS:
        raise SettingsError(f"{source} must be one of {', '.join(KINDS)}; got {value!r}")
    if key in ("max_budget", "timeout"):
        try:
            value = float(value)
        except OverflowError:
            value = math.inf
        except BaseException as exc:  # noqa: BLE001 — an untrusted number subclass may raise
            raise SettingsError(
                f"{source} returned a broken value: {safe_message(exc)}"
            ) from exc
        if not math.isfinite(value):
            raise SettingsError(f"{source} must return a finite number or None")
    return value


def _call(fn: Any, label: str) -> Any:
    """Call *fn*; anything it raises becomes a :exc:`SettingsError` naming *label*."""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        raise SettingsError(f"{label} raised: {safe_message(exc)}") from exc
