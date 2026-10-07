# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The settings of a SEA and the loader that executes one.

A SEA is a class deriving from
:class:`kiss.agents.seas.base.base_sea.BaseSea` — directly for an
ordinary session, through :class:`~kiss.agents.seas.base.base_sea.WorkerSea`
for a focused tool-bound run on the caller's tree, through
:class:`~kiss.agents.seas.base.base_sea.ChannelSea` for an agent that
serves an external service (its behaviours are
:data:`CHANNEL_BEHAVIOURS`).  Its run parameters come from its
``settings`` method, which lays the SEA's keys over what its base
classes declared::

    def settings(self, settings: dict) -> dict:
        return settings | {"tool_profile": "bash", "max_budget": 1.0}

The result is data: any of the per-run parameters of
:func:`kiss.server.sorcar.run` listed in :data:`SETTING_TYPES`, and
three keys read by the launcher: ``timeout`` (seconds a ``run_agent``
/ ``run_parallel`` call blocks for the run: the call's value, else
this setting, else 3600), ``locked`` (the keys an explicit caller
argument may not replace) and ``hidden`` (the file is no command).
Against the caller there is one precedence rule,
:data:`PRECEDENCE_RULE`, enforced by :func:`locked_conflicts` and
rendered by ``sea docs`` into every page that states it.  A relative
``work_dir`` — a SEA's setting or a call's option — is a path under
the calling task's directory (:func:`anchored_work_dir`).

The other methods of the class — ``prompt``, ``system_prompt``,
``tools``, ``tool_call_hook``, ``llm_call_hook`` — shape the run;
:mod:`kiss.agents.sorcar.sea_commands` loads the class and applies
them.

A SEA runs in one of three ways; ``run_parallel`` is N ``run_agent``
calls, so what differs is only the tab and the inheritance:

==================  ====================  ==========================================
                    ``/<name> task``      ``run_agent(agent=)`` / ``run_parallel``
==================  ====================  ==========================================
Where               the tab's own run     a daemon sub-task in its own tab
Settings honoured   all but ``timeout``   all
Parent inheritance  none (tab settings)   yes, unless the SEA is a channel or the
                                          ``inherit`` option is ``false``; budget =
                                          remaining / (N+1) for N sub-tasks
``timeout``         none                  argument > setting > 3600: the seconds the
                                          call blocks before returning the job ids
==================  ====================  ==========================================

The base class is the one thing with behaviour beyond a value passed
through — every behaviour of a channel is listed in
:data:`CHANNEL_BEHAVIOURS`.

:func:`execute_python_file` is the ONE loader every reader of a SEA
file uses: it compiles and executes the file into a throw-away module,
so every run observes the file's current contents.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import math
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WORKER_DEFAULTS, channel_work_dir

__all__ = ["WORKER_DEFAULTS", "channel_work_dir"]

logger = logging.getLogger(__name__)

PRECEDENCE_RULE = (
    "For every setting of a sub-task: what the call passes explicitly (a `run_agent` / "
    "`run_parallel` argument or `options` key) wins, then the SEA's `settings()`, then what "
    "the calling task passes on (and, for `/<name>`, the chat panel's persisted settings), "
    "then the user's defaults. A SEA may list keys in `locked`: a call that passes a "
    "different value for a locked key is refused with an error, never silently overruled."
)
"""The one precedence rule of SEA settings, stated once.

:func:`locked_conflicts` enforces the ``locked`` clause; ``sea docs``
renders the sentence into the ``<!-- sea-docs: precedence -->`` block of
every documentation page, and the ``run_agent`` / ``run_parallel`` tool
docstrings quote it, so a change here changes every statement of it.
"""

SETTING_TYPES: dict[str, type | tuple[type, ...]] = {
    "work_dir": str,
    "model": str,
    "chat_id": str,
    "use_worktree": bool,
    "auto_commit": bool,
    "max_budget": (int, float),
    "model_config": dict,
    "use_web_tools": bool,
    "auto_classify": bool,
    "use_memory": bool,
    "tool_profile": str,
    "docker_image": str,
    "timeout": (int, float),
    "locked": list,
    "hidden": bool,
}
"""Every key ``settings()`` may return, with the type its value must have."""

SETTING_DOCS: dict[str, str] = {
    "work_dir": "The directory the run works in; default: the calling task's or the tab's. A "
                "relative path is a path under the calling task's directory (the tab's for "
                "`/<name>`), in a SEA's settings and in a call's `work_dir` option alike.",
    "model": "The LLM model, a catalogue name or a model-picker SEA; default: the caller's "
             "(`\"\"`, like `None`, is no override — true of every string key).",
    "chat_id": "The chat the run's events go to; default under `run_agent`: the calling "
               "task's chat, or a new chat when nothing is inherited (a channel, an "
               "`inherit: false` call); a `/<name>` run keeps the tab's chat.",
    "use_worktree": "Run in a git worktree of the project; default: the calling task's "
                    "effective choice, else the persisted setting (an inherited or default "
                    "`True` is demoted by the classifier for non-implementation tasks, an "
                    "explicit `True` is kept).",
    "auto_commit": "Commit the run's changes when it ends; default: the calling task's "
                   "effective choice, else the persisted setting.",
    "max_budget": "USD budget of the run, a finite number; default: the caller's share or the "
                  "daemon's default.",
    "model_config": "Model configuration dict passed to the LLM (temperature, base URL, ...).",
    "use_web_tools": "Give the run the browser tools (daemon default: on).",
    "auto_classify": "Let the pre-run classifier decide the worktree mode and lite prompt "
                     "(daemon default: the persisted setting).",
    "use_memory": "Give the run the `memory_*` tools (daemon default: the persisted setting).",
    "tool_profile": "The run's toolset: `review`, `bash`, `shell+edit`, ... (default: the full "
                    "toolset).",
    "docker_image": "Run inside this Docker image (default: the host).",
    "timeout": "Seconds a `run_agent` / `run_parallel` call blocks for the run: the call's "
               "`timeout` argument wins, then this setting, then 3600. When it expires the "
               "run is not stopped: it keeps going as an `agent_job` and the call returns "
               "its job id (a job still running when the calling task ends is killed). "
               "Ignored by `/<name>`.",
    "locked": "Keys an explicit `run_agent` / `run_parallel` argument or option may not change: "
              "a differing value is an error.",
    "hidden": "`True`: the script is no `/command` and no `run_agent` agent name (loadable by "
              "path and as a base class only). It must be the literal `True` in the class's "
              "`settings()` because the command registry reads it from the source without "
              "running the script (a computed value is ignored).",
}
"""One line of documentation per :data:`SETTING_TYPES` key (rendered by ``sea docs``)."""

META_SETTINGS = ("locked", "hidden")
"""Keys that shape the settings themselves rather than the run; never lockable."""

DISPATCHER_SETTINGS = ("timeout", "locked", "hidden")
"""``settings()`` keys with no ``run`` command wire field.

``timeout`` is read by the dispatcher
(:mod:`kiss.agents.sorcar.agent_dispatch`); ``locked`` names the keys
an explicit caller argument may not replace (:func:`locked_conflicts`);
``hidden`` keeps the script out of the command registry
(:func:`declares_hidden`).  Every other key is a parameter of
:func:`kiss.server.sorcar.run`, sent as :func:`wire_field` of the key.
"""

RENAMED_SETTINGS = {
    "classify_tasks": "auto_classify",
    "model_name": "model",
}
"""Former settings keys and their current names.

``auto_classify`` says whether the daemon's classifier decides the
run's worktree mode (the wire field ``classifyTasks``); ``model_name``
is the ``SorcarAgent.run`` parameter callers keep reaching for in
place of the settings key ``model``.  The old names are refused — in
a script's ``settings()`` and in a ``run_agent`` / ``run_parallel``
``options`` object alike — with a message naming the new one; ``sea
lint --fix`` rewrites them in a script.  :func:`kiss.server.sorcar.run`
takes the current names as its keywords, so a key spelled the
settings way is right everywhere.
"""

RENAMED_OPTIONS = {
    **RENAMED_SETTINGS,
    "append_to_prompt": "add_to_prompt",
    "append_to_system_prompt": "add_to_system_prompt",
}
"""Former ``options`` keys of ``run_agent`` / ``run_parallel`` and their current names.

:data:`RENAMED_SETTINGS` plus the two prompt suffixes, which are
options only — ``append_to_prompt`` and ``append_to_system_prompt``
were the former ``run()`` keywords of the options ``add_to_prompt``
and ``add_to_system_prompt`` (the wire fields are still
``appendToPrompt`` and ``appendToSystemPrompt``).  In a script's
``settings()`` those keys are refused as removed
(:data:`REMOVED_SETTINGS`): a SEA shapes the prompts in its ``prompt``
and ``system_prompt`` methods instead.
"""

PROFILE_ALIASES = {"readonly": "review", "read_only": "review", "read-only": "review"}
"""Spellings accepted for a tool-profile key, and the key each means.

``review`` is the read-only profile and callers keep asking for it by
that property.  Every way a profile name arrives — a script's
``tool_profile`` setting (:func:`resolve_settings`), a ``run_agent`` /
``run_parallel`` argument, a ``run()`` parameter
(:func:`kiss.agents.sorcar.sorcar_agent.canonical_tool_profile`) —
replaces the alias with the key first, so locks, the ``pinned`` record
and the ``ran:`` line compare and show keys only.
"""


def alias_free_profile(name: str) -> str:
    """Return the profile *name* with every ``+``-joined part stripped and its alias replaced.

    Unknown parts pass through unchanged: this is spelling, not
    validation (:func:`kiss.agents.sorcar.sorcar_agent.canonical_tool_profile`
    validates against the profile table).
    """
    if not name.strip():
        return ""
    return "+".join(PROFILE_ALIASES.get(p.strip(), p.strip()) for p in name.split("+"))


_PROMPT_SUFFIX_REMOVED = (
    "a SEA shapes the task text in its `prompt(task)` method; `add_to_prompt` is a "
    "`run_agent` option, not a setting"
)
_SYSTEM_PROMPT_SUFFIX_REMOVED = (
    "a SEA shapes the system prompt in its `system_prompt(system_prompt)` method; "
    "`add_to_system_prompt` is a `run_agent` option, not a setting"
)

_BASE_CLASS_REMOVED = (
    "what a SEA is became its base class: derive from `WorkerSea` (a tool-bound run on the "
    "caller's tree) or `ChannelSea` (an external-service agent) instead of `BaseSea` "
    "(`from kiss.agents.seas.base.base_sea import ...`); run `uv run sea lint --fix` to "
    "rewrite the script"
)

REMOVED_SETTINGS = {
    "kind": _BASE_CLASS_REMOVED,
    "preset": _BASE_CLASS_REMOVED,
    "channel": _BASE_CLASS_REMOVED,
    "allow_fan_out": "`run_parallel` is N `run_agent` calls, so there is nothing to allow or "
                     "forbid separately; `tool_profile` chooses the toolset",
    "is_parallel": "`run_parallel` is N `run_agent` calls, so there is nothing to allow or "
                   "forbid separately; `tool_profile` chooses the toolset",
    "inherit": "a channel never inherits from the calling task and every other SEA always "
               "does; the caller's `inherit` option opts out of inheriting",
    "append_to_prompt": _PROMPT_SUFFIX_REMOVED,
    "add_to_prompt": _PROMPT_SUFFIX_REMOVED,
    "append_to_system_prompt": _SYSTEM_PROMPT_SUFFIX_REMOVED,
    "add_to_system_prompt": _SYSTEM_PROMPT_SUFFIX_REMOVED,
    "append_basic_tools": "the built-in toolset is chosen by `tool_profile` (`none` gives a "
                          "run no built-in tools)",
    "extends": "a SEA extends another by Python inheritance: derive the SEA class from the "
               "base's class (import it, or `sea_class(\"name\")`)",
}
"""Former settings keys with no replacement, and why; refused with the explanation."""

BASE_CLASS_DOCS: dict[str, str] = {
    "BaseSea": "The default: an ordinary Sorcar session with the caller's or the user's "
               "settings (`/write`, `/write_paper`, `bestrouter`).",
    "WorkerSea": "A focused tool-bound run on the caller's tree: lays the worker defaults "
                 "(no worktree, no auto-commit, no classifier, no browser, no memory) under "
                 "the SEA's own keys (`/sh`, `/ask`, `/merge`, `/remember`, `/forget`, "
                 "`/task_update`).",
    "ChannelSea": "A worker that serves an external service (Slack, email, cron), not the "
                  "caller's project; every behaviour this adds is listed in the channel "
                  "table. The command registry reads the base name from the source, so a "
                  "channel derives from `ChannelSea` by that name.",
}
"""One line of documentation per base class (rendered by ``sea docs``)."""


def base_class_defaults() -> dict[str, dict[str, Any]]:
    """Return the settings each base class lays under a subclass's own keys.

    What ``sea docs`` renders next to :data:`BASE_CLASS_DOCS`:
    :class:`~kiss.agents.seas.base.base_sea.BaseSea` lays nothing,
    :class:`~kiss.agents.seas.base.base_sea.WorkerSea` the
    :data:`WORKER_DEFAULTS`, :class:`~kiss.agents.seas.base.base_sea.ChannelSea`
    those plus the scratch ``work_dir``.
    """
    return {
        "BaseSea": {},
        "WorkerSea": dict(WORKER_DEFAULTS),
        "ChannelSea": {**WORKER_DEFAULTS, "work_dir": channel_work_dir()},
    }


CHANNEL_BEHAVIOURS: tuple[tuple[str, str], ...] = (
    ("worker",
     "It is a `WorkerSea`, and the worker keys are locked, so no call may give it a "
     "worktree, auto-commit, the classifier, the browser or memory (`resolve_settings`)."),
    ("scratch directory",
     "It runs in the shared `<home>/channel_work` scratch directory unless its own "
     "`work_dir` says otherwise (`/cron` runs in `<home>/cron_work`); `work_dir` is locked, "
     "so it never works in the caller's project (`resolve_settings`)."),
    ("no inheritance",
     "Its `run_agent` dispatch takes nothing from the calling task: not the chat, model, "
     "budget share, container or prompt suffixes; the `inherit: true` option is refused "
     "(`agent_dispatch`)."),
    ("workspace",
     "It holds its channel workspace — the `run_agent` option `workspace`, default "
     "`default`, the account its tools load credentials for — from before its `tools()` "
     "run until the run ends; `workspace` is refused for any other SEA "
     "(`sea_apply.channel_workspace`, `agent_dispatch`)."),
    ("preamble",
     "The channel preamble is appended to its system prompt before its own "
     "`system_prompt()` runs (`sea_apply.CHANNEL_PREAMBLE`)."),
    ("listed as a channel",
     "A third-party SEA folder whose class derives from `ChannelSea` by that name is a "
     "channel agent: `run_agent(agent=\"<folder>\")` finds it by name and the `channel` "
     "tool lists it (`agent_dispatch.available_channels`, `declares_channel`)."),
)
"""Every behaviour deriving from ``ChannelSea`` adds, as ``(name, what it does and where it
is enforced)``.

The one place the list exists: ``sea docs`` renders it, and the
modules named enforce exactly these.  A setting has no behaviour
beyond its value.
"""


class SeaError(Exception):
    """The one "this SEA is broken" error: a bad file, settings, method or dispatch.

    Raised by the loader, the settings resolver, the command registry,
    the dispatcher and the daemon's task runner alike, so a caller that
    wants to know "the SEA failed" catches one class and the message
    (prefixed with the SEA's file name where known) is the diagnostic.
    """


def wire_field(key: str) -> str:
    """Return the ``run`` command wire field of the ``run()`` keyword *key*.

    The wire vocabulary is the keyword vocabulary in camelCase
    (``use_web_tools`` -> ``useWebTools``), with three aliases kept from
    the wire protocol's earlier vocabulary: ``add_to_prompt`` ->
    ``appendToPrompt``, ``add_to_system_prompt`` ->
    ``appendToSystemPrompt`` and ``auto_classify`` -> ``classifyTasks``.
    This table is the only place the wire spelling exists on the
    client side: settings keys, ``options`` keys and ``run()`` keywords
    all use the snake_case name.
    """
    aliases = {
        "add_to_prompt": "appendToPrompt",
        "add_to_system_prompt": "appendToSystemPrompt",
        "auto_classify": "classifyTasks",
    }
    if key in aliases:
        return aliases[key]
    first, *rest = key.split("_")
    return first + "".join(part.capitalize() for part in rest)


def script_name(path: str) -> str:
    """Return a SEA's display name: its file stem without a ``_sea`` suffix.

    The name ``run_agent`` reports the sub-task under (``"the write_paper
    agent task ..."``) and the ``{name}`` of the channel preamble.
    """
    return Path(path).stem.removesuffix("_sea")


def safe_message(exc: BaseException) -> str:
    """Format an untrusted exception without trusting its ``__str__``.

    ``str(exc)`` runs the exception's ``__str__``, which — for an
    exception minted by an untrusted SEA — may itself raise
    anything.  A diagnostic built here must never leak such a secondary
    raise, so the conversion is guarded and falls back to the type name.

    Args:
        exc: The exception raised by untrusted SEA code.

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
    label: str = "SEA",
) -> dict[str, Any]:
    """Execute a caller-supplied Python file and return its namespace.

    The one loader of SEAs (the daemon's ``seaPath``, the
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
        label: Human-readable name of the file kind (``"SEA"``,
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


def anchored_work_dir(work_dir: str, base_dir: str) -> str:
    """Return *work_dir* as the absolute directory it names from the calling task's *base_dir*.

    The one rule for a relative ``work_dir``, wherever it is written: a
    SEA's ``work_dir`` setting, a ``run_agent`` / ``run_parallel``
    ``work_dir`` option and the daemon's staging of a SEA's setting
    all resolve it against the directory of the calling task (the tab's
    for ``/<name>``), never against the SEA file's folder.

    Args:
        work_dir: The directory as written; ``~`` is expanded.
        base_dir: The calling task's directory.

    Returns:
        *work_dir* itself when absolute, else ``base_dir/work_dir``.
    """
    path = Path(work_dir.strip()).expanduser()
    return str(path if path.is_absolute() else Path(base_dir).expanduser() / path)


def declares_hidden(path: Path) -> bool:
    """Return whether the SEA at *path* writes ``"hidden": True`` in its ``settings`` method.

    Read from the source (``ast``), never by executing the script: the
    command registry calls this for every scanned folder, and a hidden
    SEA (a test fixture, protocol plumbing, a base class other SEAs
    derive from) must be excluded without running it.  Hence the
    contract that ``hidden`` is a literal ``True`` in a dict inside the
    file's own ``settings`` method; a computed or inherited value is
    accepted by :func:`resolve_settings` but does not hide the SEA.
    """
    return declared_literal(path, "hidden") is True


def declares_channel(path: Path) -> bool:
    """Return whether the SEA at *path* derives its class from ``ChannelSea`` by that name.

    Read from the source (``ast``), never by executing the script, like
    :func:`declares_hidden`: the channel listing scans every
    third-party folder without importing the channel modules.  Hence
    the contract that a channel names ``ChannelSea`` (bare, under an
    import alias, or as the last attribute of a dotted name) in the
    bases of a class the file defines; a channel reached through a
    base class of another file is still a channel when loaded
    (``isinstance``), but is not listed.
    """
    return "ChannelSea" in _declared_bases(path)


def _declared_bases(path: Path) -> set[str]:
    """Return the names of every base class of every class *path* defines (cached per stamp)."""
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return set()
    cached = _BASES_CACHE.get(path)
    if cached is None or cached[0] != stamp:
        cached = (stamp, _class_bases(path))
        _BASES_CACHE[path] = cached
    return cached[1]


_BASES_CACHE: dict[Path, tuple[tuple[int, int], set[str]]] = {}
"""``path -> ((mtime_ns, size), {base class name})`` memo of :func:`_declared_bases`."""


def _class_bases(path: Path) -> set[str]:
    """Parse *path*; return the last name of every base of every class definition in it.

    A base imported under an alias (``from ... import ChannelSea as C``)
    counts under its imported name.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set()
    imported = {
        alias.asname: alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names if alias.asname
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    names.add(imported.get(base.id, base.id))
                elif isinstance(base, ast.Attribute):
                    names.add(base.attr)
    return names


def declared_literal(path: Path, key: str) -> Any:
    """Return the literal value the SEA at *path* writes for *key* in ``settings``, or ``None``.

    Parsed from the source, never executed (see :func:`declares_hidden`);
    only a constant value under a string-literal key in a dict inside
    the file's own ``settings`` method counts.  The parse is cached per path until the
    file's size or mtime changes, so a registry refresh costs one
    ``stat`` per SEA.
    """
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    cached = _LITERAL_CACHE.get(path)
    if cached is None or cached[0] != stamp:
        cached = (stamp, _settings_literals(path))
        _LITERAL_CACHE[path] = cached
    return cached[1].get(key)


_LITERAL_CACHE: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
"""``path -> ((mtime_ns, size), {key: literal value})`` memo of :func:`declared_literal`."""


def _settings_literals(path: Path) -> dict[str, Any]:
    """Parse *path*; return the constant ``"key": value`` entries of the dicts in ``settings()``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    literals: dict[str, Any] = {}
    for node in settings_functions(tree):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict):
                for key, value in zip(sub.keys, sub.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                        literals[str(key.value)] = value.value
    return literals


def settings_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Return every ``def settings`` of *tree* (the SEA class's method, wherever the class is)."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "settings"
    ]


def resolve_settings(declared: Mapping[str, Any], channel: bool = False) -> dict[str, Any]:
    """Return the effective settings a SEA *declared*.

    *declared* is what the SEA's ``settings`` methods returned, base
    class first (:func:`kiss.agents.sorcar.sea_commands.base_settings`
    folds them, the worker and channel defaults of the base classes
    included).  Every value is type-checked against
    :data:`SETTING_TYPES`, a ``tool_profile`` alias (``readonly``)
    becomes its key (:func:`alias_free_profile`), and a *channel*
    gets the settings side of :data:`CHANNEL_BEHAVIOURS`: ``work_dir``
    plus the worker keys are locked.  ``work_dir`` is otherwise kept as
    declared; the launcher anchors a relative one at the calling task's
    directory (:func:`anchored_work_dir`).

    Args:
        declared: The declared settings.
        channel: Whether the SEA derives from ``ChannelSea``.

    Returns:
        A new dict of the effective settings.  A key whose value is
        ``None`` is dropped — as is a string key (``work_dir``,
        ``model``, ``chat_id``, ``tool_profile``, ``docker_image``)
        whose value is ``""`` — it means "no override", so the
        caller's or the persisted value stands.

    Raises:
        SeaError: When *declared* names an unknown, renamed or removed
            key, or a value has the wrong type.
    """
    sources = {key: f"settings()[{key!r}]" for key in declared}
    for key in declared:
        if key in RENAMED_SETTINGS:
            raise SeaError(
                f"settings() key {key!r} was renamed to {RENAMED_SETTINGS[key]!r}; "
                f"run `uv run sea lint --fix` to rewrite the script"
            )
        if key in REMOVED_SETTINGS:
            raise SeaError(f"settings() key {key!r} was removed: {REMOVED_SETTINGS[key]}")
        if key not in SETTING_TYPES:
            raise SeaError(
                f"settings() has an unknown key {key!r}; "
                f"known keys: {', '.join(SETTING_TYPES)}"
            )
    # ``None`` means "no override": the caller's or persisted value
    # stands.  So does ``""`` for a string key (the spelling of "none"
    # a script computing its model, profile or directory may produce),
    # as a blank ``run_agent`` option does (``agent_dispatch.parse_options``).
    settings = {
        key: value for key, value in declared.items()
        if value is not None and not (SETTING_TYPES[key] is str and value == "")
    }
    if isinstance(settings.get("tool_profile"), str):
        settings["tool_profile"] = alias_free_profile(settings["tool_profile"])
    for key, value in settings.items():
        expected = SETTING_TYPES[key]
        wrong_bool = isinstance(value, bool) and expected is not bool
        if wrong_bool or not isinstance(value, expected):
            names = (
                " or ".join(t.__name__ for t in expected)
                if isinstance(expected, tuple) else expected.__name__
            )
            raise SeaError(
                f"{sources[key]} must be {names}, got {type(value).__name__}"
            )
        settings[key] = _check_value(sources[key], key, value)
    if channel:
        # CHANNEL_BEHAVIOURS "worker" and "scratch directory": the base
        # classes laid the values; the locks are added here so a
        # subclass that writes its own ``locked`` cannot drop them.
        settings["locked"] = sorted({*settings.get("locked", ()), "work_dir", *WORKER_DEFAULTS})
    return settings


def locked_conflicts(
    settings: Mapping[str, Any], asked: Mapping[str, Any], base_dir: str = "",
) -> str:
    """Return why explicit values clash with a script's ``locked`` settings, or ``""``.

    The precedence of a sub-task's settings is: an explicit argument of
    the call wins over the script's ``settings()``, which win over
    what the calling task passes on, which win over the user's
    persisted settings — except for the keys the script lists in
    ``locked``: an explicit argument that differs from a locked value
    is an error, never silently replaced.

    Args:
        settings: The script's merged settings.
        asked: ``{setting key: value}`` as the call passed them
            explicitly (``None`` and ``""`` count as not passed).
        base_dir: The directory a relative ``work_dir`` (asked or
            locked) is resolved against before the two are compared;
            empty compares them as given.

    Returns:
        ``""`` when nothing clashes, else one sentence naming every
        clash, e.g. ``the script locks tool_profile='bash' (asked for
        'review')``.
    """
    clashes = [
        f"{key}={settings[key]!r} (asked for {asked[key]!r})"
        for key in settings.get("locked") or ()
        if key in settings and asked.get(key) not in (None, "")
        and not _same_setting(key, asked[key], settings[key], base_dir)
    ]
    if not clashes:
        return ""
    return "the script locks " + ", ".join(clashes)


def _same_setting(key: str, asked: Any, locked: Any, base_dir: str) -> bool:
    """Return whether *asked* equals *locked*; ``work_dir`` paths are compared resolved."""
    if key == "work_dir" and isinstance(asked, str) and isinstance(locked, str):
        base = Path(base_dir).expanduser() if base_dir else Path.cwd()
        return (base / Path(asked.strip()).expanduser()).resolve() == (
            base / Path(locked.strip()).expanduser()
        ).resolve()
    return bool(asked == locked)


def _check_value(source: str, key: str, value: Any) -> Any:
    """Validate one type-checked setting beyond its type.

    ``max_budget`` and ``timeout`` must be finite
    (``coerce_budget_override`` would otherwise SILENTLY discard a
    NaN/infinite budget downstream) and are returned as floats.  A
    value whose own methods raise (an untrusted number subclass) is
    reported as broken.

    Raises:
        SeaError: Naming *source* (``settings()['timeout']``).
    """
    if key == "locked":
        lockable = [k for k in SETTING_TYPES if k not in META_SETTINGS]
        bad = [item for item in value if not isinstance(item, str) or item not in lockable]
        if bad:
            raise SeaError(
                f"{source} may only name settings keys ({', '.join(lockable)}); got {bad!r}"
            )
        return sorted(set(value))
    if key in ("max_budget", "timeout"):
        try:
            value = float(value)
        except OverflowError:
            value = math.inf
        except BaseException as exc:  # noqa: BLE001 — an untrusted number subclass may raise
            raise SeaError(
                f"{source} returned a broken value: {safe_message(exc)}"
            ) from exc
        if not math.isfinite(value):
            raise SeaError(f"{source} must return a finite number or None")
        if key == "timeout" and value <= 0:
            raise SeaError(f"{source} must be a positive number of seconds, got {value:g}")
    return value
