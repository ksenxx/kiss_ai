# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The contract of an SEA and the loader that executes one.

A SEA configures the session that runs it with optional
module-level functions.  The run parameters come from ``settings()``::

    def settings() -> dict:
        return {"kind": "worker", "tool_profile": "bash", "max_budget": 1.0}

``settings()`` is data: a ``kind`` (``session``, the default, ``worker``
or ``channel``: a named dict of defaults laid under the explicit keys,
see :func:`kind_defaults`), optionally ``extends`` (the command name or
path of a base script whose configuration this one refines, see
:func:`kiss.agents.sorcar.sea_commands.sea_layers`), any of the
per-run parameters of :func:`kiss.server.sorcar.run` listed in
:data:`SETTING_TYPES`, and two keys read by the dispatcher:
``timeout`` (seconds a ``run_agent`` call waits for this script's
sub-task, and the limit of each ``run_parallel`` child) and ``locked``
(the keys an explicit caller argument may not replace).  Against the
caller there is one precedence rule, :data:`PRECEDENCE_RULE`, enforced
by :func:`locked_conflicts` and rendered by ``sea docs`` into every
page that states it.

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
Settings honoured   all but ``timeout``   all                     all; a pinned
                                                                  ``use_worktree`` /
                                                                  ``auto_commit`` /
                                                                  ``auto_classify: True``
                                                                  or ``chat_id`` is
                                                                  refused with an error
Parent inheritance  none (tab settings)   yes, unless the kind    yes; budget =
                                          is ``channel`` or the   remaining / (N+1)
                                          ``inherit`` option
                                          is ``false``
``timeout``         none                  argument > setting      argument > setting
                                          > 3600                  > none (per child)
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

import ast
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
    "kind": str,
    "extends": str,
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
    "allow_fan_out": bool,
    "tool_profile": str,
    "docker_image": str,
    "timeout": (int, float),
    "locked": list,
    "hidden": bool,
}
"""Every key ``settings()`` may return, with the type its value must have."""

SETTING_DOCS: dict[str, str] = {
    "kind": "What the run is: `session` (the default, an ordinary Sorcar session), `worker` or "
            "`channel`; each is a dict of defaults laid under the explicit keys (see the kind "
            "table). A `channel` run holds its channel workspace, gets the channel preamble, "
            "never inherits from a calling task and is never a `run_parallel` child or an "
            "`extends` base.",
    "extends": "A base script (command name or `.py` path) whose layers run under this one: "
               "settings merge with the later layer winning, `prompt(task)` functions chain, "
               "system-prompt additions concatenate, tools union.",
    "work_dir": "The directory the run works in; default: the calling task's or the tab's.",
    "model": "The LLM model, a catalogue name or a model-picker SEA; `\"\"` or `None` keeps "
             "the caller's.",
    "chat_id": "The chat the run's events go to; default: a new chat.",
    "use_worktree": "Run in a git worktree of the project (daemon default `True`).",
    "auto_commit": "Commit the run's changes when it ends (daemon default `True`).",
    "max_budget": "USD budget of the run, a finite number; default: the caller's share or the "
                  "daemon's default.",
    "model_config": "Model configuration dict passed to the LLM (temperature, base URL, ...).",
    "use_web_tools": "Give the run the browser tools (daemon default: on).",
    "auto_classify": "Let the pre-run classifier decide the worktree mode and lite prompt "
                     "(daemon default: the persisted setting).",
    "use_memory": "Give the run the `memory_*` tools (daemon default: the persisted setting).",
    "allow_fan_out": "Let the run call `run_parallel` (default `True`).",
    "tool_profile": "The run's toolset: `review`, `bash`, `shell+edit`, ... (default: the full "
                    "toolset).",
    "docker_image": "Run inside this Docker image (default: the host).",
    "timeout": "Seconds a `run_agent` call waits for the run (default 3600) and the limit of each "
               "`run_parallel` child (default none); ignored by `/<name>`.",
    "locked": "Keys an explicit `run_agent` / `run_parallel` argument or option may not change: "
              "a differing value is an error.",
    "hidden": "`True`: the script is no `/command` and no `run_agent` agent name (loadable by "
              "path and as an `extends` base only); must be the literal `True` in `settings()`.",
}
"""One line of documentation per :data:`SETTING_TYPES` key (rendered by ``sea docs``)."""

META_SETTINGS = ("kind", "extends", "locked", "hidden")
"""Keys that shape the settings themselves rather than the run; never lockable."""

DISPATCHER_SETTINGS = ("kind", "extends", "timeout", "locked", "hidden")
"""``settings()`` keys with no ``run`` command wire field.

``kind`` and ``extends`` are resolved by :func:`resolve_settings` and
:func:`~kiss.agents.sorcar.sea_commands.sea_layers` (the daemon reads
``kind`` from the layers, :mod:`kiss.server.agent_file`); ``timeout``
is read by the dispatcher (:mod:`kiss.agents.sorcar.agent_dispatch`);
``locked`` names the keys an explicit caller argument may not replace
(:func:`locked_conflicts`); ``hidden`` keeps the script out of the
command registry (:func:`declares_hidden`).  Every other key is a parameter of
:func:`kiss.server.sorcar.run`, sent as :func:`wire_field` of the key.
"""

RENAMED_SETTINGS = {
    "is_parallel": "allow_fan_out",
    "classify_tasks": "auto_classify",
    "preset": "kind",
}
"""Former settings keys and their current names.

``allow_fan_out`` says whether the run may call ``run_parallel`` (the
wire field ``isParallel``); ``auto_classify`` whether the daemon's
classifier decides the run's worktree mode (``classifyTasks``);
``kind`` is the one axis that used to be split into ``preset``
(``session`` / ``worker`` / ``channel``) and ``kind`` (``agent`` /
``channel``).  The old names are refused with a message naming the
new one; ``sea lint --fix`` rewrites them in a script.
"""

REMOVED_SETTINGS = {
    "inherit": "a `channel` run never inherits from the calling task and every other kind "
               "always does; the caller's `inherit` option opts out of inheriting",
}
"""Former settings keys with no replacement, and why; refused with the explanation."""

KINDS = ("session", "worker", "channel")
"""The values of the ``kind`` setting (see :func:`kind_defaults`)."""

WORKER_DEFAULTS: dict[str, Any] = {
    "use_worktree": False,
    "auto_commit": False,
    "auto_classify": False,
    "allow_fan_out": False,
    "use_web_tools": False,
    "use_memory": False,
}
"""The defaults of the ``worker`` kind (and, with a ``work_dir``, of ``channel``)."""

KIND_DOCS: dict[str, str] = {
    "session": "The default: an ordinary Sorcar session with the caller's or the user's "
               "settings (`/write`, `/write_paper`, `bestrouter`).",
    "worker": "A focused tool-bound run on the caller's tree: no worktree, no auto-commit, no "
              "classifier, no fan-out, no browser, no memory (`/sh`, `/ask`, `/merge`, "
              "`/remember`, `/forget`, `/task_update`).",
    "channel": "A worker for an external service, in the shared `channel_work` scratch "
               "directory under the Sorcar home, never the caller's project; it holds its "
               "channel workspace, gets the channel preamble and never inherits from a calling "
               "task (every bundled channel agent and `/cron`).",
}
"""One line of documentation per kind (rendered by ``sea docs``)."""


def kind_defaults() -> dict[str, dict[str, Any]]:
    """Return the dict of defaults each ``kind`` lays under the script's explicit keys.

    ``session`` is empty: the run is an ordinary Sorcar session with
    the caller's or the user's settings.  ``worker`` is a focused
    tool-bound run on the caller's tree: no worktree, no auto-commit,
    no classifier, no fan-out, no browser, no memory.  ``channel`` is
    a worker for an external service with a ``work_dir`` of the shared
    ``channel_work`` scratch directory under the Sorcar home (never the caller's
    project, whose git lifecycle it does not join); the daemon and the
    dispatcher give a channel its workspace and preamble and never let
    it inherit from a calling task.  Computed on every call so a
    redirected ``$KISS_HOME`` is honoured.
    """
    return {
        "session": {},
        "worker": dict(WORKER_DEFAULTS),
        "channel": {**WORKER_DEFAULTS, "work_dir": str(kiss_home() / "channel_work")},
    }


class SeaError(Exception):
    """Base of every "this SEA is broken" error.

    :exc:`kiss.agents.sorcar.sea_commands.SeaScriptError` (raised by the
    registry and the dispatcher) and
    :exc:`kiss.server.agent_file.AgentFileError` (raised by the daemon's
    task runner) both derive from it, so a caller that only wants to
    know "the script failed" catches one class.
    """


def wire_field(key: str) -> str:
    """Return the ``run`` command wire field of the ``run()`` keyword *key*.

    The wire vocabulary is the keyword vocabulary in camelCase
    (``use_web_tools`` -> ``useWebTools``), with four aliases kept from
    the wire protocol's earlier vocabulary: ``add_to_prompt`` ->
    ``appendToPrompt``, ``add_to_system_prompt`` ->
    ``appendToSystemPrompt``, ``allow_fan_out`` -> ``isParallel`` and
    ``auto_classify`` -> ``classifyTasks``.
    """
    aliases = {
        "add_to_prompt": "appendToPrompt",
        "add_to_system_prompt": "appendToSystemPrompt",
        "allow_fan_out": "isParallel",
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

    The one loader of SEAs (the daemon's ``agentPath``, the
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


class SettingsError(SeaError, ValueError):
    """A SEA's settings are malformed (wrong type, unknown key or kind)."""


def declares_hidden(path: Path) -> bool:
    """Return whether the SEA at *path* writes ``"hidden": True`` in ``settings()``.

    Read from the source (``ast``), never by executing the script: the
    command registry calls this for every scanned folder, and a hidden
    SEA (a test fixture, protocol plumbing, a base other SEAs extend)
    must be excluded without running it.  Hence the contract that
    ``hidden`` is a literal ``True`` in a dict inside ``settings()``; a
    computed value is accepted by :func:`resolve_settings` but does not
    hide the SEA.
    """
    return declared_literal(path, "hidden") is True


def declared_literal(path: Path, key: str) -> Any:
    """Return the literal value ``settings()`` of the SEA at *path* writes for *key*, or ``None``.

    Parsed from the source, never executed (see :func:`declares_hidden`);
    only a constant value under a string-literal key in a dict inside
    ``settings()`` counts.  The parse is cached per path until the
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
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "settings":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for key, value in zip(sub.keys, sub.values, strict=True):
                        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                            literals[str(key.value)] = value.value
    return literals


def resolve_settings(namespace: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective settings of the SEA executed into *namespace*.

    Evaluates the script's ``settings()`` (when defined) and merges the
    ``kind``'s defaults under its keys.  Every value is
    type-checked against :data:`SETTING_TYPES`.  ``work_dir`` and
    ``extends`` are kept as the script returned them; the daemon and
    :func:`kiss.agents.sorcar.sea_commands.sea_layers` resolve them.

    Args:
        namespace: The script's module namespace (``module.__dict__``).

    Returns:
        A new dict: ``{"kind": name, <key>: value, ...}`` with the
        kind's defaults already merged in under the explicit keys.
        A key whose value is ``None`` is dropped — as is a ``model`` of
        ``""`` — it means "no override", so the caller's or the
        persisted value stands.

    Raises:
        SettingsError: When ``settings()`` is not a function returning a
            dict, names an unknown, renamed or removed key or an
            unknown ``kind``, a value has the wrong type, or
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
        if key in RENAMED_SETTINGS:
            raise SettingsError(
                f"settings() key {key!r} was renamed to {RENAMED_SETTINGS[key]!r}; "
                f"run `uv run sea lint --fix` to rewrite the script"
            )
        if key in REMOVED_SETTINGS:
            raise SettingsError(f"settings() key {key!r} was removed: {REMOVED_SETTINGS[key]}")
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
    kind = declared.get("kind", "session")
    return {"kind": kind, **kind_defaults()[kind], **declared}


def merge_settings(chain: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the effective settings of a script and the bases it extends.

    *chain* lists resolved settings (:func:`resolve_settings`) from the
    outermost base to the script itself; a later entry's key wins.  The
    effective ``kind`` is the last one other than ``session``
    (``session`` changes nothing, so it never masks a base's kind).
    ``extends`` is dropped (the chain has resolved it) and ``hidden``
    is the script's own, never a base's.

    Args:
        chain: The resolved settings, base first.

    Returns:
        The merged settings dict; ``{"kind": "session"}`` for an
        empty chain.
    """
    merged: dict[str, Any] = {}
    for settings in chain:
        merged.update(settings)
    merged["kind"] = next(
        (s["kind"] for s in reversed(chain) if s["kind"] != "session"), "session",
    )
    merged.pop("extends", None)
    # ``hidden`` describes one script, not what extends it: a hidden
    # base is the normal case, and its children are ordinary commands.
    merged.pop("hidden", None)
    if chain and "hidden" in chain[-1]:
        merged["hidden"] = chain[-1]["hidden"]
    locks = {key for s in chain for key in s.get("locked", ())}
    if merged.get("kind") == "channel":
        # A channel is closed: it runs in its own scratch directory, never
        # the caller's, and no call may give it a worktree, auto-commit,
        # the classifier, fan-out, the browser or memory.
        locks.update(key for key in kind_defaults()["channel"] if key in merged)
    if locks:
        # A base that locks a key keeps it locked in every script extending it.
        merged["locked"] = sorted(locks)
    return merged


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
    reported as broken.  ``kind`` must be one of :data:`KINDS`.

    Raises:
        SettingsError: Naming *source* (``settings()['timeout']``).
    """
    if key == "kind" and value not in KINDS:
        raise SettingsError(f"{source} must be one of {', '.join(KINDS)}; got {value!r}")
    if key == "locked":
        lockable = [k for k in SETTING_TYPES if k not in META_SETTINGS]
        bad = [item for item in value if not isinstance(item, str) or item not in lockable]
        if bad:
            raise SettingsError(
                f"{source} may only name settings keys ({', '.join(lockable)}); got {bad!r}"
            )
        return sorted(set(value))
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
        if key == "timeout" and value <= 0:
            raise SettingsError(f"{source} must be a positive number of seconds, got {value:g}")
    return value


def _call(fn: Any, label: str) -> Any:
    """Call *fn*; anything it raises becomes a :exc:`SettingsError` naming *label*."""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 — untrusted script code may raise anything
        raise SettingsError(f"{label} raised: {safe_message(exc)}") from exc
