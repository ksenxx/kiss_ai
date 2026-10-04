# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``run_agent`` and ``agent_job`` tools: run any agent script (SEA) on a task, now.

:func:`make_run_agent_tool` builds ``run_agent`` per task.  Its
``agent`` argument names what to run (:func:`resolve_agent`): empty
for a plain sub-agent, a ``.py`` path, or the name of a registered
slash command (:func:`kiss.agents.sorcar.sea_commands.list_commands`:
the bundled SEAs such as ``write_paper``, the channel SEAs such as
``slack`` and ``gmail``, ``cron``, and the user's own ``SEAS.md``
folders).  Every spelling resolves to one agent-script path and takes
one dispatch path (:func:`_run_agent`):

1. the script's effective ``settings()`` are read in the calling
   process (:func:`kiss.agents.sorcar.sea_commands.sea_settings`) for
   the ``timeout``, ``inherit`` and ``work_dir`` the dispatcher needs;
2. unless ``inherit`` is ``False``, the arguments the call left empty
   are inherited from the calling agent (:func:`inherit_from_parent`:
   model, budget share, chat, prompt suffixes, web/memory flags,
   container, worktree/auto-commit choices, fan-out flag, extra tools);
3. the sub-task is submitted to the kiss-web daemon
   (:func:`kiss.agents.sorcar.daemon_client.run` with the script as
   ``extension_agent_path``) as a nested sub-agent of the calling task,
   and the call blocks for its result — or, with ``wait="false"``,
   returns a job id at once for :func:`agent_job` to wait on, check or
   kill.

The daemon executes the script once more, applies its settings and
getters (:mod:`kiss.agents.sorcar.sea_settings`,
:mod:`kiss.server.agent_file`) and, for ``kind: "channel"``, holds the
channel workspace (the ``workspace`` option, forwarded as a wire
field) for the run's lifetime.

One precedence rule holds for every setting of the sub-task: the agent
script's ``settings()`` win, then the tool's arguments, then what the
calling task passes on, then the user's persisted settings.  The one
exception is ``timeout``: an explicit argument beats the script's
setting, because it bounds the CALLER's wait.  Inside the kiss-web
daemon the sub-task is submitted back through the daemon's own local
endpoint (recorded at boot by the cron scheduler); standalone runs use
the standard endpoint resolution and need a reachable daemon.
"""

import dataclasses
import difflib
import importlib.util
import json
import logging
import math
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.daemon_client import TaskResult
from kiss.agents.sorcar.sea_commands import sea_script_in
from kiss.agents.sorcar.sea_settings import SETTING_TYPES, safe_message, script_name
from kiss.agents.sorcar.useful_tools import (
    remap_vanished_worktree,
    rewrite_parent_repo_paths,
)
from kiss.core.config import DEFAULT_CONFIG, kiss_home
from kiss.core.vscode_config import load_config

logger = logging.getLogger(__name__)

DEFAULT_AGENT_PATH = str(
    Path(__file__).resolve().parents[1] / "seas" / "dummy" / "dummy_sea.py"
)
"""Agent script run when the ``run_agent`` tool's ``agent`` is empty.

The bundled ``src/kiss/agents/seas/dummy/dummy_sea.py`` — an SEA that defines
no getters, so the sub-task is a plain Sorcar session on the given task
in the calling task's work directory (path mode, with the standard
worktree/auto-commit lifecycle).  Held as the absolute path of the
installed file so the default works from any work directory, not only
a checkout of this repository.
"""

DEFAULT_DISPATCH_TIMEOUT_SECONDS = 3600.0
"""Default bound on the wait for a dispatched sub-task's result.

Used when the ``run_agent`` tool's ``timeout`` argument is empty and
the agent script's ``settings()`` declares no ``timeout`` (see
:func:`resolve_timeout`); a per-call value overrides both.  One hour
rather than minutes: a sub-task that writes a paper or runs a test
suite legitimately takes that long, and a stopped sub-task loses its
result, so the default errs towards waiting.  When the wait times out, the tool
returns an error string and the sub-task is STOPPED
(:func:`kiss.agents.sorcar.daemon_client.run` is called with
``stop_on_timeout=True``, which also awaits the stop's
terminal-status confirmation before returning; if a wedged daemon
never confirms it within the bounded grace, the error string says the
task may still be running instead of claiming it was stopped): a
surviving sub-task would keep spending invisibly.  Work the sub-task
completed before the stop (side effects) is not reported back to the
calling task; its spend, carried by the stopped task's final result,
is charged to the calling task.
"""



def stop_unconfirmed_error(name: str, timeout: float) -> str:
    """Return the ``run_agent`` error string for an unconfirmed stop.

    Returned by the tool exactly when the dispatch timed out AND the
    daemon never confirmed the requested stop
    (``daemon_client.StopUnconfirmedTimeoutError``), so the sub-task may
    still be running.  Programmatic callers of the tool
    (``cron_agent._run_prompt_job``) compare the reply against this
    exact string — never a substring, which unrelated text such as an
    endpoint URL or file path in a connection error could contain — to
    keep the run's
    scratch directory instead of deleting it under a possibly live task.

    Args:
        name: The dispatched agent's display name (the script's stem or
            the channel name).
        timeout: The wait bound in seconds that expired.

    Returns:
        The complete error string.
    """
    return (
        f"Error: the {name} agent task did not finish within "
        f"{timeout:g}s; a stop was requested but the daemon never "
        f"confirmed it, so the task MAY STILL BE RUNNING (and "
        f"spending) on the daemon. Check what it already did "
        f"before retrying with a larger `timeout` argument."
    )

_NON_CHANNEL_MODULES = frozenset({"a2a", "oai"})
"""SEA folders of the third-party package that are not user-facing channels.

``a2a`` (agent-to-agent protocol plumbing) and ``oai`` (an
OpenAI-compatible HTTP server) live in the package for infrastructure
reasons but are not services a user asks Sorcar to act on, so they are
hidden from the tool.
"""


@dataclass(frozen=True)
class RunOptions:
    """Optional per-run overrides of a dispatched sub-task.

    The parsed form of the ``run_agent`` tool's ``options`` argument:
    one field per key of :data:`OPTION_TYPES` — the agent-script
    settings vocabulary (:data:`~kiss.agents.sorcar.sea_settings.SETTING_TYPES`)
    minus what the tool takes as its own arguments — plus
    ``system_prompt``, the replacement base system prompt a programmatic
    caller may pass (it is not an ``options`` key: a script's
    ``system_prompt()`` is the user-facing way).  ``None`` / empty means
    "not passed": the calling agent's own value applies where it has
    one (see :func:`inherit_from_parent`), otherwise the persisted
    setting or the daemon's default decides.  An agent script's
    ``settings()`` still win over every value here on the daemon.
    """

    work_dir: str = ""
    chat_id: str = ""
    use_worktree: bool | None = None
    auto_commit: bool | None = None
    model_config: dict[str, Any] | None = None
    use_web_tools: bool | None = None
    classify_tasks: bool | None = None
    use_memory: bool | None = None
    is_parallel: bool | None = None
    tool_profile: str = ""
    docker_image: str = ""
    inherit: bool | None = None
    workspace: str = ""
    add_to_prompt: str = ""
    add_to_system_prompt: str = ""
    system_prompt: str = ""


_TOOL_ARGUMENT_SETTINGS = frozenset({
    "preset", "extends", "kind", "timeout", "model", "max_budget", "tool_profile",
})
"""Settings keys the ``options`` JSON object does not accept.

``preset``, ``extends`` and ``kind`` describe a script, not a call;
``timeout``, ``model``, ``max_budget`` and ``tool_profile`` are the
tool's own arguments.
"""

OPTION_TYPES: dict[str, type] = {
    **{
        key: expected
        for key, expected in SETTING_TYPES.items()
        if key not in _TOOL_ARGUMENT_SETTINGS and isinstance(expected, type)
    },
    "workspace": str,
    "add_to_prompt": str,
    "add_to_system_prompt": str,
}
"""The keys the ``options`` JSON object accepts, with the type of each value.

Derived from :data:`~kiss.agents.sorcar.sea_settings.SETTING_TYPES`,
so the tool and the scripts share one vocabulary: what a script may
pin in ``settings()``, a caller may pass in ``options``; plus
``workspace`` (the account a channel agent's run holds),
``add_to_prompt`` (text appended to the task) and
``add_to_system_prompt``, the option form of a script's
``add_to_system_prompt()`` getter.
"""

FANOUT_REFUSED: dict[str, Any] = {"use_worktree": True, "auto_commit": True, "classify_tasks": True}
"""Setting values a ``run_parallel`` child cannot honour (see :func:`fanout_conflict`)."""


def fanout_conflict(values: Mapping[str, Any]) -> str:
    """Return why a ``run_parallel`` child cannot run with *values*, or ``""``.

    A fan-out child is a thread of the caller acting on the caller's
    tree in the caller's chat: it can take no worktree, no auto-commit,
    no classifier and no other chat, and a channel agent (which holds a
    workspace and inherits nothing) runs through ``run_agent`` only.
    The one rule for a script's ``settings()`` and a call's ``options``
    alike, so a pinned value is refused loudly instead of ignored.

    Args:
        values: Merged settings or parsed options; a ``chat_id`` key
            present with any value counts as pinned.

    Returns:
        The reason, or ``""`` when the values fit a fan-out child.
    """
    if values.get("kind") == "channel":
        return "is a channel agent, which run_parallel cannot run; use run_agent"
    for key, refused in FANOUT_REFUSED.items():
        if values.get(key) is refused:
            return (
                f"pins {key}: true, which a run_parallel child (a thread of the caller "
                f"on its own tree) cannot honour; use run_agent"
            )
    if "chat_id" in values:
        return "pins chat_id, but a run_parallel child runs in the caller's chat; use run_agent"
    if values.get("workspace"):
        return "names a workspace, which only a channel agent's run_agent dispatch holds"
    return ""

_RENAMED_OPTIONS = {
    "append_to_prompt": "add_to_prompt",
    "append_to_system_prompt": "add_to_system_prompt",
}
"""Former ``options`` keys and their current names (one release of pointed errors)."""


def _parse_bool(name: str, value: Any) -> bool | None:
    """Parse an optional boolean option.

    Args:
        name: The option's name, for the error message.
        value: A JSON boolean, the word ``"true"`` / ``"false"`` (any
            case, surrounding whitespace ignored) or empty / ``None``
            for "not passed".

    Returns:
        The boolean, or ``None`` when *value* is empty.

    Raises:
        ValueError: When *value* is neither empty nor a boolean.
    """
    if value is None or isinstance(value, bool):
        return value
    word = str(value).strip().lower()
    if not word:
        return None
    if word in ("true", "false"):
        return word == "true"
    raise ValueError(f"{name} must be true or false, got {value!r}.")


def parse_run_options(options: str, tool_profile: str = "") -> RunOptions:
    """Parse the ``options`` argument of ``run_agent`` / ``run_parallel``.

    Args:
        options: A JSON object string whose keys are keyword
            parameters of :func:`kiss.server.sorcar.run`
            (:data:`OPTION_TYPES`), or empty for no overrides.
            Booleans may also be given as the strings ``"true"`` /
            ``"false"``; ``null`` means "not passed".
        tool_profile: The tool's own ``tool_profile`` argument, carried
            on the returned options; validated against
            :func:`kiss.agents.sorcar.sorcar_agent.resolve_tool_profile`.

    Returns:
        The parsed options.

    Raises:
        ValueError: When *options* is not a JSON object, names an
            unknown key, has a value of the wrong type, or
            *tool_profile* is unknown.
    """
    from kiss.agents.sorcar.sorcar_agent import resolve_tool_profile

    resolve_tool_profile(tool_profile)
    if not options.strip():
        return RunOptions(tool_profile=tool_profile.strip())
        return RunOptions()
    try:
        raw = json.loads(options)
    except ValueError as e:
        raise ValueError(f"options must be a JSON object, got {options!r}: {e}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"options must be a JSON object, got {options!r}.")
    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _RENAMED_OPTIONS:
            raise ValueError(
                f"options key {key!r} was renamed to {_RENAMED_OPTIONS[key]!r}."
            )
        expected = OPTION_TYPES.get(key)
        if expected is None:
            raise ValueError(
                f"options has an unknown key {key!r}; known keys: "
                f"{', '.join(OPTION_TYPES)}."
            )
        if value is None:
            continue
        if expected is bool:
            parsed[key] = _parse_bool(key, value)
        elif isinstance(value, str) and expected is str:
            parsed[key] = (
                value.strip() if key in ("chat_id", "work_dir", "workspace") else value
            )
        elif isinstance(value, expected) and not isinstance(value, bool):
            parsed[key] = value
        else:
            raise ValueError(
                f"options[{key!r}] must be a JSON {expected.__name__}, "
                f"got {type(value).__name__}."
            )
    return RunOptions(tool_profile=tool_profile.strip(), **parsed)


def _package_dir() -> Path | None:
    """Return the directory of the third-party agents package.

    Located through the import system without importing the package's
    (heavy, optional) modules.

    Returns:
        The package directory, or ``None`` when the package is absent.
    """
    try:
        spec = importlib.util.find_spec("kiss.agents.third_party_agents")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def available_channels() -> list[str]:
    """Return the names of the installed third-party channel agents.

    A channel is any SEA folder ``<channel>/<channel>_sea.py`` in the
    third-party agents package (private ``_``-prefixed folders and the
    known non-channel infrastructure SEAs excluded).  The scan reads
    the directory listing only — no channel module is imported.

    Returns:
        Sorted channel names, e.g. ``["discord", ..., "slack", ...]``;
        empty when the package is absent.
    """
    package_dir = _package_dir()
    if package_dir is None:
        return []
    return sorted(
        sea_dir.name
        for sea_dir in package_dir.iterdir()
        if not sea_dir.name.startswith("_")
        and sea_dir.name not in _NON_CHANNEL_MODULES
        and sea_script_in(sea_dir).is_file()
    )


def _squash(name: str) -> str:
    """Normalize a channel name for forgiving lookup.

    Case, spaces, hyphens, and underscores are ignored, so
    ``"Home Assistant"``, ``"home-assistant"`` and ``"HOMEASSISTANT"``
    all match the ``homeassistant`` channel.

    Args:
        name: A user- or model-supplied channel name.

    Returns:
        The lowercase name with separator characters removed.
    """
    return re.sub(r"[\s\-_]+", "", name.strip().lower())


def _daemon_endpoint_file() -> str | None:
    """Return the endpoint file of the daemon hosting this process, if any.

    Inside the kiss-web daemon the cron scheduler records the daemon's
    own endpoint file at boot; dispatched sub-tasks must go back
    through it.  Standalone (no scheduler running in this process)
    returns ``None`` and :func:`kiss.server.sorcar.run` applies its
    standard endpoint resolution.

    Returns:
        The daemon endpoint file, or ``None`` when not inside a daemon.
    """
    from kiss.agents.sorcar import cron_agent

    return cron_agent._daemon_endpoint_file


def _attribute_dispatch_usage(parent_agent: Any, result: Any, epoch: Any = None) -> None:
    """Fold a dispatched sub-task's spend into the calling agent.

    The daemon's terminal ``result`` event carries the sub-task's
    cost, tokens, and steps (parsed into the
    :class:`~kiss.agents.sorcar.daemon_client.TaskResult`).  Without
    this fold, that spend would vanish from the calling task's
    accounting — the parent's end-of-task cost, its live usage
    header, and its persisted per-task cost would all lie low —
    exactly the gap :func:`~kiss.agents.sorcar.sorcar_agent._attribute_sub_usage`
    already closes for ``run_parallel`` sub-agents and ``talk`` TTS
    calls.

    Args:
        parent_agent: The agent that called ``run_agent``; ``None``
            (standalone use, where no calling agent exists)
            attributes nothing.
        result: The dispatched sub-task's ``TaskResult``; ``None``
            (an aborted wait that saw no spend) attributes nothing.
        epoch: The parent's usage-ledger epoch captured when the
            dispatch started (:func:`dispatch_epoch`), so a background
            job that finishes after the parent's run ended settles into
            that run's ledger, never into a later run's; ``None`` for
            the parent's current epoch.
    """
    if parent_agent is None or result is None:
        return
    try:
        from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

        _attribute_sub_usage(
            parent_agent,
            float(getattr(result, "cost", 0.0) or 0.0),
            int(getattr(result, "tokens", 0) or 0),
            int(getattr(result, "steps", 0) or 0),
            epoch=epoch,
        )
    except Exception:  # pragma: no cover — attribution must never break dispatch
        logger.warning("dispatched sub-task usage attribution failed", exc_info=True)


def dispatch_epoch(parent_agent: Any) -> Any:
    """Return *parent_agent*'s current usage-ledger epoch token, or ``None``.

    Captured when a dispatch starts and bound to every spend
    attribution it makes, so a sub-task that outlives the calling
    run (a ``run_agent(wait="false")`` job the model never waited for)
    charges the run that started it (see
    ``RelentlessAgent._usage_epoch``).
    """
    epoch_of = getattr(parent_agent, "_usage_epoch", None)
    return epoch_of() if callable(epoch_of) else None


@dataclass(frozen=True)
class Inherited:
    """What a path-mode sub-task takes over from the calling agent.

    The values :func:`inherit_from_parent` resolves for the arguments
    the ``run_agent`` call left empty — the same inheritance a
    ``run_parallel`` child gets from its parent
    (``SorcarAgent._run_tasks_parallel``), so a sub-task dispatched
    through the daemon behaves like an in-process sub-agent of the
    caller instead of like a task typed into a fresh chat panel.
    """

    model_name: str
    budget: float | None
    options: RunOptions
    docker_image: str
    use_worktree: bool | None
    auto_commit: bool | None


def _parent_model_config(
    parent_agent: Any, model_name: str, script_picks_model: bool,
) -> dict[str, Any] | None:
    """Return the calling agent's model configuration if it fits the sub-task's model.

    A Sorcar agent keeps the ``model_config`` its ``run`` received as
    the public ``model_config`` attribute; it was supplied for the
    model the run was LAUNCHED with (``_launch_model_name``), and a
    later ``set_model`` switch leaves it in place while changing
    ``model_name``.  An endpoint and its key belong to that launch
    model, so the configuration is returned only when *model_name* is
    the launch model (``model_name`` when the agent records no launch
    model) AND the agent script does not pick the model itself: the
    daemon applies a script's ``model`` setting on top of the wire
    fields without touching ``modelConfig`` (``apply_agent_overrides``),
    so a script-chosen model would otherwise run against the caller's
    endpoint.  An empty configuration is reported as ``None``.

    Args:
        parent_agent: The agent calling ``run_agent``.
        model_name: The model the sub-task will run unless the script
            overrides it.
        script_picks_model: Whether the agent script's ``settings()``
            name a ``model``.

    Returns:
        A copy of the configuration dict, or ``None``.
    """
    launch_model = str(
        getattr(parent_agent, "_launch_model_name", None)
        or getattr(parent_agent, "model_name", "")
        or ""
    )
    if not model_name or model_name != launch_model or script_picks_model:
        return None
    config = getattr(parent_agent, "model_config", None)
    return dict(config) if isinstance(config, dict) and config else None


def inherit_from_parent(
    parent_agent: Any,
    model_name: str,
    budget: float | None,
    options: RunOptions,
    script_picks_model: bool = False,
) -> Inherited:
    """Fill the empty ``run_agent`` arguments from the calling agent.

    The ONE inheritance table of a sub-agent: ``run_agent`` dispatches
    and ``run_parallel`` fan-outs (``SorcarAgent._run_tasks_parallel``)
    both build their children's arguments here.  An explicit argument
    always wins; only an empty one is inherited:

    - ``model_name``: the caller's model.
    - ``model_config``: the caller's, but ONLY when the sub-task runs
      the model the caller was launched with and the agent script
      does not pick its own model — an endpoint and its key belong to
      that model, so a different model (one the caller switched to
      with ``set_model``, or one the script's ``model`` setting
      selects on the daemon) runs with default provider routing (see
      :func:`_parent_model_config`).
    - ``budget``: the caller's remaining budget split as for a
      one-task fan-out (``_subagent_budget_share(1)``: half of what is
      left, the other half reserved for the caller to process the
      result), ``None`` (daemon default) when the caller has no budget
      context yet.
    - ``chat_id``: the caller's chat, so the sub-task starts with the
      conversation's earlier tasks and results as context.
    - ``system_prompt`` / ``add_to_system_prompt``: the caller's
      own replacement base prompt (``_base_system_prompt``, blank
      unless its run was given one — the classifier's SYSTEM vs
      SYSTEM_LITE choice is never stored there, so the sub-task is
      still classified on its own) and append-only suffix
      (``_system_prompt_suffix``), so a run's extra system
      instructions constrain its whole task tree through ``run_agent``
      exactly as through ``run_parallel``.  An agent script's
      ``system_prompt()`` still replaces the base prompt on the
      daemon, and its ``add_to_system_prompt()`` text is added after
      the inherited suffix.
    - ``add_to_prompt``: the suffix the caller's own task prompt
      was given (``_prompt_suffix``, the ``appendToPrompt`` of its
      run), so the sub-task's prompt ends with the same text.  An
      agent script's ``prompt(task)`` getter then rewrites the whole
      task text on the daemon.
    - ``is_parallel``: whether the caller may fan out itself
      (``_is_parallel``), so a sequential caller (a ``worker`` preset,
      a user who turned fan-out off) does not hand ``run_parallel``
      back to its children.
    - the caller's extra tools (its agent script's ``add_to_tools()``
      list, plus those the caller inherited itself): not resolved
      here — a callable cannot travel the wire — but requested from
      the daemon with the ``inherit_tools`` flag :func:`dispatch_result`
      sends, so the sub-task has the tools the inherited system prompt
      refers to.  The daemon adds them to the sub-task's built-in
      toolset after the sub-task's own script's ``add_to_tools()``
      tools; a script on the ``none`` tool profile keeps exactly its
      own set.
    - ``use_web_tools`` / ``use_memory``: the caller's per-run
      settings (``_use_web_tools`` / ``_use_memory_override``).
    - ``docker_image``: the call's own ``docker_image`` option, else
      ``container:<id>`` of the caller's live Docker
      container, so the sub-task acts in the same container instead of
      on the host.
    - ``use_worktree`` / ``auto_commit``: the caller's EFFECTIVE
      choices for its own run (``use_worktree_enabled`` after the
      classifier's demotion, ``auto_commit_enabled``), ``None`` when
      the caller does not carry them (a plain ``ChatSorcarAgent`` or a
      caller that has not run yet), in which case the persisted
      settings apply as before.  Both are ``False`` when the caller's
      container is inherited: an attached ``DockerManager`` works in
      the container's working directory — the caller's mounted tree —
      so a worktree created on the host for the sub-task would never
      be the directory its tools act in; the sub-task works in the
      caller's tree like a ``run_parallel`` child and the caller's own
      lifecycle commits the result.

    Args:
        parent_agent: The agent calling ``run_agent``; ``None`` (no
            caller) inherits nothing.
        model_name: The call's ``model_name`` argument; empty inherits.
        budget: The call's parsed ``max_budget``; ``None`` inherits.
        options: The call's parsed optional arguments.
        script_picks_model: Whether the sub-task's agent script names
            a ``model`` in its ``settings()``, which blocks the
            ``model_config`` inheritance.

    Returns:
        The resolved values.

    Raises:
        BudgetExceededError: When the budget is inherited and the
            caller has nothing left to spend (the same signal a
            ``run_parallel`` fan-out raises).
    """
    if parent_agent is None:
        return Inherited(model_name, budget, options, options.docker_image, None, None)
    model_name = model_name or str(getattr(parent_agent, "model_name", "") or "")
    model_config = options.model_config
    if model_config is None:
        model_config = _parent_model_config(parent_agent, model_name, script_picks_model)
    if budget is None:
        share: Callable[[int], float | None] | None = getattr(
            parent_agent, "_subagent_budget_share", None,
        )
        if callable(share):
            budget = share(1)
    options = dataclasses.replace(
        options,
        chat_id=options.chat_id or str(getattr(parent_agent, "_chat_id", "") or ""),
        system_prompt=(
            options.system_prompt
            or str(getattr(parent_agent, "_base_system_prompt", "") or "")
        ),
        add_to_system_prompt=(
            options.add_to_system_prompt
            or str(getattr(parent_agent, "_system_prompt_suffix", "") or "")
        ),
        add_to_prompt=(
            options.add_to_prompt
            or str(getattr(parent_agent, "_prompt_suffix", "") or "")
        ),
        model_config=model_config,
        is_parallel=(
            getattr(parent_agent, "_is_parallel", None)
            if options.is_parallel is None else options.is_parallel
        ),
        use_web_tools=(
            getattr(parent_agent, "_use_web_tools", None)
            if options.use_web_tools is None else options.use_web_tools
        ),
        use_memory=(
            getattr(parent_agent, "_use_memory_override", None)
            if options.use_memory is None else options.use_memory
        ),
    )
    docker_image = options.docker_image
    container = getattr(getattr(parent_agent, "docker_manager", None), "container", None)
    if not docker_image and container is not None and getattr(container, "id", None):
        from kiss.agents.sorcar.docker_manager import ATTACH_PREFIX

        docker_image = ATTACH_PREFIX + str(container.id)
    if docker_image:
        return Inherited(model_name, budget, options, docker_image, False, False)
    use_worktree = getattr(parent_agent, "use_worktree_enabled", None)
    auto_commit = getattr(parent_agent, "auto_commit_enabled", None)
    return Inherited(
        model_name, budget, options, docker_image,
        None if use_worktree is None else bool(use_worktree),
        None if auto_commit is None else bool(auto_commit),
    )


def _dispatch(
    name: str,
    prompt: str,
    agent_path: str,
    work_dir: str,
    model_name: str,
    budget: float | None,
    timeout: float,
    parent_agent: Any = None,
    scope_work_dir: str = "",
    options: RunOptions = RunOptions(),
    inherit: bool = False,
    settings: dict[str, Any] | None = None,
    workspace: str = "",
    cancel: threading.Event | None = None,
    running: threading.Event | None = None,
) -> str:
    """Submit an agent-script task to the kiss-web daemon and wait for its YAML result.

    :func:`dispatch_result` with the same arguments, formatted for the
    calling model: the sub-task's YAML result ("success" and "summary"
    keys), or the error message.
    """
    result = dispatch_result(
        name, prompt, agent_path, work_dir, model_name, budget, timeout,
        parent_agent=parent_agent, scope_work_dir=scope_work_dir,
        options=options, inherit=inherit, settings=settings, workspace=workspace,
        cancel=cancel, running=running,
    )
    if isinstance(result, str):
        return result
    summary = result.text or ("" if result.success else "Task failed")
    return str(yaml.safe_dump(
        {"success": result.success, "summary": summary}, sort_keys=False,
    ))


def dispatch_result(
    name: str,
    prompt: str,
    agent_path: str,
    work_dir: str,
    model_name: str,
    budget: float | None,
    timeout: float,
    parent_agent: Any = None,
    scope_work_dir: str = "",
    options: RunOptions = RunOptions(),
    inherit: bool = False,
    settings: dict[str, Any] | None = None,
    workspace: str = "",
    cancel: threading.Event | None = None,
    running: threading.Event | None = None,
) -> TaskResult | str:
    """Submit an agent-script task to the kiss-web daemon and wait.

    The tail of :func:`_run_agent`: calls
    :func:`kiss.server.sorcar.run` with *agent_path* as its
    *extension_agent_path* and returns the daemon's
    :class:`~kiss.agents.sorcar.daemon_client.TaskResult` (which
    carries the sub-task's persisted ``task_id``) — or a clean error
    string.  It raises only for the inherited-budget case described
    under *inherit*.  Callers that need the sub-task's id
    (rsi7d's clone replays) use this; :func:`_dispatch` formats the
    result for a model.

    Args:
        name: Display name of the agent for error messages (the
            channel name, or the script's file stem).
        prompt: The full prompt for the sub-task.
        agent_path: Absolute path of the agent script.
        work_dir: Working directory for the sub-task; created when
            absent (an agent script's ``work_dir()`` still wins).
        model_name: LLM model for the sub-task; empty for the daemon
            default (an agent script's ``model()`` still wins).
        budget: Per-task USD budget override; ``None`` for the daemon
            default.
        timeout: Maximum seconds to wait for the sub-task's result.
            On timeout the sub-task is stopped and an error string is
            returned (see :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`).
        parent_agent: The agent calling ``run_agent``, when there is
            one: the sub-task's cost/tokens/steps are folded into its
            task accounting (see :func:`_attribute_dispatch_usage`),
            and its persisted task id / frontend tab id make the
            sub-task a nested sub-agent of the calling task (same tab
            behavior as a ``run_parallel`` sub-task).
        scope_work_dir: The CALLING task's work directory, recorded on
            the sub-task's registry tab (``scopeWorkDir``) alongside
            *work_dir* (the channel/cron scratch directory it executes
            in).  Informational: the tab is shown on every client
            regardless.  Empty (standalone use) records nothing.
        options: The caller's optional per-run overrides (the
            ``run_agent`` tool's optional arguments, parsed).  Each
            one replaces the inherited or persisted default of its
            field; the agent script's ``settings()`` still win over
            all of them on the daemon.
        inherit: Whether the arguments left empty are filled from
            *parent_agent* (see :func:`inherit_from_parent`): the
            caller's model (and, for the same model, its model
            configuration), a share of its remaining budget, its chat,
            its web-tools and memory settings, its live Docker
            container, and its effective worktree / auto-commit
            choices.  ``True`` for every ``run_agent`` dispatch except
            a script whose ``inherit`` setting is ``False`` (the
            ``channel`` preset) — a sub-task on the same project is a
            sub-agent of the caller, whereas a channel or cron session
            acts on an external service from a scratch directory on
            the host and must not inherit the caller's chat context or
            container.  ``False`` (the default) also for programmatic
            callers that pass every value explicitly (rsi7d's clone
            replays).  When the budget is inherited and the caller has
            nothing left to spend, ``BudgetExceededError`` propagates —
            the same signal a ``run_parallel`` fan-out raises.
        settings: The agent script's resolved settings
            (:func:`~kiss.agents.sorcar.sea_commands.sea_settings`),
            when the caller has them; a ``model`` in them blocks the
            ``model_config`` inheritance.  ``None`` means unknown.
        workspace: Workspace/account identifier a ``kind: "channel"``
            run holds for its lifetime (the daemon's task runner enters
            it); empty means ``"default"``.  Ignored by other scripts.
        cancel: An event a background job's ``agent_job(..., "kill")``
            sets; the wait then stops the sub-task and returns an
            error string saying so.  ``None`` for a blocking call.

    Returns:
        The sub-task's :class:`TaskResult`, or an error message.
    """
    epoch = dispatch_epoch(parent_agent)
    # The calling task's identity, threaded through the daemon so the
    # dispatched run is a SUB-AGENT of that task: its tab then behaves
    # exactly like a ``run_parallel`` sub-task's (a nested sub-agent
    # tab under the caller's tab via the run's own ``new_tab``
    # broadcast, a history row nested under the calling task, and a
    # ``subagentDone`` when it ends) instead of a top-level tab.
    # Standalone use (no calling agent, or one that has not persisted
    # a task row) dispatches an ordinary top-level task, unchanged.
    # Any child a reviewer dispatches carries the reviewer marker (see
    # kiss.agents.sorcar.fanout_guard) so it gets the same read-only
    # tool profile.
    from kiss.agents.sorcar import daemon_client
    from kiss.agents.sorcar.fanout_guard import is_review_task
    from kiss.agents.sorcar.sorcar_agent import _persisted_task_id

    _is_rev = getattr(parent_agent, "_is_reviewer_subagent", None)
    parent_reviewer = bool(_is_rev()) if callable(_is_rev) else False
    parent_task_id = _persisted_task_id(parent_agent)
    parent_tab_id = ""
    if parent_task_id:
        # The tab the webviews show the caller under (its own tab id,
        # or — when the caller is itself a sub-agent — its
        # ``{parent}__sub_{task}`` tab), the same resolution nested
        # ``run_parallel`` fan-outs use.  A persisted task id
        # proves the caller is a ``ChatSorcarAgent``, which always has
        # the resolver; the guard only covers duck-typed callers.
        resolve_tab = getattr(parent_agent, "_subagent_parent_tab_id", None)
        if callable(resolve_tab):
            parent_tab_id = str(resolve_tab() or "")
    # A caller whose worktree was already torn down (a finished worktree
    # task) hands over the removed directory; creating it here would
    # leave an unregistered husk under ``.kiss-worktrees/``.
    work_dir = str(remap_vanished_worktree(Path(work_dir)))
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    # The arguments the caller left empty come from the calling agent
    # (its model, budget share, chat, web/memory settings, container,
    # effective worktree/auto-commit), the way a ``run_parallel`` child
    # inherits them.  Channel-preset dispatches and explicit
    # programmatic callers skip this.
    inherited = inherit_from_parent(
        parent_agent if inherit else None, model_name, budget, options,
        script_picks_model="model" in (settings or {}),
    )
    model_name, budget, options = inherited.model_name, inherited.budget, inherited.options
    # A sub-task of an unattended (cron) run inherits the no-questions
    # rule: without it a channel agent asked the user for an approval
    # nobody could give and blocked until the run_agent timeout.  It
    # travels in ``add_to_prompt``, which the daemon adds after an
    # agent script's ``prompt(task)`` has produced the prompt body.
    # After ``inherit_from_parent`` so it follows the suffix inherited
    # from the caller (the caller's own copy of this preamble, when
    # the caller is itself such a sub-task, is kept rather than
    # doubled).
    from kiss.agents.sorcar import cron_agent

    if cron_agent.is_unattended(parent_agent):
        options = dataclasses.replace(
            options,
            add_to_prompt=cron_agent.unattended_child_suffix(options.add_to_prompt),
        )
    # The caller's explicit overrides win over the defaults: the
    # calling agent's effective choices for its own run when it
    # carries them, else the user's persisted "Use worktree" / "Auto
    # commit" settings — the same values a task submitted from the
    # chat panel runs with — not a hard-coded ``True`` that would
    # ignore a user who turned them off.  The agent script's
    # ``settings()`` (a ``channel`` preset pins both off) still win on
    # the daemon.
    cfg = load_config()  # fills every key from DEFAULTS
    default_worktree = (
        bool(cfg["is_worktree"])
        if inherited.use_worktree is None else inherited.use_worktree
    )
    default_auto_commit = (
        bool(cfg["auto_commit_mode"])
        if inherited.auto_commit is None else inherited.auto_commit
    )
    use_worktree = (
        default_worktree if options.use_worktree is None else options.use_worktree
    )
    auto_commit = (
        default_auto_commit if options.auto_commit is None else options.auto_commit
    )
    try:
        result = daemon_client.run(
            prompt,
            extension_agent_path=agent_path,
            work_dir=work_dir,
            scope_work_dir=scope_work_dir,
            parent_task_id=parent_task_id,
            parent_tab_id=parent_tab_id,
            parent_reviewer=parent_reviewer or is_review_task(prompt),
            model=model_name,
            chat_id=options.chat_id,
            system_prompt=options.system_prompt,
            use_worktree=use_worktree,
            auto_commit=auto_commit,
            classify_tasks=options.classify_tasks,
            max_budget=budget,
            model_config=options.model_config,
            use_web_tools=options.use_web_tools,
            use_memory=options.use_memory,
            is_parallel=True if options.is_parallel is None else options.is_parallel,
            append_to_system_prompt=options.add_to_system_prompt,
            append_to_prompt=options.add_to_prompt,
            tool_profile=options.tool_profile,
            docker_image=inherited.docker_image,
            workspace=workspace,
            # The caller's extra tools (its script's ``add_to_tools()``)
            # cannot travel the wire: the daemon takes them off the
            # running caller, which ``parent_task_id`` names.
            inherit_tools=inherit,
            timeout=timeout,
            stop_on_timeout=True,
            endpoint_file=_daemon_endpoint_file(),
            cancel=cancel,
            running=running,
        )
    except daemon_client.StopUnconfirmedTimeoutError:
        return stop_unconfirmed_error(name, timeout)
    except daemon_client.CancelledError as e:
        _attribute_dispatch_usage(parent_agent, e.result, epoch)
        if not e.confirmed:
            return (
                f"Error: agent_job kill sent a stop to the {name} agent task but the "
                f"daemon did not confirm it stopped within "
                f"{daemon_client._STOP_CONFIRM_GRACE_SECONDS:g}s; it may still be running."
            )
        return f"Error: the {name} agent task was stopped by agent_job kill."
    except TimeoutError as e:
        # A confirmed stop carries the stopped task's spend, which still
        # counts towards the caller.
        spend = ""
        if isinstance(e, daemon_client.StoppedOnTimeoutError):
            _attribute_dispatch_usage(parent_agent, e.result, epoch)
            spend = f", though its ${e.result.cost:.4f} spend is counted in this task's cost"
        return (
            f"Error: the {name} agent task did not finish within "
            f"{timeout:g}s and was stopped; work it completed before "
            f"the stop (side effects) is not reported here{spend}. "
            f"Check what it already did before retrying with a larger "
            f"`timeout` argument."
        )
    except Exception as e:
        _attribute_dispatch_usage(parent_agent, getattr(e, "task_result", None), epoch)
        logger.warning("agent dispatch failed", exc_info=True)
        return f"Error: the {name} agent task could not run: {e}"
    except BaseException as e:
        # The calling task was stopped (an injected KeyboardInterrupt)
        # while waiting: the sub-task is stopped too, and what it spent
        # so far still counts towards the caller's cost.
        _attribute_dispatch_usage(parent_agent, getattr(e, "task_result", None), epoch)
        raise
    _attribute_dispatch_usage(parent_agent, result, epoch)
    return result


def parse_budget(max_budget: str) -> float | None | str:
    """Parse a tool's ``max_budget`` argument: ``None`` when empty, else a positive finite float.

    Returns:
        The budget, ``None`` for an empty argument, or an error string.
    """
    try:
        budget = float(max_budget) if max_budget.strip() else None
    except ValueError:
        return f"Error: max_budget must be a number, got {max_budget!r}."
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        return f"Error: max_budget must be a positive finite number, got {max_budget!r}."
    return budget


def _run_agent(
    parent_work_dir: str,
    task: str,
    agent: str = "",
    model: str = "",
    tool_profile: str = "",
    max_budget: str = "",
    timeout: str = "",
    options: str = "",
    parent_agent: Any = None,
    wait: str = "",
) -> str:
    """Run an agent script on a task immediately.

    The implementation behind the per-task ``run_agent`` tool built by
    :func:`make_run_agent_tool`, which captures *parent_work_dir*; the
    remaining arguments are the tool's (see its docstring).  One body
    for every agent: resolve the script (:func:`resolve_agent`), read
    its settings, pick the work directory, inherit from the caller
    unless the script's ``inherit`` is ``False``, dispatch.

    Args:
        parent_work_dir: Work directory of the calling task.  A
            relative agent path or ``work_dir`` option is resolved
            against it and a sub-task runs in it by default.  Empty
            (standalone use) resolves relative paths against the
            process working directory and runs sub-tasks in
            ``~/.kiss/agent_work``.
        task: The task for the agent.
        agent: What to run (see :func:`resolve_agent`).
        model: LLM model for the sub-task; empty for the calling
            agent's model (a non-inheriting sub-task: the daemon
            default).
        tool_profile: Tool profile of the sub-task; empty keeps the
            daemon's usual choice.
        max_budget: Per-task USD budget override as a number string;
            empty for half of the calling agent's remaining budget (a
            non-inheriting sub-task: the daemon default).
        timeout: Maximum seconds to wait for the sub-task's result, as
            a number string; empty for the script's ``timeout``
            setting, else :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`.
            On timeout the sub-task is stopped and an error string is
            returned.
        options: JSON object of further run settings (see
            :func:`parse_run_options`).
        parent_agent: The agent calling ``run_agent``, when there is
            one; the sub-task's spend is folded into its task
            accounting (see :func:`_attribute_dispatch_usage`).
        wait: ``"false"`` returns at once with a job id for
            :func:`agent_job`; anything else blocks for the result.

    Returns:
        The sub-task's YAML result ("success" and "summary" keys), the
        job notice (``wait="false"``), or an error message.
    """
    if not task.strip():
        return "Error: task must be a non-empty string."
    budget = parse_budget(max_budget)
    if isinstance(budget, str):
        return budget
    try:
        run_options = parse_run_options(options, tool_profile)
    except ValueError as e:
        return f"Error: {e}"
    try:
        blocking = _parse_bool("wait", wait) is not False
    except ValueError as e:
        return f"Error: {e}"
    resolved = resolve_agent(agent, parent_work_dir)
    if isinstance(resolved, str):
        return resolved
    agent_path, name = resolved
    from kiss.agents.sorcar.sea_commands import SeaScriptError, sea_settings

    try:
        settings = sea_settings(Path(agent_path))
    except SeaScriptError as e:
        return f"Error: {e}"
    seconds = resolve_timeout(timeout, settings)
    if isinstance(seconds, str):
        return seconds
    # One rule for every script: its ``settings()`` decide.  A script
    # with ``inherit: False`` (the ``channel`` preset: a channel agent,
    # cron) takes nothing from the calling task — not its chat, model,
    # budget share, container or prompt suffixes — and runs in its own
    # ``work_dir`` (the preset's scratch directory).  Every other
    # script is a sub-agent on the caller's project (or on the
    # ``work_dir`` option, resolved against it): the arguments left
    # empty are inherited from the calling agent (see
    # ``inherit_from_parent``), and the task's references to the main
    # checkout are rewritten to the caller's worktree.
    inherit = bool(settings.get(
        "inherit", True if run_options.inherit is None else run_options.inherit,
    ))
    work_dir = parent_work_dir or str(kiss_home() / "agent_work")
    if run_options.work_dir:
        requested = Path(run_options.work_dir).expanduser()
        work_dir = str(requested if requested.is_absolute() else Path(work_dir) / requested)
    work_dir = str(settings.get("work_dir") or "") or work_dir
    if inherit and DEFAULT_CONFIG.dispatch_path_rewrite:
        task = rewrite_parent_repo_paths(task, parent_work_dir)
    kwargs: dict[str, Any] = {
        "name": name, "prompt": task, "agent_path": agent_path, "work_dir": work_dir,
        "model_name": model, "budget": budget, "timeout": seconds,
        "parent_agent": parent_agent, "scope_work_dir": parent_work_dir,
        "options": run_options, "inherit": inherit, "settings": settings,
        "workspace": run_options.workspace,
    }
    if blocking:
        return _dispatch(**kwargs)
    return start_agent_job(name, kwargs, parent_agent)


@dataclass
class AgentJob:
    """A ``run_agent(wait="false")`` sub-task running in the background.

    Attributes:
        job_id: The id ``agent_job`` looks the job up by.
        name: The agent's display name.
        owner: The agent whose ``run_agent`` started the job; only that
            agent's ``agent_job`` tool sees it.
        cancel: Set by ``agent_job(..., "kill")``; the dispatch's read
            loop stops the sub-task when it sees it.
        running: Set once the sub-task's tab exists on every client
            (its initial ``status running=true``) or the dispatch ended
            without one.
        thread: The thread running :func:`_dispatch`, already started.
        result: The tool text the dispatch returned; ``""`` while running.
    """

    job_id: str
    name: str
    owner: Any
    cancel: threading.Event
    running: threading.Event
    thread: threading.Thread
    result: str = ""


_AGENT_JOBS: dict[str, AgentJob] = {}
"""Every background sub-task started in this process, by job id.

Guarded by :data:`_AGENT_JOBS_LOCK`.
"""

_AGENT_JOBS_LOCK = threading.Lock()


def start_agent_job(name: str, kwargs: dict[str, Any], owner: Any) -> str:
    """Start :func:`_dispatch` with *kwargs* in a thread and return the job notice.

    The job is published only after its thread has started, so every
    ``agent_job`` action finds a joinable thread, and the notice is
    returned only once the sub-task's tab is open (bounded by
    :data:`_JOB_START_GRACE_SECONDS`): the spawn then falls inside the
    ``run_agent`` call's time window, which is how every surface files
    a sub-agent tab under the call that started it.

    Args:
        name: The agent's display name.
        kwargs: The keyword arguments of :func:`_dispatch`; the job's
            ``cancel`` event is added.
        owner: The calling agent (``None`` for standalone use).

    Returns:
        The text the ``run_agent`` tool returns: the job id and how to
        use ``agent_job`` on it.
    """
    job_id = f"agent-{uuid.uuid4().hex[:8]}"
    cancel, running = threading.Event(), threading.Event()
    job = AgentJob(job_id, name, owner, cancel, running, threading.Thread())
    job.thread = threading.Thread(
        target=_finish_agent_job,
        args=(job, {**kwargs, "cancel": cancel, "running": running}),
        name=f"agent-job-{job_id}", daemon=True,
    )
    job.thread.start()
    with _AGENT_JOBS_LOCK:
        _AGENT_JOBS[job_id] = job
    running.wait(_JOB_START_GRACE_SECONDS)
    return (
        f"Started the {name} agent task as job {job_id}; its tab is open. "
        f"agent_job({job_id!r}, 'wait') blocks until it finishes and returns its "
        f"result, 'tail' reports its status, 'kill' stops it. Wait for or kill "
        f"it before finishing."
    )


def _finish_agent_job(job: AgentJob, kwargs: dict[str, Any]) -> None:
    """Thread body of a background job: record the dispatch's text on *job*."""
    try:
        job.result = _dispatch(**kwargs)
    except BaseException as exc:  # noqa: BLE001 — the thread must record any failure
        logger.warning("agent job %s failed", job.job_id, exc_info=True)
        job.result = f"Error: the {job.name} agent task could not run: {safe_message(exc)}"
    finally:
        job.running.set()


def agent_jobs_of(owner: Any) -> dict[str, AgentJob]:
    """Return the background jobs *owner*'s ``run_agent`` started, by job id."""
    with _AGENT_JOBS_LOCK:
        return {job_id: job for job_id, job in _AGENT_JOBS.items() if job.owner is owner}


def make_agent_job_tool(owner: Any = None) -> Callable[..., str]:
    """Build the ``agent_job`` tool of the agent *owner*.

    Each agent sees only the jobs its own ``run_agent`` started: a job
    id is not a capability another chat may wait on or kill.

    Args:
        owner: The agent the tool is built for (``None`` for standalone
            use, which sees the jobs started without an owner).

    Returns:
        The ``agent_job(job_id, action, timeout_seconds)`` tool.
    """

    def agent_job(job_id: str, action: str = "tail", timeout_seconds: str = "") -> str:
        """Wait for, check or kill a sub-task started by ``run_agent(..., wait="false")``.

        Args:
            job_id: The id ``run_agent`` returned (``agent-1a2b3c4d``).
            action: ``"tail"`` (default) reports whether the task is
                still running and, once it has finished, its result;
                ``"wait"`` blocks until it finishes (at most
                ``timeout_seconds``) and returns its result; ``"kill"``
                stops it.
            timeout_seconds: How long ``"wait"`` blocks at most, as a
                number string; empty waits for the task's own
                ``timeout`` to end it.

        Returns:
            The job's status, or the sub-task's YAML result ("success"
            and "summary" keys) once it has finished, or an error
            message.
        """
        jobs = agent_jobs_of(owner)
        job = jobs.get(job_id.strip())
        if job is None:
            known = ", ".join(sorted(jobs)) or "none"
            return f"Error: unknown agent job {job_id!r}; this task's jobs: {known}."
        if action == "wait":
            seconds = parse_wait_seconds(timeout_seconds)
            if isinstance(seconds, str):
                return seconds
            job.thread.join(seconds)
        elif action == "kill":
            job.cancel.set()
            job.thread.join(_JOB_KILL_GRACE_SECONDS)
        elif action != "tail":
            return f"Error: action must be tail, wait or kill, got {action!r}."
        if job.thread.is_alive():
            return f"Job {job_id} ({job.name} agent task) is still running."
        return job.result

    return agent_job


def parse_wait_seconds(timeout_seconds: str) -> float | None | str:
    """Parse ``agent_job``'s ``timeout_seconds``.

    ``None`` when empty, else a finite non-negative float.

    Returns:
        The seconds, ``None`` for an empty argument, or an error string.
    """
    if not timeout_seconds.strip():
        return None
    try:
        seconds = float(timeout_seconds)
    except ValueError:
        return f"Error: timeout_seconds must be a number, got {timeout_seconds!r}."
    if not math.isfinite(seconds) or seconds < 0:
        return (
            "Error: timeout_seconds must be a finite non-negative number, "
            f"got {timeout_seconds!r}."
        )
    return seconds


_JOB_KILL_GRACE_SECONDS = 30.0
"""How long ``agent_job(..., "kill")`` waits for the stopped sub-task's dispatch to return."""

_JOB_START_GRACE_SECONDS = 30.0
"""How long ``run_agent(wait="false")`` waits for the sub-task's tab before returning its notice."""


_GENERIC_AGENT_NAMES = frozenset({
    "general", "agent", "sorcar", "kiss", "codereview", "codereviewer",
    "reviewer", "review", "analysis", "analyst", "assistant", "default",
    "subagent", "worker", "helper", "llm", "model",
})
"""Names models invent for "another copy of me" (16 dispatches in the
7-day audit of 2026-09-19).  Each means what an empty ``agent`` means: a
plain Sorcar sub-agent (:data:`DEFAULT_AGENT_PATH`) on the task."""


def resolve_agent(agent: str, parent_work_dir: str) -> tuple[str, str] | str:
    """Resolve the ``run_agent`` tool's ``agent`` argument to an agent-script path.

    Three rules, in order, for every spelling a model may use:

    1. empty, or a generic label such as ``"general"`` / ``"reviewer"``
       (:data:`_GENERIC_AGENT_NAMES`): the plain sub-agent script
       :data:`DEFAULT_AGENT_PATH`;
    2. a path (ends in ``.py`` or contains a separator): that agent
       script, a relative path resolved against *parent_work_dir*;
    3. a registered slash-command name (case, spaces, hyphens and
       underscores ignored: "Home Assistant" is ``homeassistant``) —
       a bundled SEA (``write_paper``), a channel (``slack``),
       ``cron``, or a ``SEAS.md`` folder: that command's script.

    Args:
        agent: The argument as the model passed it.
        parent_work_dir: Work directory relative paths resolve against;
            empty resolves against the process working directory.

    Returns:
        ``(absolute_script_path, name)``, *name* being what the
        sub-task is reported as (the command name, else the script's
        :func:`~kiss.agents.sorcar.sea_settings.script_name`); an error
        string naming the closest command when nothing matches or the
        path is not a readable ``.py`` file.
    """
    from kiss.agents.sorcar import sea_commands
    from kiss.agents.sorcar.daemon_client import resolve_agent_path

    requested = agent.strip()
    squashed = _squash(requested)
    if not requested or squashed in _GENERIC_AGENT_NAMES:
        requested = DEFAULT_AGENT_PATH
    if requested.endswith(".py") or "/" in requested or "\\" in requested:
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute() and parent_work_dir:
            candidate = Path(parent_work_dir) / candidate
        try:
            agent_path = resolve_agent_path(str(candidate))
        except ValueError as e:
            return f"Error: {e}"
        return agent_path, script_name(agent_path)
    commands = sea_commands.list_commands()
    for name in commands:
        if _squash(name) == squashed:
            sea_path = sea_commands.get_command(name)
            if sea_path is not None:
                return str(sea_path), name
    return _unknown_agent_error(agent, squashed, commands)


def resolve_timeout(timeout: str, settings: dict[str, Any]) -> float | str:
    """Resolve the ``run_agent`` tool's ``timeout`` argument to seconds.

    An explicit positive number wins; an empty argument takes the agent
    script's own ``timeout`` setting (``settings()["timeout"]``, see
    :mod:`kiss.agents.sorcar.sea_settings`), else
    :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`.

    Args:
        timeout: The argument as the model passed it.
        settings: The resolved settings of the script the sub-task
            runs.

    Returns:
        The seconds to wait, or an error string for a malformed
        argument.
    """
    if timeout.strip():
        try:
            wait = float(timeout)
        except ValueError:
            return f"Error: timeout must be a number of seconds, got {timeout!r}."
        if not math.isfinite(wait) or wait <= 0:
            return (
                f"Error: timeout must be a positive finite number of seconds, "
                f"got {timeout!r}."
            )
        return wait
    declared = settings.get("timeout")
    if isinstance(declared, int | float) and declared > 0:
        return float(declared)
    return DEFAULT_DISPATCH_TIMEOUT_SECONDS


def _unknown_agent_error(agent: str, squashed: str, commands: list[str]) -> str:
    """Build the ``run_agent`` error for a name that matches nothing.

    Args:
        agent: The name as the model passed it.
        squashed: Its case/space/hyphen-insensitive form.
        commands: Registered slash-command names.

    Returns:
        An error string naming the closest known command when there is
        one, and listing what ``run_agent`` accepts.
    """
    by_squashed = {_squash(name): name for name in commands}
    close = difflib.get_close_matches(squashed, list(by_squashed), n=1, cutoff=0.6)
    hint = f" Did you mean {by_squashed[close[0]]!r}?" if close else ""
    return (
        f"Error: unknown agent {agent!r} — not a registered slash command "
        f"and not a path to a .py agent script.{hint} Leave agent empty for "
        f"a plain sub-agent. Commands: {', '.join(commands) or 'none registered'}."
    )


def make_run_agent_tool(
    work_dir: str, parent_agent: Any = None,
) -> Callable[..., str]:
    """Build the ``run_agent`` tool for the agent running in *work_dir*.

    The tool executes in the daemon process, whose own working
    directory is unrelated to the user's project, so the calling
    task's work directory must be captured here (exactly like
    ``make_skill_tool``): it anchors relative agent-script paths and
    is the work directory the dispatched path-mode sub-task runs in.

    Args:
        work_dir: The calling task's work directory.  Empty applies
            the standalone defaults (see :func:`_run_agent`).
        parent_agent: The agent this tool is built for, when there is
            one.  Each dispatched sub-task's cost/tokens/steps are
            folded into its task accounting so the calling task's
            end-of-task cost includes ``run_agent`` spend.  ``None``
            (standalone use) disables attribution.

    Returns:
        The ``run_agent`` tool callable.
    """

    def run_agent(
        task: str,
        agent: str = "",
        model: str = "",
        tool_profile: str = "",
        max_budget: str = "",
        timeout: str = "",
        options: str = "",
        wait: str = "",
    ) -> str:
        """Run an agent (channel, slash-command SEA, cron or any agent script) on a task now.

        Call it RIGHT AWAY, without exploring any source code, when the
        task is to act on an external messaging service, mailbox or
        device channel (pass the request through as the task; the
        channel agent has its own authenticated tools), to manage
        scheduled automations (``agent="cron"``; a messaging gateway is
        a cron task too, see SYSTEM.md), or whenever the user names an
        agent file or a slash command to run a task with.

        Available channels: {channels}.  The built-in ``"cron"`` agent
        is always available.  The agent file's ``settings()`` win over
        the arguments here, which win over what this task passes on
        (its model, half of its remaining budget, chat, prompt
        suffixes, extra tools, container, worktree/auto-commit and
        fan-out choices; a channel or cron sub-task inherits none of
        these and runs in ``~/.kiss/channel_work``).  The call blocks
        until the task finishes or ``timeout`` expires; a timed-out
        task is stopped (its side effects are not reported; its spend
        still counts here).

        Args:
            task: The task for the agent, e.g. "Send 'hello' to the
                #sorcar channel".  An agent file's ``prompt(task)``,
                if defined, turns it into the prompt.
            agent: Empty (default) runs a plain Sorcar sub-agent.
                Otherwise: the path of a ``.py`` agent-script file
                (relative to this task's work directory) or a
                slash-command name: ``"cron"``, a channel (``"slack"``,
                ``"gmail"``, "Home Assistant" ...) or a bundled SEA
                (``"write_paper"``; what ``/write_paper ...`` runs).
                Generic labels (``"general"``, ``"reviewer"``,
                ``"worker"``) mean the plain sub-agent.
            model: LLM model for the sub-task; empty uses this task's
                model (the daemon default for a channel/cron sub-task).
            tool_profile: ``"review"`` gives the sub-task the read-only
                toolset; ``"shell"``, ``"assistant"``, ``"bash"``,
                ``"none"`` or groups joined with ``+``
                (``"shell+edit+browser"``) as for ``run_parallel``.
                Empty keeps the full toolset.
            max_budget: Per-task USD budget as a number string; empty
                gives the sub-task half of this task's remaining budget
                (the daemon default for a channel/cron sub-task).
            timeout: Maximum seconds to wait, as a number string;
                empty uses the agent file's ``timeout`` setting, else
                3600.  Check what a timed-out task already did before
                retrying with a larger value.
            options: Optional JSON object of run settings, each only
                when you need to override the inherited value:
                ``work_dir`` (run the sub-task in another directory,
                relative to this task's), ``chat_id``, ``workspace``
                (the account of a multi-account channel; default
                ``"default"``), ``add_to_system_prompt`` /
                ``add_to_prompt`` (appended text), ``model_config``
                (JSON object), ``docker_image``, and the booleans
                ``inherit`` (``false``: take nothing from this task),
                ``use_worktree``, ``auto_commit``, ``classify_tasks``,
                ``use_web_tools``, ``use_memory``, ``is_parallel``.
                Example: ``'{"use_web_tools": false}'``.  Usually leave
                it empty.
            wait: ``"false"`` returns at once with a job id instead of
                blocking; then ``agent_job(job_id, "wait")`` returns
                the result, ``"tail"`` its status, ``"kill"`` stops it.
                Use it to run several agents at once or to keep
                working while a long sub-task runs; wait for or kill
                every job before finishing.  Empty (default) blocks.

        Returns:
            The sub-task's YAML result ("success" and "summary" keys),
            the job notice (``wait="false"``), or an error message
            (unknown agent — naming the closest command — or a
            timeout).
        """
        return _run_agent(
            work_dir, task, agent, model, tool_profile, max_budget,
            timeout, options, parent_agent, wait,
        )

    run_agent.__doc__ = (run_agent.__doc__ or "").replace(
        "{channels}", ", ".join(available_channels()) or "none installed"
    )
    return run_agent
