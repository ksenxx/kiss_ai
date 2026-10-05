# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``run_agent`` and ``agent_job`` tools: run any SEA on a task, now.

:func:`make_run_agent_tool` builds ``run_agent`` per task.  Its
``agent`` argument names what to run (:func:`resolve_agent`): empty
for a plain sub-agent, a ``.py`` path, or the name of a registered
slash command (:func:`kiss.agents.sorcar.sea_commands.list_commands`:
the bundled SEAs such as ``write_paper``, the channel SEAs such as
``slack`` and ``gmail``, ``cron``, and the user's own ``SEAS.md``
folders).  Every spelling resolves to one SEA path and takes
one dispatch path (:func:`_run_agent`):

1. the script's effective ``settings()`` are read in the calling
   process (:func:`kiss.agents.sorcar.sea_commands.sea_settings`) for
   the ``timeout``, ``kind`` and ``work_dir`` the dispatcher needs;
2. unless the kind is ``channel`` or the ``inherit`` option is
   ``false``, the arguments the call left empty are inherited from the
   calling agent (:func:`inherit_from_parent`:
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
:mod:`kiss.agents.sorcar.agent_file`) and, for ``kind: "channel"``, holds the
channel workspace (the ``workspace`` option, forwarded as a wire
field) for the run's lifetime.

One precedence rule holds for every setting of the sub-task,
:data:`kiss.agents.sorcar.sea_settings.PRECEDENCE_RULE`: the tool's
explicit arguments and options win, then the SEA's ``settings()``,
then what the calling task passes on, then the user's persisted
settings — except that an argument differing from a setting the SEA
lists in ``locked`` is an error.  Inside the kiss-web
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
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.daemon_client import TaskResult
from kiss.agents.sorcar.run_config import (
    PROVENANCE_EXPLICIT,
    PROVENANCE_INHERITED,
    run_config_line,
)
from kiss.agents.sorcar.sea_commands import sea_script_in
from kiss.agents.sorcar.sea_settings import (
    META_SETTINGS,
    PRECEDENCE_RULE,
    REMOVED_SETTINGS,
    RENAMED_SETTINGS,
    SETTING_TYPES,
    declared_literal,
    declares_hidden,
    locked_conflicts,
    safe_message,
    script_name,
)
from kiss.agents.sorcar.useful_tools import (
    remap_vanished_worktree,
    rewrite_parent_repo_paths,
)
from kiss.core.config import DEFAULT_CONFIG, kiss_home
from kiss.core.vscode_config import load_config

logger = logging.getLogger(__name__)

DEFAULT_AGENT_PATH = str(
    Path(__file__).resolve().parents[1] / "seas" / "sorcar" / "sorcar_sea.py"
)
"""Agent script run when the ``run_agent`` tool's ``agent`` is empty.

The bundled ``src/kiss/agents/seas/sorcar/sorcar_sea.py`` — a hidden SEA that
pins no setting, so the sub-task is a plain Sorcar session on the given task
in the calling task's work directory (path mode, with the standard
worktree/auto-commit lifecycle).  Held as the absolute path of the
installed file so the default works from any work directory, not only
a checkout of this repository.
"""

DEFAULT_DISPATCH_TIMEOUT_SECONDS = 3600.0
"""Default bound on how long a ``run_agent`` call blocks for its sub-task.

Used when the ``run_agent`` tool's ``timeout`` argument is empty and
the SEA's ``settings()`` declares no ``timeout`` (see
:func:`resolve_timeout`); a per-call value overrides both.  One hour
rather than minutes: a sub-task that writes a paper or runs a test
suite legitimately takes that long.  The bound is on the CALL, not on
the sub-task: every ``run_agent`` dispatch runs as an :class:`AgentJob`
on its own thread, the call joins that thread for the bound, and when
the bound expires the call returns the job's still-running notice (see
:func:`job_notice`) while the sub-task keeps running.  The caller then
collects or stops it with ``agent_job(id, "wait" | "kill")``; a job it
never resolves is killed when the calling run ends
(:func:`kill_jobs_of`), so no sub-task outlives its caller.
"""


@dataclass(frozen=True)
class RunOptions:
    """Optional per-run overrides of a dispatched sub-task.

    The parsed form of the ``run_agent`` / ``run_parallel`` ``options``
    argument: one field per key of :data:`OPTION_TYPES` — the SEA
    settings vocabulary (:data:`~kiss.agents.sorcar.sea_settings.SETTING_TYPES`)
    minus the keys that describe a script — plus ``system_prompt``, the
    replacement base system prompt a programmatic caller may pass (it
    is not an ``options`` key: a SEA's ``system_prompt()`` is the
    user-facing way).  The tool's ``model``, ``tool_profile``,
    ``max_budget`` and ``timeout`` arguments are shortcuts for the
    options of the same name (:func:`parse_run_options` merges them).
    ``None`` / empty means "not passed".  A value here is explicit, so
    it ranks first in :data:`~kiss.agents.sorcar.sea_settings.PRECEDENCE_RULE`:
    it wins over the SEA's ``settings()`` unless the SEA locks the key
    (then the call is refused), and over what the calling agent passes
    on (see :func:`inherit_from_parent`).
    """

    model: str = ""
    max_budget: float | None = None
    timeout: float | None = None
    work_dir: str = ""
    chat_id: str = ""
    use_worktree: bool | None = None
    auto_commit: bool | None = None
    model_config: dict[str, Any] | None = None
    use_web_tools: bool | None = None
    auto_classify: bool | None = None
    use_memory: bool | None = None
    allow_fan_out: bool | None = None
    tool_profile: str = ""
    docker_image: str = ""
    inherit: bool | None = None
    workspace: str = ""
    add_to_prompt: str = ""
    add_to_system_prompt: str = ""
    system_prompt: str = ""


OPTION_TYPES: dict[str, type | tuple[type, ...]] = {
    **{key: expected for key, expected in SETTING_TYPES.items() if key not in META_SETTINGS},
    "inherit": bool,
    "workspace": str,
    "add_to_prompt": str,
    "add_to_system_prompt": str,
}
"""The keys the ``options`` JSON object accepts, with the type of each value.

The SEA settings vocabulary
(:data:`~kiss.agents.sorcar.sea_settings.SETTING_TYPES`) minus the
keys that describe a script rather than a run
(:data:`~kiss.agents.sorcar.sea_settings.META_SETTINGS`: ``kind``,
``extends``, ``locked``, ``hidden``), so the tool and the SEAs share
one vocabulary: what a SEA may pin in ``settings()``, a caller may
pass in ``options``.  The tool's ``model``, ``tool_profile``,
``max_budget`` and ``timeout`` arguments are shortcuts for the options
of the same name (both may be given when they agree).  Plus four
call-only keys: ``inherit`` (``false``: the sub-task takes nothing
from the calling task), ``workspace`` (the account a channel agent's
run holds), ``add_to_prompt`` (text appended to the task) and
``add_to_system_prompt``, the option form of a SEA's
``add_to_system_prompt()`` getter.
"""

ARGUMENT_OPTIONS = ("model", "tool_profile", "max_budget", "timeout")
"""The options the ``run_agent`` / ``run_parallel`` tools also take as arguments."""

OPTION_DOCS: dict[str, str] = {
    "work_dir": "The directory the sub-task works in; a relative path is resolved against "
                "the calling task's directory (a SEA's own `work_dir` setting is relative "
                "to the SEA's folder instead).",
    "inherit": "`false`: the sub-task takes nothing from the calling task (no model, chat, "
               "prompt suffixes, tools or container; a `run_parallel` child still gets its "
               "budget share); default `true`. A `channel` run never inherits, so `true` is "
               "refused there.",
    "workspace": "The account a `kind: channel` agent's run holds (its channel workspace); "
                 "refused for any other kind and by `run_parallel`.",
    "add_to_prompt": "Text appended to the task after the SEA's `prompt(task)`.",
    "add_to_system_prompt": "Text appended to the system prompt after the SEA's "
                            "`add_to_system_prompt()`.",
}
"""Documentation of the option keys that are not ``settings()`` keys, or mean something
else as an option (``work_dir``: relative to the caller, not the script), for ``sea docs``.

Every other option is documented by
:data:`~kiss.agents.sorcar.sea_settings.SETTING_DOCS`.
"""

FANOUT_REFUSED: dict[str, Any] = {"use_worktree": True, "auto_commit": True, "auto_classify": True}
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


def _parse_number(name: str, value: Any) -> float | None:
    """Parse an optional positive finite number (``max_budget``, ``timeout``).

    Args:
        name: The option's name, for the error message.
        value: A JSON number, a numeric string (surrounding whitespace
            ignored) or empty / ``None`` for "not passed".

    Returns:
        The number, or ``None`` when *value* is empty.

    Raises:
        ValueError: When *value* is not a positive finite number.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value) if not isinstance(value, bool) else "no"
    except (TypeError, ValueError, OverflowError):
        number = "no"
    if isinstance(number, str):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return number


def parse_run_options(
    options: str,
    tool_profile: str = "",
    model: str = "",
    max_budget: str = "",
    timeout: str = "",
) -> RunOptions:
    """Parse the ``options`` argument of ``run_agent`` / ``run_parallel`` and its shortcuts.

    Args:
        options: A JSON object string in the settings vocabulary
            (:data:`OPTION_TYPES`), or empty for no overrides.
            Booleans may also be given as the strings ``"true"`` /
            ``"false"``; ``null`` means "not passed".
        tool_profile: The tool's ``tool_profile`` argument: a shortcut
            for the option of the same name, which may repeat but not
            contradict it.  Canonicalised by
            :func:`kiss.agents.sorcar.sorcar_agent.canonical_tool_profile`
            (``readonly`` becomes ``review``).
        model: The tool's ``model`` argument, the same way.
        max_budget: The tool's ``max_budget`` argument (a positive
            finite number as text), the same way.
        timeout: The tool's ``timeout`` argument (seconds as text), the
            same way.

    Returns:
        The parsed options, the shortcuts merged in.

    Raises:
        ValueError: When *options* is not a JSON object, names an
            unknown key, has a value of the wrong type, an option
            contradicts the argument of the same name, a number is not
            positive and finite, or the tool profile is unknown.
    """
    from kiss.agents.sorcar.sorcar_agent import canonical_tool_profile

    parsed: dict[str, Any] = {}
    if options.strip():
        try:
            raw = json.loads(options)
        except ValueError as e:
            raise ValueError(f"options must be a JSON object, got {options!r}: {e}") from None
        if not isinstance(raw, dict):
            raise ValueError(f"options must be a JSON object, got {options!r}.")
        for key, value in raw.items():
            parsed_value = _parse_option(key, value)
            if parsed_value is not None:
                parsed[key] = parsed_value
    if parsed.get("tool_profile"):
        parsed["tool_profile"] = canonical_tool_profile(parsed["tool_profile"])
    arguments = {
        "model": model.strip(),
        "tool_profile": canonical_tool_profile(tool_profile),
        "max_budget": _parse_number("max_budget", max_budget),
        "timeout": _parse_number("timeout", timeout),
    }
    for key, argument in arguments.items():
        if argument in (None, ""):
            continue
        option = parsed.get(key)
        if option is not None and option != argument:
            raise ValueError(
                f"options[{key!r}] = {option!r} contradicts the {key} argument "
                f"{argument!r}; pass one of them."
            )
        parsed[key] = argument
    return RunOptions(**parsed)


def options_keyword_hint(unknown: dict[str, Any]) -> str:
    """Explain the keywords a ``run_agent`` / ``run_parallel`` call passed that are no argument.

    Installed on the two tools as their ``unknown_arguments_hint``
    (:func:`kiss.core.kiss_agent.unknown_argument_hint`): a run
    setting passed as a keyword (``use_worktree=False``) is shown as
    the ``options`` object to pass instead, a renamed or removed key
    gets its current name or the reason, and anything else the
    closest option key when one is spelled closely enough.

    Args:
        unknown: ``{keyword: value}`` for every keyword the signature
            does not take.

    Returns:
        One sentence per keyword, newline-joined.
    """
    lines = []
    settings = {k: v for k, v in unknown.items() if k in OPTION_TYPES}
    if settings:
        example = json.dumps(settings)
        lines.append(
            f"{', '.join(settings)} is a run setting, not an argument; pass it in the "
            f"`options` JSON object: options='{example}'."
        )
    for key, value in unknown.items():
        if key in settings:
            continue
        if key in RENAMED_SETTINGS:
            example = json.dumps({RENAMED_SETTINGS[key]: value})
            lines.append(
                f"{key} was renamed to {RENAMED_SETTINGS[key]}; pass options='{example}'."
            )
        elif key in REMOVED_SETTINGS:
            lines.append(f"{key} was removed: {REMOVED_SETTINGS[key]}.")
        else:
            close = difflib.get_close_matches(key, list(OPTION_TYPES), n=1, cutoff=0.7)
            hint = f" Did you mean the option {close[0]!r}?" if close else ""
            lines.append(f"{key} is neither an argument nor an options key.{hint}")
    return "\n".join(lines)


def _parse_option(key: str, value: Any) -> Any:
    """Parse one ``options`` entry; ``None`` (for ``null`` and a blank string too) is "not passed".

    Raises:
        ValueError: When *key* is unknown (naming the current key for a
            renamed setting) or *value* has the wrong type.
    """
    expected = OPTION_TYPES.get(key)
    if expected is None:
        if key in RENAMED_SETTINGS:
            raise ValueError(
                f"options key {key!r} was renamed to {RENAMED_SETTINGS[key]!r}; "
                f"use the new name."
            )
        if key in REMOVED_SETTINGS:
            raise ValueError(f"options key {key!r} was removed: {REMOVED_SETTINGS[key]}.")
        close = difflib.get_close_matches(key, list(OPTION_TYPES), n=1, cutoff=0.7)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"options has an unknown key {key!r}; known keys: {', '.join(OPTION_TYPES)}.{hint}"
        )
    if value is None:
        return None
    if expected is bool:
        return _parse_bool(key, value)
    if key in ("max_budget", "timeout"):
        return _parse_number(f"options[{key!r}]", value)
    if isinstance(value, str) and expected is str:
        if not value.strip():
            return None
        if key in ("chat_id", "work_dir", "workspace", "model", "tool_profile"):
            return value.strip()
        return value
    if isinstance(value, expected) and not isinstance(value, bool):
        return value
    raise ValueError(
        f"options[{key!r}] must be a JSON {getattr(expected, '__name__', 'number')}, "
        f"got {type(value).__name__}."
    )


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

    A channel is a SEA folder ``<channel>/<channel>_sea.py`` of the
    third-party agents package whose ``settings()`` write ``"kind":
    "channel"`` and not ``"hidden": True``.  The package's other SEAs
    (``a2a``, a session SEA whose tools call peer agents; ``oai``, a
    hidden server set up from a terminal) are not channels.  The scan
    reads the directory listing and parses the two literals from
    source — no channel module is imported.

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
        and sea_script_in(sea_dir).is_file()
        and declared_literal(sea_script_in(sea_dir), "kind") == "channel"
        and not declares_hidden(sea_script_in(sea_dir))
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
    fields: tuple[str, ...] = ()
    """The setting keys that were filled from the caller (``model``, ``chat_id``, ...)."""

    def provenance(self, explicit: Iterable[str]) -> dict[str, str]:
        """Return the ``provenance`` wire field of a run: ``{setting key: explicit | inherited}``.

        Args:
            explicit: The setting keys the call passed itself
                (:func:`explicit_keys`).
        """
        marks = dict.fromkeys(self.fields, PROVENANCE_INHERITED)
        marks.update(dict.fromkeys(explicit, PROVENANCE_EXPLICIT))
        return marks


def explicit_values(
    model_name: str, budget: float | None, options: RunOptions,
) -> dict[str, Any]:
    """Return the settings a ``run_agent`` / ``run_parallel`` call passed explicitly.

    Args:
        model_name: The call's ``model`` argument (empty: not passed).
        budget: The call's parsed ``max_budget`` (``None``: not passed).
        options: The call's parsed options; a field that is ``None`` or
            empty was not passed.

    Returns:
        ``{setting key: value}`` for every value passed.
    """
    values: dict[str, Any] = {
        key: getattr(options, key) for key in OPTION_TYPES
        if getattr(options, key, None) not in (None, "")
    }
    if model_name:
        values["model"] = model_name
    if budget is not None:
        values["max_budget"] = budget
    return values


def explicit_keys(model_name: str, budget: float | None, options: RunOptions) -> list[str]:
    """Return the setting keys a ``run_agent`` / ``run_parallel`` call passed explicitly.

    See :func:`explicit_values`.
    """
    return list(explicit_values(model_name, budget, options))


def _filled_keys(asked: Inherited, got: Inherited) -> tuple[str, ...]:
    """Return the setting keys whose value *got* has and *asked* left empty."""
    pairs = [
        ("model", asked.model_name, got.model_name),
        ("max_budget", asked.budget, got.budget),
        ("docker_image", asked.docker_image, got.docker_image),
        ("use_worktree", asked.use_worktree, got.use_worktree),
        ("auto_commit", asked.auto_commit, got.auto_commit),
    ]
    pairs.extend(
        (key, getattr(asked.options, key), getattr(got.options, key))
        for key in OPTION_TYPES if key != "docker_image"
    )
    pairs.append(("system_prompt", asked.options.system_prompt, got.options.system_prompt))
    return tuple(
        key for key, before, after in pairs
        if before in (None, "") and after not in (None, "")
    )


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
    model) AND the SEA does not pick the model itself: the
    daemon applies a script's ``model`` setting on top of the wire
    fields without touching ``modelConfig`` (``apply_agent_overrides``),
    so a script-chosen model would otherwise run against the caller's
    endpoint.  An empty configuration is reported as ``None``.

    Args:
        parent_agent: The agent calling ``run_agent``.
        model_name: The model the sub-task will run unless the script
            overrides it.
        script_picks_model: Whether the SEA's ``settings()``
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
      the model the caller was launched with and the SEA
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
      exactly as through ``run_parallel``.  A SEA's
      ``system_prompt()`` still replaces the base prompt on the
      daemon, and its ``add_to_system_prompt()`` text is added after
      the inherited suffix.
    - ``add_to_prompt``: the suffix the caller's own task prompt
      was given (``_prompt_suffix``, the ``appendToPrompt`` of its
      run), so the sub-task's prompt ends with the same text.  An
      SEA's ``prompt(task)`` getter then rewrites the whole
      task text on the daemon.
    - ``allow_fan_out``: whether the caller may fan out itself
      (``_is_parallel``), so a sequential caller (a ``worker`` kind,
      a user who turned fan-out off) does not hand ``run_parallel``
      back to its children.
    - the caller's extra tools (its SEA's ``add_to_tools()``
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
        script_picks_model: Whether the sub-task's SEA names
            a ``model`` in its ``settings()``, which blocks the
            ``model_config`` inheritance.

    Returns:
        The resolved values.

    Raises:
        BudgetExceededError: When the budget is inherited and the
            caller has nothing left to spend (the same signal a
            ``run_parallel`` fan-out raises).
    """
    asked = Inherited(model_name, budget, options, options.docker_image, None, None)
    if parent_agent is None:
        return asked
    got = _inherit_from_parent(parent_agent, model_name, budget, options, script_picks_model)
    return dataclasses.replace(got, fields=_filled_keys(asked, got))


def _inherit_from_parent(
    parent_agent: Any,
    model_name: str,
    budget: float | None,
    options: RunOptions,
    script_picks_model: bool,
) -> Inherited:
    """:func:`inherit_from_parent` for a caller that exists, without the ``fields`` record."""
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
        allow_fan_out=(
            getattr(parent_agent, "_is_parallel", None)
            if options.allow_fan_out is None else options.allow_fan_out
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


def unconfirmed_stop_error(name: str) -> str:
    """Return the error a killed job records when the daemon never confirmed the stop.

    The exact text (never a substring, which unrelated text such as a
    path in a connection error could contain) lets a programmatic
    caller that killed the job (``cron_agent._run_prompt_job``) tell a
    sub-task that may still be running from one that is dead, and keep
    the run's scratch directory for it.

    Args:
        name: The dispatched agent's display name.

    Returns:
        The complete error string.
    """
    from kiss.agents.sorcar import daemon_client

    return (
        f"Error: a stop was sent to the {name} agent task but the daemon did not "
        f"confirm it stopped within {daemon_client._STOP_CONFIRM_GRACE_SECONDS:g}s; "
        f"it MAY STILL BE RUNNING (and spending) on the daemon."
    )


def notice_job_id(text: str) -> str:
    """Return the job id a :func:`job_notice` names, or ``""`` for any other tool text."""
    match = re.match(r"(?:Started|The) .* as job (agent-[0-9a-f]{8})\b", text)
    return match.group(1) if match else ""


def format_dispatch_result(result: TaskResult | str, timeout: float, alias: str = "") -> str:
    """Format what :func:`dispatch_result` returned for the calling model.

    The sub-task's YAML result (``ran``, ``success`` and ``summary``
    keys), or the error message as it is.  *timeout* is the call's
    bound shown on the ``ran`` line; *alias* is the registered command
    name of a SEA reached by path, shown there too
    (:func:`~kiss.agents.sorcar.run_config.run_config_line`).
    """
    if isinstance(result, str):
        return result
    summary = result.text or ("" if result.success else "Task failed")
    return str(yaml.safe_dump(
        {
            "ran": run_config_line({**result.settings, "timeout": timeout}, alias),
            "success": result.success,
            "summary": summary,
        },
        sort_keys=False,
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
    timeout_explicit: bool = False,
) -> TaskResult | str:
    """Submit a SEA task to the kiss-web daemon and wait.

    The tail of :func:`_run_agent`: calls
    :func:`kiss.server.sorcar.run` with *agent_path* as its
    *extension_agent_path* and returns the daemon's
    :class:`~kiss.agents.sorcar.daemon_client.TaskResult` (which
    carries the sub-task's persisted ``task_id``) — or a clean error
    string.  It raises only for the inherited-budget case described
    under *inherit*.  Callers that need the sub-task's id
    (rsi7d's clone replays) use this; :func:`format_dispatch_result`
    formats the result for a model.

    Args:
        name: Display name of the agent for error messages (the
            channel name, or the script's file stem).
        prompt: The full prompt for the sub-task.
        agent_path: Absolute path of the SEA file.
        work_dir: Working directory for the sub-task; created when
            absent.
        model_name: LLM model for the sub-task; empty for the daemon
            default.
        budget: Per-task USD budget override; ``None`` for the daemon
            default.
        timeout: The calling ``run_agent`` call's bound in seconds,
            recorded on the daemon as the run's ``timeout`` (its
            ``task_settings`` and the lock check).  Not a deadline
            here: this function waits for the sub-task's result until
            it arrives or *cancel* is set; the caller bounds the wait
            by joining the thread it runs this on (:func:`_run_agent`).
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
            one is explicit, so it replaces the inherited or persisted
            default of its field and the SEA's ``settings()`` value
            (:data:`~kiss.agents.sorcar.sea_settings.PRECEDENCE_RULE`;
            a locked key was checked before the dispatch).
        inherit: Whether the arguments left empty are filled from
            *parent_agent* (see :func:`inherit_from_parent`): the
            caller's model (and, for the same model, its model
            configuration), a share of its remaining budget, its chat,
            its web-tools and memory settings, its live Docker
            container, and its effective worktree / auto-commit
            choices.  ``True`` for every ``run_agent`` dispatch except
            a ``kind: "channel"`` script or an ``inherit: false``
            option — a sub-task on the same project is a sub-agent of
            the caller, whereas a channel or cron session
            acts on an external service from a scratch directory on
            the host and must not inherit the caller's chat context or
            container.  ``False`` (the default) also for programmatic
            callers that pass every value explicitly (rsi7d's clone
            replays).  When the budget is inherited and the caller has
            nothing left to spend, ``BudgetExceededError`` propagates —
            the same signal a ``run_parallel`` fan-out raises.
        settings: The SEA's resolved settings
            (:func:`~kiss.agents.sorcar.sea_commands.sea_settings`),
            when the caller has them; a ``model`` in them blocks the
            ``model_config`` inheritance.  ``None`` means unknown.
        workspace: Workspace/account identifier a ``kind: "channel"``
            run holds for its lifetime (the daemon's task runner enters
            it); empty means ``"default"``.  Ignored by other scripts.
        cancel: The event that stops the sub-task (:func:`kill_agent_job`
            sets it); the wait then sends the daemon a ``stop``, awaits
            its confirmation and returns an error string saying so.
            ``None`` never stops it.
        timeout_explicit: Whether *timeout* was passed by the call
            (marked explicit in the run's ``provenance``) rather than
            taken from the script or the default.

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
    # inherits them.  Channel dispatches and explicit programmatic
    # callers skip this.
    explicit = explicit_keys(model_name, budget, options)
    if timeout_explicit:
        explicit.append("timeout")
    inherited = inherit_from_parent(
        parent_agent if inherit else None, model_name, budget, options,
        # The script's model applies only when the call names none.
        script_picks_model="model" in (settings or {}) and not model_name,
    )
    model_name, budget, options = inherited.model_name, inherited.budget, inherited.options
    # A sub-task of an unattended (cron) run inherits the no-questions
    # rule: without it a channel agent asked the user for an approval
    # nobody could give and blocked until the run_agent timeout.  It
    # travels in ``add_to_prompt``, which the daemon adds after an
    # SEA's ``prompt(task)`` has produced the prompt body.
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
    # ignore a user who turned them off.  These are inherited values,
    # so the SEA's ``settings()`` (the ``channel`` kind pins both off)
    # rank above them on the daemon (sea_settings.PRECEDENCE_RULE).
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
            classify_tasks=options.auto_classify,
            max_budget=budget,
            model_config=options.model_config,
            use_web_tools=options.use_web_tools,
            use_memory=options.use_memory,
            is_parallel=True if options.allow_fan_out is None else options.allow_fan_out,
            append_to_system_prompt=options.add_to_system_prompt,
            append_to_prompt=options.add_to_prompt,
            tool_profile=options.tool_profile,
            docker_image=inherited.docker_image,
            workspace=workspace,
            # The caller's extra tools (its script's ``add_to_tools()``)
            # cannot travel the wire: the daemon takes them off the
            # running caller, which ``parent_task_id`` names.
            inherit_tools=inherit,
            provenance=inherited.provenance(explicit),
            # No deadline on this wait: the call's bound is enforced by
            # the caller joining the job thread; the daemon only records
            # it.  The cancel event is the one way to stop the sub-task.
            timeout=None,
            record_timeout=timeout,
            endpoint_file=_daemon_endpoint_file(),
            cancel=cancel,
            running=running,
        )
    except daemon_client.CancelledError as e:
        # A confirmed stop carries the stopped task's spend, which still
        # counts towards the caller.
        _attribute_dispatch_usage(parent_agent, e.result, epoch)
        if not e.confirmed:
            return unconfirmed_stop_error(name)
        spend = ""
        if e.result is not None and e.result.cost:
            spend = f"; its ${e.result.cost:.4f} spend is counted in this task's cost"
        return (
            f"Error: the {name} agent task was stopped before it finished; work it "
            f"completed before the stop (side effects) is not reported here{spend}."
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
    """Run a SEA on a task immediately.

    The implementation behind the per-task ``run_agent`` tool built by
    :func:`make_run_agent_tool`, which captures *parent_work_dir*; the
    remaining arguments are the tool's (see its docstring).  One body
    for every agent: resolve the script (:func:`resolve_agent`), read
    its settings, pick the work directory, inherit from the caller
    unless the script is a channel or the call says ``inherit: false``,
    dispatch.

    Args:
        parent_work_dir: Work directory of the calling task.  A
            relative agent path or ``work_dir`` option is resolved
            against it and a sub-task runs in it by default.  Empty
            (standalone use) resolves relative paths against the
            process working directory and runs sub-tasks in
            ``agent_work`` under the Sorcar home.
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
        timeout: Maximum seconds this call blocks for the sub-task's
            result, as a number string; empty for the script's
            ``timeout`` setting, else
            :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`.  When it expires
            the sub-task keeps running as an ``agent_job`` and the
            call returns the job's notice (see :func:`job_notice`).
        options: JSON object of further run settings (see
            :func:`parse_run_options`).
        parent_agent: The agent calling ``run_agent``, when there is
            one; the sub-task's spend is folded into its task
            accounting (see :func:`_attribute_dispatch_usage`).
        wait: ``"false"`` returns at once with a job id for
            :func:`agent_job`; anything else blocks for the result.

    Returns:
        The sub-task's YAML result ("success" and "summary" keys), the
        job notice (``wait="false"``, or the ``timeout`` expired), or
        an error message.
    """
    if not task.strip():
        return "Error: task must be a non-empty string."
    try:
        run_options = parse_run_options(options, tool_profile, model, max_budget, timeout)
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
    seconds = resolve_timeout(run_options.timeout, settings)
    # sea_settings.PRECEDENCE_RULE: an explicit argument or option wins
    # over the SEA's ``settings()``, which win over what the calling
    # task passes on; a key the SEA locks may not be replaced.  A
    # ``kind: "channel"`` SEA (a channel agent, cron) takes nothing from
    # the calling task — not its chat, model, budget share, container
    # or prompt suffixes — and runs in its own ``work_dir`` (the kind's
    # scratch directory); so does a call with ``inherit: false``.  Every
    # other SEA is a sub-agent on the caller's project (or on the
    # ``work_dir`` option, resolved against it): the arguments left
    # empty are inherited from the calling agent (see
    # ``inherit_from_parent``), and the task's references to the main
    # checkout are rewritten to the caller's worktree.
    asked = explicit_values(run_options.model, run_options.max_budget, run_options)
    conflict = locked_conflicts(settings, asked, parent_work_dir)
    if conflict:
        return f"Error: {name}: {conflict}"
    channel = settings.get("kind") == "channel"
    if channel and run_options.inherit:
        return f"Error: {name}: a channel agent never inherits from the calling task"
    if run_options.workspace and not channel:
        return (
            f"Error: {name}: options['workspace'] applies to a channel agent only; "
            f"{name} is a {settings.get('kind') or 'session'} SEA"
        )
    inherit = not channel and run_options.inherit is not False
    work_dir = parent_work_dir or str(kiss_home() / "agent_work")
    if run_options.work_dir:
        requested = Path(run_options.work_dir).expanduser()
        work_dir = str(requested if requested.is_absolute() else Path(work_dir) / requested)
    else:
        work_dir = str(settings.get("work_dir") or "") or work_dir
    if inherit and DEFAULT_CONFIG.dispatch_path_rewrite:
        task = rewrite_parent_repo_paths(task, parent_work_dir)
    kwargs: dict[str, Any] = {
        "name": name, "prompt": task, "agent_path": agent_path, "work_dir": work_dir,
        "model_name": run_options.model, "budget": run_options.max_budget, "timeout": seconds,
        "parent_agent": parent_agent, "scope_work_dir": parent_work_dir,
        "options": run_options, "inherit": inherit, "settings": settings,
        "workspace": run_options.workspace, "timeout_explicit": run_options.timeout is not None,
        "alias": command_alias(agent_path) if is_agent_path(agent.strip()) else "",
    }
    job = start_agent_job(name, kwargs, parent_agent)
    try:
        if not blocking:
            wait_until_started(job)
            return job_notice(job, None)
        finished = join_agent_job(job, seconds)
    except BaseException:
        # The call was interrupted (the tool call's Stop button, or the
        # calling task stopped), whether while the sub-task was starting
        # or while it ran: the sub-task is cancelled too, as a blocking
        # call's sub-task always has been.  Not joined here, so the Stop
        # stays prompt: the job thread stops the sub-task on its own and
        # the run's end (:func:`kill_jobs_of`) collects what is left.
        job.cancel.set()
        raise
    if finished:
        forget_agent_job(job)
        return job.result
    return job_notice(job, seconds)


@dataclass
class AgentJob:
    """A ``run_agent`` sub-task running on its own thread.

    Every dispatch is one: a blocking call joins the thread for its
    ``timeout`` and takes the result when the thread finishes in time;
    otherwise (or with ``wait="false"``) the job stays registered and
    the caller's ``agent_job`` tool waits for, inspects or kills it.

    Attributes:
        job_id: The id ``agent_job`` looks the job up by.
        name: The agent's display name.
        owner: The agent whose ``run_agent`` started the job; only that
            agent's ``agent_job`` tool sees it.
        cancel: Set by :func:`kill_agent_job`; the dispatch's read loop
            stops the sub-task when it sees it.
        running: Set once the sub-task's tab exists on every client
            (its initial ``status running=true``) or the dispatch ended
            without one.
        thread: The thread running :func:`dispatch_result`, already
            started.
        started: ``time.monotonic()`` when the thread was started.
        workspace: The channel workspace the sub-task holds while it
            runs (``kind: "channel"`` SEAs), else ``""``.
        outcome: What :func:`dispatch_result` returned (the sub-task's
            :class:`TaskResult`, or an error string); ``None`` while
            running.
        result: The tool text for *outcome* (the YAML result or the
            error); ``""`` while running.
    """

    job_id: str
    name: str
    owner: Any
    cancel: threading.Event
    running: threading.Event
    thread: threading.Thread
    started: float = 0.0
    workspace: str = ""
    outcome: TaskResult | str | None = None
    result: str = ""


_AGENT_JOBS: dict[str, AgentJob] = {}
"""Every unresolved sub-task dispatched in this process, by job id.

A job leaves the registry when its blocking call takes its result or
when its owner's run ends (:func:`kill_jobs_of`).  Guarded by
:data:`_AGENT_JOBS_LOCK`.
"""

_AGENT_JOBS_LOCK = threading.Lock()


def start_agent_job(name: str, kwargs: dict[str, Any], owner: Any) -> AgentJob:
    """Start :func:`dispatch_result` with *kwargs* in a thread and register the job.

    The job is published only after its thread has started, so every
    ``agent_job`` action finds a joinable thread.  The function does
    not wait for the sub-task to start: a blocking ``run_agent`` call
    joins the thread under its one ``timeout`` (startup included) and
    a ``wait="false"`` call waits for the tab with
    :func:`wait_until_started`.

    Args:
        name: The agent's display name.
        kwargs: The keyword arguments of :func:`dispatch_result`, plus
            the ``alias`` :func:`format_dispatch_result` shows; the
            job's ``cancel`` and ``running`` events are added.
        owner: The calling agent (``None`` for standalone use).

    Returns:
        The registered job.
    """
    job_id = f"agent-{uuid.uuid4().hex[:8]}"
    cancel, running = threading.Event(), threading.Event()
    job = AgentJob(
        job_id, name, owner, cancel, running, threading.Thread(),
        workspace=str(kwargs.get("workspace") or ""),
    )
    job.thread = threading.Thread(
        target=_finish_agent_job,
        args=(job, {**kwargs, "cancel": cancel, "running": running}),
        name=f"agent-job-{job_id}", daemon=True,
    )
    job.started = time.monotonic()
    job.thread.start()
    with _AGENT_JOBS_LOCK:
        _AGENT_JOBS[job_id] = job
    return job


def wait_until_started(job: AgentJob) -> None:
    """Block until *job*'s sub-task has its tab (or the dispatch ended), bounded.

    A ``run_agent(wait="false")`` call returns its notice only after
    this, bounded by :data:`_JOB_START_GRACE_SECONDS`: the spawn then
    falls inside the call's time window, which is how every surface
    files a sub-agent tab under the call that started it.  Polled in
    slices like :func:`join_agent_job`, so a Stop of the call lands.
    """
    from kiss.core import tool_interrupt

    deadline = time.monotonic() + _JOB_START_GRACE_SECONDS
    while not job.running.is_set() and time.monotonic() < deadline:
        tool_interrupt.raise_if_interrupted()
        job.running.wait(_JOB_WAKE_SECONDS)


def _finish_agent_job(job: AgentJob, kwargs: dict[str, Any]) -> None:
    """Thread body of a job: record the dispatch's outcome and text on *job*."""
    alias = kwargs.pop("alias", "")
    try:
        job.outcome = dispatch_result(**kwargs)
        job.result = format_dispatch_result(job.outcome, kwargs["timeout"], alias)
    except BaseException as exc:  # noqa: BLE001 — the thread must record any failure
        logger.warning("agent job %s failed", job.job_id, exc_info=True)
        job.outcome = f"Error: the {job.name} agent task could not run: {safe_message(exc)}"
        job.result = job.outcome
    finally:
        job.running.set()


def job_notice(job: AgentJob, waited: float | None) -> str:
    """Return the ``run_agent`` text for a job that is still running.

    Args:
        job: The job.
        waited: The seconds the call blocked before giving up (its
            ``timeout``), or ``None`` for a ``wait="false"`` call that
            did not wait.

    Returns:
        The job id, what it holds, and how to use ``agent_job`` on it.
    """
    held = f' (it holds channel workspace "{job.workspace}")' if job.workspace else ""
    if waited is None:
        lead = f"Started the {job.name} agent task as job {job.job_id}{held}; its tab is open."
    else:
        lead = (
            f"The {job.name} agent task is still running after {waited:g}s as job "
            f"{job.job_id}{held}; its tab stays open."
        )
    return (
        f"{lead} agent_job({job.job_id!r}, 'wait') blocks until it finishes and "
        f"returns its result, 'tail' reports its status, 'kill' stops it. Wait for "
        f"or kill it before finishing; a job still running when this task ends is "
        f"killed."
    )


def join_agent_job(job: AgentJob, seconds: float | None) -> bool:
    """Block on *job*'s thread for at most *seconds* and say whether it finished.

    Joins in short slices rather than one long ``join``: a task's Stop
    is an asynchronously injected ``KeyboardInterrupt`` that Python
    delivers only between bytecodes, never inside a blocking C-level
    wait, and the tool call's own Stop button is a cooperative
    interrupt (:func:`kiss.core.tool_interrupt.raise_if_interrupted`)
    that must be polled.  Either propagates out of this function.

    Args:
        job: The job to wait for.
        seconds: The bound; ``None`` waits until the job finishes.

    Returns:
        ``True`` when the job's thread has finished, ``False`` when the
        bound expired first.
    """
    from kiss.core import tool_interrupt

    deadline = None if seconds is None else time.monotonic() + seconds
    while job.thread.is_alive():
        tool_interrupt.raise_if_interrupted()
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            return False
        job.thread.join(_JOB_WAKE_SECONDS if left is None else min(_JOB_WAKE_SECONDS, left))
    return True


def kill_agent_job(job: AgentJob) -> str:
    """Stop *job*'s sub-task and return the tool text for it.

    Sets the job's cancel event, which the dispatch's read loop turns
    into a daemon ``stop`` and a bounded wait for its confirmation, and
    joins the thread for :data:`_JOB_KILL_GRACE_SECONDS`.  A finished
    job is left as it is.

    Returns:
        The job's result text (the stop error, or the result it had
        already produced), or a still-running notice when the daemon
        did not answer within the grace.
    """
    job.cancel.set()
    job.thread.join(_JOB_KILL_GRACE_SECONDS)
    if job.thread.is_alive():
        return f"Job {job.job_id} ({job.name} agent task) is still running."
    return job.result


def forget_agent_job(job: AgentJob) -> None:
    """Drop *job* from the registry (its result has been taken)."""
    with _AGENT_JOBS_LOCK:
        _AGENT_JOBS.pop(job.job_id, None)


def agent_jobs_of(owner: Any) -> dict[str, AgentJob]:
    """Return the registered jobs *owner*'s ``run_agent`` started, by job id."""
    with _AGENT_JOBS_LOCK:
        return {job_id: job for job_id, job in _AGENT_JOBS.items() if job.owner is owner}


def live_agent_jobs(owner: Any) -> list[AgentJob]:
    """Return *owner*'s jobs whose sub-task is still running."""
    return [job for job in agent_jobs_of(owner).values() if job.thread.is_alive()]


def kill_jobs_of(owner: Any) -> list[str]:
    """Stop every running job of *owner* and drop all of its jobs from the registry.

    Called when *owner*'s run ends, so a sub-task the run neither
    waited for nor killed (a ``wait="false"`` job, a call whose
    ``timeout`` expired, or one whose call was interrupted) does not
    outlive it: every cancel goes out at once, each job thread turns
    it into the daemon's stop, and the sub-agent tab closes through the
    ordinary ``subagentDone`` flow.  The threads are joined for
    :data:`_JOB_END_GRACE_SECONDS` in all, long enough for a
    cooperative stop to confirm and settle the sub-task's spend into
    this run, short enough that a stopped parent still ends promptly
    (and before the daemon's stop watchdog injects a second interrupt
    into its cleanup); a job that takes longer finishes stopping on its
    own thread.  Finished jobs are dropped without a stop.

    Returns:
        The ids of the jobs that were still running.
    """
    jobs = agent_jobs_of(owner)
    live = [job for job in jobs.values() if job.thread.is_alive()]
    # Registry and cancels first: they must survive an interrupt that
    # lands in the bounded join below.
    for job in jobs.values():
        forget_agent_job(job)
    for job in live:
        job.cancel.set()
    deadline = time.monotonic() + _JOB_END_GRACE_SECONDS
    for job in live:
        job.thread.join(max(0.0, deadline - time.monotonic()))
    return [job.job_id for job in live]


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
        """Wait for, check or kill a sub-task ``run_agent`` left running.

        A job is a ``run_agent(..., wait="false")`` call's sub-task, or
        one whose call returned at its ``timeout`` while the sub-task
        kept running.

        Args:
            job_id: The id ``run_agent`` returned (``agent-1a2b3c4d``).
            action: ``"tail"`` (default) reports whether the task is
                still running and, once it has finished, its result;
                ``"wait"`` blocks until it finishes (at most
                ``timeout_seconds``) and returns its result; ``"kill"``
                stops it.
            timeout_seconds: How long ``"wait"`` blocks at most, as a
                number string; empty waits until the task finishes.

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
            join_agent_job(job, seconds)
        elif action == "kill":
            return kill_agent_job(job)
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
"""How long :func:`kill_agent_job` waits for the stopped sub-task's dispatch to return."""

_JOB_WAKE_SECONDS = 0.5
"""Slice length of :func:`join_agent_job`'s join loop (how fast a Stop is seen)."""

_JOB_END_GRACE_SECONDS = 3.0
"""How long :func:`kill_jobs_of` waits, in all, for a run's cancelled jobs at its end."""

_JOB_START_GRACE_SECONDS = 30.0
"""How long ``run_agent(wait="false")`` waits for the sub-task's tab before returning its notice."""


_GENERIC_AGENT_NAMES = frozenset({
    "general", "agent", "sorcar", "kiss", "analysis", "analyst", "assistant",
    "default", "llm", "model",
})
"""Names models invent for "another copy of me" (16 dispatches in the
7-day audit of 2026-09-19).  Each means what an empty ``agent`` means: a
plain Sorcar sub-agent (:data:`DEFAULT_AGENT_PATH`) on the task.
``worker`` is not one of them: it is a ``kind`` (a tool-bound run), so
``agent="worker"`` gets the usual "no such command" error instead of
silently running a ``session``; nor are the :data:`_REVIEWER_NAMES`,
which ask for a toolset, not an agent."""

_REVIEWER_NAMES = frozenset({"reviewer", "review", "codereview", "codereviewer"})
"""Names that mean "a read-only reviewer".  A reviewer is a plain
sub-agent with ``tool_profile="review"``; accepting the name as an
agent would silently run the full toolset, so :func:`resolve_agent`
refuses it and says how to spell the intent."""


def resolve_agent(agent: str, parent_work_dir: str) -> tuple[str, str] | str:
    """Resolve the ``run_agent`` tool's ``agent`` argument to a SEA path.

    Three rules, in order, for every spelling a model may use:

    1. empty, or a generic label such as ``"general"`` / ``"assistant"``
       (:data:`_GENERIC_AGENT_NAMES`): the plain sub-agent SEA
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
    if is_agent_path(requested):
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


def is_agent_path(agent: str) -> bool:
    """Return whether a ``run_agent`` ``agent`` argument is a path (rule 2 of ``resolve_agent``)."""
    return agent.endswith(".py") or "/" in agent or "\\" in agent


def command_alias(agent_path: str) -> str:
    """Return the registered command name whose script is *agent_path*, or ``""``.

    Lets a call that reached a bundled or ``SEAS.md`` SEA by its path
    learn the name it could have used instead
    (``run_agent(agent="review_paper", ...)``).
    """
    from kiss.agents.sorcar import sea_commands

    for name in sea_commands.list_commands():
        if str(sea_commands.get_command(name)) == agent_path:
            return name
    return ""


def resolve_timeout(timeout: float | None, settings: dict[str, Any]) -> float:
    """Resolve how long a ``run_agent`` call waits for its sub-task, in seconds.

    The call's ``timeout`` wins; ``None`` takes the agent script's own
    ``timeout`` setting (``settings()["timeout"]``, see
    :mod:`kiss.agents.sorcar.sea_settings`), else
    :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`.

    Args:
        timeout: The call's parsed ``timeout`` argument or option
            (:attr:`RunOptions.timeout`), ``None`` when not passed.
        settings: The resolved settings of the script the sub-task
            runs.

    Returns:
        The seconds to wait.
    """
    if timeout is not None:
        return timeout
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
    if squashed in _REVIEWER_NAMES:
        return (
            f"Error: {agent!r} is not an agent. A reviewer is a plain sub-agent with the "
            f"read-only toolset: leave agent empty and pass tool_profile=\"review\"."
        )
    by_squashed = {_squash(name): name for name in commands}
    close = difflib.get_close_matches(squashed, list(by_squashed), n=1, cutoff=0.6)
    hint = f" Did you mean {by_squashed[close[0]]!r}?" if close else ""
    return (
        f"Error: unknown agent {agent!r} — not a registered slash command "
        f"and not a path to a .py SEA file.{hint} Leave agent empty for "
        f"a plain sub-agent. Commands: {', '.join(commands) or 'none registered'}."
    )


def make_run_agent_tool(
    work_dir: str, parent_agent: Any = None,
) -> Callable[..., str]:
    """Build the ``run_agent`` tool for the agent running in *work_dir*.

    The tool executes in the daemon process, whose own working
    directory is unrelated to the user's project, so the calling
    task's work directory must be captured here (exactly like
    ``make_skill_tool``): it anchors relative SEA paths and
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
        """Run a SEA (a channel agent, a slash command, cron or a ``.py`` file) on a task now.

        Call it RIGHT AWAY, without exploring any source code, when the
        task is to act on an external messaging service, mailbox or
        device channel (pass the request through as the task; the
        channel agent has its own authenticated tools), to manage
        scheduled automations (``agent="cron"``; a messaging gateway is
        a cron task too, see SYSTEM.md), or whenever the user names an
        agent file or a slash command to run a task with.

        Available channels: {channels}.  ``"cron"`` is always available.

        {precedence}

        A sub-task inherits what this task passes on: its model, half
        of its remaining budget, chat, prompt suffixes, extra tools,
        container and worktree / auto-commit / fan-out choices.  A
        channel or cron sub-task inherits none of these and runs in
        the Sorcar home's ``channel_work`` directory.  The call blocks
        until the task finishes or ``timeout`` expires; at ``timeout``
        the task keeps running as an ``agent_job`` and the call returns
        its job id (``agent_job(id, "wait")`` collects the result,
        ``"kill"`` stops it; a job still running when this task ends is
        killed).

        Args:
            task: The task text, e.g. "Send 'hello' to #sorcar"; the
                SEA's ``prompt(task)``, if defined, shapes it.
            agent: Empty = a plain Sorcar sub-agent; a slash-command
                name (``"slack"``, ``"cron"``, ``"write_paper"``) = that
                command; a path ending in ``.py`` (relative to this
                task's work directory) = that SEA file.  A generic label
                (``"general"``, ``"assistant"``) also means the plain
                sub-agent; ``"reviewer"`` is not an agent (pass
                ``tool_profile="review"`` instead).
            model: LLM model; empty = this task's model (the daemon
                default for a channel/cron sub-task).
            tool_profile: ``"review"`` (read-only; ``"readonly"`` is
                accepted for it), ``"shell"``, ``"assistant"``,
                ``"bash"``, ``"none"`` or groups joined with ``+``
                (``"shell+edit+browser"``), as for ``run_parallel``;
                empty = the full toolset, except that a sub-task of a
                reviewer, or one whose task reads as a review, gets
                ``"review"`` unless the task asks for changes (the
                ``ran`` line then says ``tools=review(inferred)``).
            max_budget: USD budget as a number string; empty = half of
                this task's remaining budget (the daemon default for a
                channel/cron sub-task).
            timeout: Seconds this call blocks, as a number string;
                empty = the SEA's ``timeout`` setting, else 3600.  When
                it expires the task is not stopped: the call returns
                its job id for ``agent_job``.
            options: JSON object of run settings to override, e.g.
                ``'{"use_web_tools": false}'``; usually empty.  Its
                keys are the SEA settings vocabulary: ``model``,
                ``tool_profile``, ``max_budget``, ``timeout`` (the
                four arguments above are shortcuts for these),
                ``work_dir`` (relative to this task's; a SEA's own
                ``work_dir`` setting is relative to the SEA's folder),
                ``chat_id``,
                ``workspace`` (the account of a multi-account channel),
                ``add_to_system_prompt`` / ``add_to_prompt`` (appended
                text), ``model_config`` (JSON object), ``docker_image``,
                and the booleans ``inherit`` (``false`` = take nothing
                from this task), ``use_worktree``, ``auto_commit``,
                ``auto_classify``, ``use_web_tools``, ``use_memory``,
                ``allow_fan_out``.
            wait: ``"false"`` = return a job id at once; then
                ``agent_job(job_id, "wait")`` returns the result,
                ``"tail"`` its status, ``"kill"`` stops it.  Use it to
                run several agents at once or to keep working while a
                long sub-task runs; wait for or kill every job before
                finishing.  Empty (default) blocks until done.

        Returns:
            The sub-task's YAML result: a ``ran`` line first (the SEA
            and kind it ran as, its model, tool profile, budget and
            timeout, which values it inherited from this task, which
            inherited or default values the SEA pinned to its own, e.g.
            ``pinned=use_worktree(True->False)``, and, when the
            pre-run classifier dropped a worktree default for a
            non-development task, ``classified=use_worktree(True->False)``
            — an explicit ``use_worktree`` option is never dropped; a
            SEA reached by path that is also a command ends with
            ``(also agent="name")``),
            then ``success`` and ``summary``; the job notice
            (``wait="false"``, or ``timeout`` expired with the task
            still running); or an error message (unknown agent — naming
            the closest command — or a locked key the call contradicts).
        """
        return _run_agent(
            work_dir, task, agent, model, tool_profile, max_budget,
            timeout, options, parent_agent, wait,
        )

    run_agent.__doc__ = (
        (run_agent.__doc__ or "")
        .replace("{channels}", ", ".join(available_channels()) or "none installed")
        .replace("{precedence}", PRECEDENCE_RULE)
    )
    run_agent.unknown_arguments_hint = options_keyword_hint  # type: ignore[attr-defined]
    return run_agent
