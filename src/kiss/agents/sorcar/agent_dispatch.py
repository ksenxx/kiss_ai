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

1. the SEA is loaded in the calling process
   (:func:`kiss.agents.sorcar.sea_commands.load_sea`) for its class (a
   channel derives from ``ChannelSea``) and the ``timeout`` and
   ``work_dir`` of its effective ``settings()``;
2. unless the SEA is a channel or the ``inherit`` option is
   ``false``, the arguments the call left empty are inherited from the
   calling agent (:func:`inherit_from_parent`:
   model, budget share, chat, prompt suffixes, web/memory flags,
   container, extra tools);
3. the sub-task is submitted to the kiss-web daemon
   (:func:`kiss.agents.sorcar.daemon_client.run` with the script as
   ``sea_path``) as a nested sub-agent of the calling task,
   and the call blocks for its result — or, with ``wait="false"``,
   returns a job id at once for :func:`agent_job` to wait on, check or
   kill.

The daemon executes the script once more, applies its settings and
getters (:mod:`kiss.agents.sorcar.sea_settings`,
:mod:`kiss.agents.sorcar.sea_apply`) and, for a channel, holds the
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

from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import math
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar._concurrency import locked, locked_call
from kiss.agents.sorcar.daemon_client import TaskResult
from kiss.agents.sorcar.fanout_guard import is_review_task, parse_tasks_json
from kiss.agents.sorcar.machines import find_machine, is_local_machine, load_machines
from kiss.agents.sorcar.run_config import (
    PROVENANCE_EXPLICIT,
    PROVENANCE_INHERITED,
    run_config_line,
)
from kiss.agents.sorcar.sea_commands import _third_party_dir, sea_script_in
from kiss.agents.sorcar.sea_settings import (
    META_SETTINGS,
    PRECEDENCE_RULE,
    REMOVED_SETTINGS,
    RENAMED_OPTIONS,
    SETTING_TYPES,
    SeaError,
    anchored_work_dir,
    declares_channel,
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
from kiss.core.kiss_error import BudgetExceededError

logger = logging.getLogger(__name__)

DEFAULT_AGENT_PATH = str(
    Path(__file__).resolve().parents[1] / "seas" / "sorcar" / "sorcar_sea.py"
)
"""SEA run when the ``run_agent`` tool's ``agent`` is empty.

The bundled ``src/kiss/agents/seas/sorcar/sorcar_sea.py`` — a hidden SEA that
pins no setting, so the sub-task is a plain Sorcar session on the given task
in the calling task's work directory (path mode).  Held as the absolute
path of the
installed file so the default works from any work directory, not only
a checkout of this repository.
"""

EMPTY_TASK_ERROR = "Error: task must be a non-empty string."
"""The ``run_agent`` / ``run_parallel`` result for a blank task text."""

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
    argument: one field per key of :data:`RUN_OPTION_KEYS` — the SEA
    settings vocabulary (:data:`~kiss.agents.sorcar.sea_settings.SETTING_TYPES`)
    minus the keys that describe a script — plus ``system_prompt``, the
    replacement base system prompt a programmatic caller may pass (it
    is not an ``options`` key: a SEA's ``system_prompt`` method is the
    user-facing way).  The tool's ``model``, ``tool_profile``,
    ``max_budget`` and ``timeout`` arguments land in the fields of the
    same name (:func:`parse_run_options` puts them there; the
    ``options`` object refuses those keys).
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
    model_config: dict[str, Any] | None = None
    use_web_tools: bool | None = None
    auto_classify: bool | None = None
    use_memory: bool | None = None
    tool_profile: str = ""
    docker_image: str = ""
    inherit: bool | None = None
    workspace: str = ""
    add_to_prompt: str = ""
    add_to_system_prompt: str = ""
    system_prompt: str = ""


ARGUMENT_OPTIONS = ("model", "tool_profile", "max_budget", "timeout")
"""The run settings the ``run_agent`` / ``run_parallel`` tools take as arguments.

Each has exactly one place in a call: the argument.  Inside the
``options`` JSON object the same key is refused with a message naming
the argument, so a value is never given twice.
"""

OPTION_TYPES: dict[str, type | tuple[type, ...]] = {
    **{
        key: expected for key, expected in SETTING_TYPES.items()
        if key not in META_SETTINGS and key not in ARGUMENT_OPTIONS
    },
    "inherit": bool,
    "workspace": str,
    "add_to_prompt": str,
    "add_to_system_prompt": str,
}
"""The keys the ``options`` JSON object accepts, with the type of each value.

The SEA settings vocabulary
(:data:`~kiss.agents.sorcar.sea_settings.SETTING_TYPES`) minus the
keys that describe a script rather than a run
(:data:`~kiss.agents.sorcar.sea_settings.META_SETTINGS`: ``locked``,
``hidden``) and minus the four settings the
tool takes as arguments (:data:`ARGUMENT_OPTIONS`), so the tool and
the SEAs share one vocabulary: what a SEA may pin in ``settings()``, a
caller may pass in ``options`` or, for those four, as the argument.
Plus four call-only keys: ``inherit`` (``false``: the sub-task takes
nothing from the calling task), ``workspace`` (the account a channel
agent's run holds), ``add_to_prompt`` (text appended to the task) and
``add_to_system_prompt`` (text appended to the system prompt).
"""

RUN_OPTION_KEYS = (*ARGUMENT_OPTIONS, *OPTION_TYPES)
"""Every run setting a call may pass, as an argument or an ``options`` key: the
:class:`RunOptions` fields except ``system_prompt``."""

OPTION_DOCS: dict[str, str] = {
    "work_dir": "The directory the sub-task works in; a relative path is a path under the "
                "calling task's directory, as a SEA's own `work_dir` setting is.",
    "inherit": "`false`: the sub-task takes nothing from the calling task (no model, chat, "
               "prompt suffixes, tools or container; its budget is the daemon default); "
               "default `true`. A channel never inherits, so `true` is refused there.",
    "workspace": "The account a channel's run holds (its channel workspace); refused for "
                 "any SEA that is not a channel.",
    "add_to_prompt": "Text appended to the task after the SEA's `prompt(task)`.",
    "add_to_system_prompt": "Text appended to the system prompt before the SEA's "
                            "`system_prompt(system_prompt)` sees it.",
}
"""Documentation of the option keys that are not ``settings()`` keys, or whose option
form needs its own words (``work_dir``), for ``sea docs``.

Every other option is documented by
:data:`~kiss.agents.sorcar.sea_settings.SETTING_DOCS`.
"""

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
    """Parse the ``options`` argument of ``run_agent`` / ``run_parallel`` and the four
    settings the tools take as arguments.

    Args:
        options: A JSON object string in the settings vocabulary
            (:data:`OPTION_TYPES`), or empty for no overrides.
            Booleans may also be given as the strings ``"true"`` /
            ``"false"``; ``null`` means "not passed".
        tool_profile: The tool's ``tool_profile`` argument; empty when
            not passed.  Canonicalised by
            :func:`kiss.agents.sorcar.sorcar_agent.canonical_tool_profile`
            (``readonly`` becomes ``review``).
        model: The tool's ``model`` argument.
        max_budget: The tool's ``max_budget`` argument (a positive
            finite number as text).
        timeout: The tool's ``timeout`` argument (seconds as text).

    Returns:
        The parsed options with the four arguments in the fields of the
        same name.

    Raises:
        ValueError: When *options* is not a JSON object, names an
            unknown key or one of the four argument settings
            (:data:`ARGUMENT_OPTIONS`), has a value of the wrong type,
            a number is not positive and finite, or the tool profile is
            unknown.
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
    return RunOptions(
        **parsed,
        model=model.strip(),
        tool_profile=canonical_tool_profile(tool_profile),
        max_budget=_parse_number("max_budget", max_budget),
        timeout=_parse_number("timeout", timeout),
    )


def options_keyword_hint(unknown: dict[str, Any]) -> str:
    """Explain the keywords a ``run_agent`` / ``run_parallel`` call passed that are no argument.

    Installed on the two tools as their ``unknown_arguments_hint``
    (:func:`kiss.core.kiss_agent.unknown_argument_hint`): a run
    setting passed as a keyword (``use_memory=False``) is shown as
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
        if key in RENAMED_OPTIONS and RENAMED_OPTIONS[key] in ARGUMENT_OPTIONS:
            lines.append(
                f"{key} was renamed to {RENAMED_OPTIONS[key]}; pass "
                f"{RENAMED_OPTIONS[key]}={json.dumps(value)}."
            )
        elif key in RENAMED_OPTIONS:
            example = json.dumps({RENAMED_OPTIONS[key]: value})
            lines.append(
                f"{key} was renamed to {RENAMED_OPTIONS[key]}; pass options='{example}'."
            )
        elif key in REMOVED_SETTINGS:
            lines.append(f"{key} was removed: {REMOVED_SETTINGS[key]}.")
        else:
            close = _closest_option(key)
            hint = f" Did you mean the option {close!r}?" if close else ""
            lines.append(f"{key} is neither an argument nor an options key.{hint}")
    return "\n".join(lines)


def _closest_option(key: str) -> str:
    """Return the options key spelled closest to the unknown *key*, or ``""``."""
    close = difflib.get_close_matches(key, list(OPTION_TYPES), n=1, cutoff=0.7)
    return close[0] if close else ""


def _parse_option(key: str, value: Any) -> Any:
    """Parse one ``options`` entry; ``None`` (for ``null`` and a blank string too) is "not passed".

    Raises:
        ValueError: When *key* is unknown (naming the current key for a
            renamed setting), is a setting the tool takes as an argument
            (naming the argument) or *value* has the wrong type.
    """
    expected = OPTION_TYPES.get(key)
    if expected is None:
        if key in ARGUMENT_OPTIONS:
            raise ValueError(
                f"options key {key!r} is the {key} argument of this tool; "
                f"pass {key}=... instead of putting it in options."
            )
        if key in RENAMED_OPTIONS:
            raise ValueError(
                f"options key {key!r} was renamed to {RENAMED_OPTIONS[key]!r}; "
                f"use the new name."
            )
        if key in REMOVED_SETTINGS:
            raise ValueError(f"options key {key!r} was removed: {REMOVED_SETTINGS[key]}.")
        close = _closest_option(key)
        hint = f" Did you mean {close!r}?" if close else ""
        raise ValueError(
            f"options has an unknown key {key!r}; known keys: {', '.join(OPTION_TYPES)}.{hint}"
        )
    if value is None:
        return None
    if expected is bool:
        return _parse_bool(key, value)
    if isinstance(value, str) and expected is str:
        if not value.strip():
            return None
        if key in ("chat_id", "work_dir", "workspace"):
            return value.strip()
        return value
    if isinstance(value, expected) and not isinstance(value, bool):
        return value
    raise ValueError(
        f"options[{key!r}] must be a JSON {getattr(expected, '__name__', 'number')}, "
        f"got {type(value).__name__}."
    )


def available_channels() -> list[str]:
    """Return the names of the installed third-party channel agents.

    A channel is a SEA folder ``<channel>/<channel>_sea.py`` of the
    third-party agents package whose class derives from ``ChannelSea``
    and whose ``settings()`` do not write ``"hidden": True``
    (CHANNEL_BEHAVIOURS "listed as a channel").  The package's other SEAs
    (``a2a``, a session SEA whose tools call peer agents; ``oai``, a
    hidden server set up from a terminal) are not channels.  The scan
    reads the directory listing and parses the two literals from
    source — no channel module is imported.

    Returns:
        Sorted channel names, e.g. ``["discord", ..., "slack", ...]``;
        empty when the package is absent.
    """
    package_dir = _third_party_dir()
    if package_dir is None:
        return []
    return sorted(
        sea_dir.name
        for sea_dir in package_dir.iterdir()
        if not sea_dir.name.startswith("_")
        and sea_script_in(sea_dir).is_file()
        and declares_channel(sea_script_in(sea_dir))
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


def _attribute_dispatch_usage(
    parent_agent: Any, result: Any, epoch: Any = None, parent_task_id: str = "",
) -> None:
    """Fold a dispatched sub-task's spend into the calling task.

    The daemon's terminal ``result`` event carries the sub-task's
    cost, tokens, and steps (parsed into the
    :class:`~kiss.agents.sorcar.daemon_client.TaskResult`).  Without
    this fold, that spend would vanish from the calling task's
    accounting — the parent's end-of-task cost, its live usage
    header, and its persisted per-task cost would all lie low.

    A calling task with a persisted ``task_history`` row and a server
    printer is charged through the printer's ``charge_task_usage``
    bridge (:func:`~kiss.server.task_update.charge_side_channel_usage`):
    while its row is unfinished the spend is banked on its live
    ledger (bound to *epoch*) and its new totals go out as a
    ``usage_info``, so the chat header follows every fold — including
    one landing after the run's terminal ``result``; once the row is
    finished (a job that outlived :func:`kill_jobs_of`'s grace and
    settled after the row was saved) the spend is added to the row and
    its finished ancestors instead, with a persisted ``usage_info``
    for each, so the history, the Spend panel and the replayed
    transcript all show it.  A caller without a row or without that
    printer (standalone use) gets the plain ledger fold
    (:func:`~kiss.agents.sorcar.sorcar_agent._attribute_sub_usage`).

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
        parent_task_id: The calling task's persisted row id captured
            when the dispatch started (the agent may since have moved
            on to another task), or ``""`` when it has none.
    """
    if parent_agent is None or result is None:
        return
    cost = float(getattr(result, "cost", 0.0) or 0.0)
    tokens = int(getattr(result, "tokens", 0) or 0)
    steps = int(getattr(result, "steps", 0) or 0)
    try:
        charge = getattr(getattr(parent_agent, "printer", None), "charge_task_usage", None)
        if parent_task_id and callable(charge):
            charge(parent_agent, parent_task_id, cost, tokens, steps, epoch=epoch)
            return
        from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

        _attribute_sub_usage(parent_agent, cost, tokens, steps, epoch=epoch)
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
    ``run_parallel`` child gets, since that is N ``run_agent`` calls —
    so a sub-task dispatched through the daemon behaves like a
    sub-agent of the caller instead of like a task typed into a fresh
    chat panel.
    """

    model_name: str
    budget: float | None
    options: RunOptions
    docker_image: str
    fields: tuple[str, ...] = ()
    """The setting keys that were filled from the caller (``model``, ``chat_id``, ...)."""

    def provenance(self, explicit: Iterable[str]) -> dict[str, str]:
        """Return the ``provenance`` wire field of a run: ``{setting key: explicit | inherited}``.

        Args:
            explicit: The setting keys the call passed itself (the keys
                of :func:`explicit_values`).
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
        key: getattr(options, key) for key in RUN_OPTION_KEYS
        if getattr(options, key, None) not in (None, "")
    }
    if model_name:
        values["model"] = model_name
    if budget is not None:
        values["max_budget"] = budget
    return values


def _filled_keys(asked: Inherited, got: Inherited) -> tuple[str, ...]:
    """Return the setting keys whose value *got* has and *asked* left empty.

    The five settings :class:`Inherited` resolves outside ``options``
    are compared on those attributes; the rest on the ``options``
    fields.
    """
    pairs = [
        ("model", asked.model_name, got.model_name),
        ("max_budget", asked.budget, got.budget),
        ("docker_image", asked.docker_image, got.docker_image),
    ]
    resolved_outside_options = {key for key, _, _ in pairs}
    pairs.extend(
        (key, getattr(asked.options, key), getattr(got.options, key))
        for key in RUN_OPTION_KEYS if key not in resolved_outside_options
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
    fields without touching ``modelConfig`` (``apply_sea``),
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
    siblings: int = 1,
    reserving: Sequence[AgentJob] = (),
) -> Inherited:
    """Fill the empty ``run_agent`` arguments from the calling agent.

    The ONE inheritance table of a sub-agent: ``run_agent`` dispatches
    and ``run_parallel`` fan-outs (N dispatches) both build their
    children's arguments here.  An explicit argument always wins; only
    an empty one is inherited:

    - ``model_name``: the caller's model.
    - ``model_config``: the caller's, but ONLY when the sub-task runs
      the model the caller was launched with and the SEA
      does not pick its own model — an endpoint and its key belong to
      that model, so a different model (one the caller switched to
      with ``set_model``, or one the script's ``model`` setting
      selects on the daemon) runs with default provider routing (see
      :func:`_parent_model_config`).
    - ``budget``: the caller's remaining budget, less what its
      still-running sub-tasks hold (*reserving*), shared among the
      *siblings* sub-tasks the call starts plus the caller
      (``_subagent_budget_share(siblings)``: half of what is left for
      one sub-task, the other half reserved for the caller to process
      the result), ``None`` (daemon default) when the caller has no
      budget context yet.
    - ``chat_id``: the caller's chat, so the sub-task starts with the
      conversation's earlier tasks and results as context.
    - ``system_prompt`` / ``add_to_system_prompt``: the caller's
      own replacement base prompt (``_base_system_prompt``, blank
      unless its run was given one — the classifier's SYSTEM vs
      SYSTEM_LITE choice is never stored there, so the sub-task is
      still classified on its own) and append-only suffix
      (``_system_prompt_suffix``), so a run's extra system
      instructions constrain its whole task tree through ``run_agent``
      exactly as through ``run_parallel``.  What the caller's own
      SEA's ``system_prompt`` method returned is not forwarded, as
      its ``prompt`` method's return is not: the sub-task's SEA's
      ``system_prompt`` method sees the assembled prompt (base plus
      suffix) on the daemon and returns the sub-task's.
    - ``add_to_prompt``: the suffix the caller's own task prompt
      was given (``_prompt_suffix``, the ``appendToPrompt`` of its
      run), so the sub-task's prompt ends with the same text.  An
      SEA's ``prompt(task)`` method then rewrites the whole
      task text on the daemon.
    - the caller's extra tools (its SEA's ``tools()``
      list, plus those the caller inherited itself): not resolved
      here — a callable cannot travel the wire — but requested from
      the daemon with the ``inherit_tools`` flag :func:`dispatch_result`
      sends, so the sub-task has the tools the inherited system prompt
      refers to.  The daemon adds them to the sub-task's built-in
      toolset after the sub-task's own SEA's ``tools()``
      tools; a script on the ``none`` tool profile keeps exactly its
      own set.
    - ``use_web_tools`` / ``use_memory``: the caller's per-run
      settings (``_use_web_tools`` / ``_use_memory_override``).
    - ``docker_image``: the call's own ``docker_image`` option, else
      ``container:<id>`` of the caller's live Docker
      container, so the sub-task acts in the same container instead of
      on the host.

    Args:
        parent_agent: The agent calling ``run_agent``; ``None`` (no
            caller) inherits nothing.
        model_name: The call's ``model_name`` argument; empty inherits.
        budget: The call's parsed ``max_budget``; ``None`` inherits.
        options: The call's parsed optional arguments.
        script_picks_model: Whether the sub-task's SEA names
            a ``model`` in its ``settings()``, which blocks the
            ``model_config`` inheritance.
        siblings: How many sub-tasks the call starts at once.
        reserving: The caller's sub-tasks that were running when the
            call was planned (:func:`live_agent_jobs`, read once per
            call by :func:`plan_run_agent`); what those still unfinished
            now hold (:func:`reserved_budget`) is taken off the
            remainder before the split.

    Returns:
        The resolved values.

    Raises:
        BudgetExceededError: When the budget is inherited and the
            caller has nothing left to spend, or its running sub-tasks
            hold all of it.  Raised through a job's thread
            (``run_agent``, a ``run_parallel`` child) it becomes that
            sub-task's error result text instead.
    """
    asked = Inherited(model_name, budget, options, options.docker_image)
    if parent_agent is None:
        return asked
    got = _inherit_from_parent(
        parent_agent, model_name, budget, options, script_picks_model, siblings, reserving,
    )
    return dataclasses.replace(got, fields=_filled_keys(asked, got))


def _unreserved_share(
    parent_agent: Any, share: float | None, siblings: int, reserved: float,
) -> float | None:
    """Return *share* less the part of *reserved* that falls on one sub-task.

    ``_subagent_budget_share(siblings)`` splits the caller's remainder
    ``remaining / (siblings + 1)`` ways; splitting ``remaining -
    reserved`` the same way is *share* minus ``reserved / (siblings +
    1)``, so the caller still keeps its own share of what is really
    left after the budgets its running sub-tasks already hold.

    Raises:
        BudgetExceededError: When those budgets hold all of the
            remainder.
    """
    if share is None or reserved <= 0:
        return share
    left = share - reserved / (max(siblings, 0) + 1)
    if left <= 0:
        raise BudgetExceededError(
            f"Agent {getattr(parent_agent, 'name', '?')} has no remaining budget for a "
            f"sub-task: ${reserved:.4f} of its remainder is reserved by sub-tasks that are "
            "still running"
        )
    return left


def _inherit_from_parent(
    parent_agent: Any,
    model_name: str,
    budget: float | None,
    options: RunOptions,
    script_picks_model: bool,
    siblings: int,
    reserving: Sequence[AgentJob],
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
            # The reservations before the ledger: a sub-task that
            # finishes in between is then counted twice for this one
            # call (its spend is on the ledger and its budget still
            # held), never dropped from both.
            reserved = reserved_budget(reserving)
            budget = _unreserved_share(parent_agent, share(siblings), siblings, reserved)
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
    return Inherited(model_name, budget, options, docker_image)


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


def format_dispatch_result(
    result: TaskResult | str, timeout: float, alias: str = "", machine: str = "",
) -> str:
    """Format what :func:`dispatch_result` returned for the calling model.

    The sub-task's YAML result (``ran``, ``success`` and ``summary``
    keys), or the error message as it is.  *timeout* is the call's
    bound shown on the ``ran`` line; *alias* is the registered command
    name of a SEA reached by path, shown there too
    (:func:`~kiss.agents.sorcar.run_config.run_config_line`); *machine*
    is the remote machine the sub-task ran on, ending the line with
    ``machine=<name>`` (empty: it ran here, nothing is shown).
    """
    if isinstance(result, str):
        return result
    summary = result.text or ("" if result.success else "Task failed")
    ran = run_config_line({**result.settings, "timeout": timeout}, alias)
    return str(yaml.safe_dump(
        {
            "ran": f"{ran} machine={machine}" if machine else ran,
            "success": result.success,
            "summary": summary,
        },
        sort_keys=False,
    ))


def dispatch_result(
    name: str,
    prompt: str,
    sea_path: str,
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
    job: AgentJob | None = None,
    timeout_explicit: bool = False,
    siblings: int = 1,
    machine: str = "",
    sea_command: str = "",
    reserving: Sequence[AgentJob] = (),
) -> TaskResult | str:
    """Submit a SEA task to the kiss-web daemon and wait.

    The tail of :func:`start_run_agent`'s job: calls
    :func:`kiss.server.sorcar.run` with *sea_path* as its
    *sea_path* and returns the daemon's
    :class:`~kiss.agents.sorcar.daemon_client.TaskResult` (which
    carries the sub-task's persisted ``task_id``) — or a clean error
    string.  Every production dispatch runs this on a job thread
    (:func:`start_agent_job`), which records anything it raises (the
    inherited-budget case described under *inherit*) as the job's
    error text.  Callers that need the sub-task's id
    (rsi7d's clone replays) use this; :func:`format_dispatch_result`
    formats the result for a model.

    Args:
        name: Display name of the agent for error messages (the
            channel name, or the script's file stem).
        prompt: The full prompt for the sub-task.
        sea_path: Absolute path of the SEA file.
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
            it arrives or *job*'s cancel is set; the caller bounds the wait
            by joining the thread it runs this on (:func:`_run_agent`,
            :func:`run_agents_parallel`).
        parent_agent: The agent calling ``run_agent``, when there is
            one: the sub-task's cost/tokens/steps are folded into its
            task accounting (see :func:`_attribute_dispatch_usage`),
            and its persisted task id / frontend tab id make the
            sub-task a nested sub-agent of the calling task.
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
            its web-tools and memory settings, and its live Docker
            container.  ``True`` for every ``run_agent`` dispatch except
            a channel SEA or an ``inherit: false``
            option — a sub-task on the same project is a sub-agent of
            the caller, whereas a channel or cron session
            acts on an external service from a scratch directory on
            the host and must not inherit the caller's chat context or
            container.  ``False`` (the default) also for programmatic
            callers that pass every value explicitly (rsi7d's clone
            replays).  When the budget is inherited and the caller has
            nothing left to spend, ``BudgetExceededError`` propagates
            (the job thread turns it into the job's error text).
        settings: The SEA's resolved settings
            (:func:`~kiss.agents.sorcar.sea_commands.sea_settings`),
            when the caller has them; a ``model`` in them blocks the
            ``model_config`` inheritance.  ``None`` means unknown.
        workspace: Workspace/account identifier a channel
            run holds for its lifetime (the daemon's task runner enters
            it); empty means ``"default"``.  Ignored by other scripts.
        job: The job this dispatch runs as (:func:`start_agent_job`),
            when there is one.  Its ``cancel`` event stops the sub-task
            (:func:`kill_agent_job` sets it; the wait then sends the
            daemon a ``stop``, awaits its confirmation and returns an
            error string saying so), its ``running`` event is set when
            the sub-task's tab exists, and its ``budget`` records the
            budget the sub-task was given so later calls of the same
            caller take it off their share (*reserving*).  ``None``
            (standalone use) never stops the sub-task.
        timeout_explicit: Whether *timeout* was passed by the call
            (marked explicit in the run's ``provenance``) rather than
            taken from the script or the default.
        siblings: How many sub-tasks the calling tool call starts at
            once; an inherited budget is the caller's remaining budget
            shared among them plus the caller.
        machine: The name of the registered remote machine the
            sub-task runs on (see :func:`plan_run_agent`); *work_dir*
            is then a path there and is not created here, and the
            caller's worktrees are not reported.  Empty (the default)
            runs it on this daemon.
        sea_command: The registered command the machine runs its own
            copy of, instead of *sea_path* (see :func:`remote_sea`);
            empty sends *sea_path*.
        reserving: The caller's sub-tasks that were running when the
            call was planned (:func:`live_agent_jobs`, read once per
            tool call by :func:`plan_run_agent`); what those still
            unfinished hold when the budget is resolved is taken off
            the caller's remainder before an inherited budget is split.

    Returns:
        The sub-task's :class:`TaskResult`, or an error message.
    """
    from kiss.agents.sorcar import daemon_client

    epoch = dispatch_epoch(parent_agent)
    if not machine:
        # A caller whose worktree was already torn down (a finished
        # worktree task) hands over the removed directory; creating it
        # here would leave an unregistered husk under ``.kiss-worktrees/``.
        work_dir = str(remap_vanished_worktree(Path(work_dir)))
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    kwargs = dispatch_command(
        prompt, sea_path, work_dir, model_name, budget, timeout, parent_agent,
        scope_work_dir, options, inherit, settings, workspace, timeout_explicit, siblings,
        machine, sea_command, reserving,
    )
    if job is not None:
        # Reserved from here until the job finishes: a later call of
        # the caller sees it in :func:`reserved_budget` before this
        # sub-task's spend reaches the caller's ledger.
        job.budget = kwargs["max_budget"]
    parent_task_id = kwargs["parent_task_id"]
    try:
        result = daemon_client.run(
            prompt,
            **kwargs,
            # The caller's worktrees, read at send time: a table that
            # cannot be read fails the send like any other send error.
            # Paths of this machine: meaningless to another one.
            isolated_repos=None if machine else _isolated_repos_of(parent_agent),
            # No deadline on this wait: the call's bound is enforced by
            # the caller joining the job thread; the daemon only records
            # it.  The cancel event is the one way to stop the sub-task.
            timeout=None,
            endpoint_file=_daemon_endpoint_file(),
            cancel=None if job is None else job.cancel,
            running=None if job is None else job.running,
        )
    except daemon_client.CancelledError as e:
        # A confirmed stop carries the stopped task's spend, which still
        # counts towards the caller.
        _attribute_dispatch_usage(parent_agent, e.result, epoch, parent_task_id)
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
        _attribute_dispatch_usage(
            parent_agent, getattr(e, "task_result", None), epoch, parent_task_id,
        )
        logger.warning("agent dispatch failed", exc_info=True)
        return f"Error: the {name} agent task could not run: {e}"
    _attribute_dispatch_usage(parent_agent, result, epoch, parent_task_id)
    return result


def dispatch_command(
    prompt: str,
    sea_path: str,
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
    timeout_explicit: bool = False,
    siblings: int = 1,
    machine: str = "",
    sea_command: str = "",
    reserving: Sequence[AgentJob] = (),
) -> dict[str, Any]:
    """Build the keyword arguments of the ``run`` :func:`dispatch_result` sends.

    The dispatcher's side of a sub-task's settings, as one pure step
    (it reads *parent_agent* and writes nothing: the sub-task's
    directory is created by :func:`dispatch_result` before this call):
    the explicit values of the call, what the empty ones inherit from
    *parent_agent* (:func:`inherit_from_parent`), the ``provenance``
    marks that tell the daemon which values were explicit, and the
    calling task's identity.  :func:`dispatch_result` passes the
    result to :func:`kiss.agents.sorcar.daemon_client.run`, which
    encodes it as the wire command (:func:`~kiss.agents.sorcar.daemon_client.run_command`);
    a check of the dispatcher against the daemon's own evaluation
    drives these two functions and ``apply_sea`` on the command they
    produce.  The arguments are those of :func:`dispatch_result`, with
    *work_dir* already remapped and created.  With *machine* (a run
    on a remote machine) what cannot cross machines is not inherited:
    the caller's chat (its chat id names nothing there), its live
    container and its extra tools; the call's own ``chat_id`` and
    ``docker_image`` options still apply, as values on the machine.

    Returns:
        The keyword arguments of :func:`~kiss.agents.sorcar.daemon_client.run`
        except the wait's own (``timeout``, ``endpoint_file``,
        ``cancel``, ``running``) and ``isolated_repos``, which
        :func:`dispatch_result` reads off the caller at send time;
        ``record_timeout`` is *timeout*.
    """
    # The calling task's identity, threaded through the daemon so the
    # dispatched run is a SUB-AGENT of that task: a nested sub-agent
    # tab under the caller's tab via the run's own ``new_tab``
    # broadcast, a history row nested under the calling task, and a
    # ``subagentDone`` when it ends, instead of a top-level tab.
    # Standalone use (no calling agent, or one that has not persisted
    # a task row) dispatches an ordinary top-level task, unchanged.
    # Any child a reviewer dispatches carries the reviewer marker (see
    # kiss.agents.sorcar.fanout_guard) so it gets the same read-only
    # tool profile.
    from kiss.agents.sorcar.sorcar_agent import _persisted_task_id, subagent_parent_tab_id_of

    _is_rev = getattr(parent_agent, "_is_reviewer_subagent", None)
    parent_reviewer = bool(_is_rev()) if callable(_is_rev) else False
    parent_task_id = _persisted_task_id(parent_agent)
    # The tab the chat pages show the caller under (its own tab id, or —
    # when the caller is itself a sub-agent — its ``{parent}__sub_{task}``
    # tab).  A caller that has not persisted a row yet (before its first
    # run) dispatches an ordinary top-level task, so its tab id stays out.
    parent_tab_id = subagent_parent_tab_id_of(parent_agent) if parent_task_id else ""
    # The arguments the caller left empty come from the calling agent
    # (its model, budget share, chat, web/memory settings, container).
    # Channel dispatches and explicit programmatic callers skip this.
    explicit = list(explicit_values(model_name, budget, options))
    if timeout_explicit:
        explicit.append("timeout")
    inherited = inherit_from_parent(
        parent_agent if inherit else None, model_name, budget, options,
        # The script's model applies only when the call names none.
        script_picks_model="model" in (settings or {}) and not model_name,
        siblings=siblings, reserving=reserving,
    )
    if machine:
        inherited = _without_cross_machine_inheritance(inherited, options)
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
    return {
        "sea_path": sea_path,
        "work_dir": work_dir,
        "scope_work_dir": scope_work_dir,
        "parent_task_id": parent_task_id,
        "parent_tab_id": parent_tab_id,
        "parent_reviewer": parent_reviewer or is_review_task(prompt),
        # A cron run's sub-tasks are cron rows too (an in-chat job's
        # children carry no unattended text, so the flag travels).
        "cron_job": cron_agent.is_cron_run(parent_agent),
        "model": model_name,
        "chat_id": options.chat_id,
        "system_prompt": options.system_prompt,
        "auto_classify": options.auto_classify,
        "max_budget": budget,
        "model_config": options.model_config,
        "use_web_tools": options.use_web_tools,
        "use_memory": options.use_memory,
        # The Python API's ``is_parallel`` (a sequential caller does
        # not hand ``run_parallel`` to its children); not a setting.
        "is_parallel": bool(getattr(parent_agent, "_is_parallel", True)) if inherit else True,
        "add_to_system_prompt": options.add_to_system_prompt,
        "add_to_prompt": options.add_to_prompt,
        "tool_profile": options.tool_profile,
        "docker_image": inherited.docker_image,
        "workspace": workspace,
        # The caller's extra tools (its SEA's ``tools()``)
        # cannot travel the wire: the daemon takes them off the
        # running caller, which ``parent_task_id`` names — not on
        # another machine, where the caller does not run.
        "inherit_tools": inherit and not machine,
        "provenance": inherited.provenance(explicit),
        "record_timeout": timeout,
        "machine": machine,
        "sea_command": sea_command,
    }


def _without_cross_machine_inheritance(inherited: Inherited, asked: RunOptions) -> Inherited:
    """Return *inherited* without what a run on another machine cannot take from its caller.

    The caller's chat id and ``container:<id>`` name nothing on the
    machine: the call's own ``chat_id`` and ``docker_image`` options
    (*asked*) stand instead, and the two keys leave the inherited
    fields of the run's ``provenance``.
    """
    return dataclasses.replace(
        inherited,
        options=dataclasses.replace(inherited.options, chat_id=asked.chat_id),
        docker_image=asked.docker_image,
        fields=tuple(f for f in inherited.fields if f not in ("chat_id", "docker_image")),
    )


def start_run_agent(
    parent_work_dir: str,
    task: str,
    agent: str = "",
    model: str = "",
    tool_profile: str = "",
    max_budget: str = "",
    timeout: str = "",
    options: str = "",
    parent_agent: Any = None,
    siblings: int = 1,
    machine: str = "",
) -> AgentJob | str:
    """Resolve, check and start a SEA sub-task as an :class:`AgentJob`.

    The body behind ``run_agent`` (:func:`_run_agent`):
    :func:`plan_run_agent` resolves and checks the call, then the
    dispatch runs on a job thread (``run_parallel`` plans once and
    starts one such job per task, see :func:`run_agents_parallel`).
    The arguments are those of :func:`plan_run_agent`.

    Returns:
        The started job, or :func:`plan_run_agent`'s error message.
    """
    plan = plan_run_agent(
        parent_work_dir, task, agent, model, tool_profile, max_budget, timeout, options,
        parent_agent, siblings, machine,
    )
    if isinstance(plan, str):
        return plan
    return start_agent_job(plan["name"], plan, parent_agent)


def plan_run_agent(
    parent_work_dir: str,
    task: str,
    agent: str = "",
    model: str = "",
    tool_profile: str = "",
    max_budget: str = "",
    timeout: str = "",
    options: str = "",
    parent_agent: Any = None,
    siblings: int = 1,
    machine: str = "",
) -> dict[str, Any] | str:
    """Resolve and check a SEA sub-task call; return :func:`dispatch_result`'s arguments.

    The dispatcher's checks before anything runs: parse the options
    (:func:`parse_run_options`), resolve the script
    (:func:`resolve_agent`), load it for its class and settings, refuse
    an explicit value that contradicts a locked key
    (:func:`~kiss.agents.sorcar.sea_settings.locked_conflicts`), pick
    the work directory, and decide whether the sub-task inherits from
    the caller (not for a channel or an ``inherit: false`` call).  The
    plan's ``timeout`` is the seconds the caller should block before
    returning the job's notice (:func:`resolve_timeout`).

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
            empty for the calling agent's remaining budget shared among
            *siblings* sub-tasks plus the caller (a non-inheriting
            sub-task: the daemon default).
        timeout: Seconds the call blocks, as a number string; empty
            for the script's ``timeout`` setting, else
            :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`.
        options: JSON object of further run settings (see
            :func:`parse_run_options`).
        parent_agent: The agent calling the tool, when there is one;
            the sub-task's spend is folded into its task accounting
            (see :func:`_attribute_dispatch_usage`).
        siblings: How many sub-tasks the call starts at once (the
            budget is shared among them, see
            :meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent._subagent_budget_share`,
            after what the caller's running sub-tasks already hold is
            taken off: ``reserving`` in the plan, :func:`live_agent_jobs`
            read here, once per call).
        machine: The name or host/IP of a registered remote machine
            (:func:`~kiss.agents.sorcar.machines.find_machine`) the
            sub-task runs on; empty, or this machine's own name or
            address (:func:`~kiss.agents.sorcar.machines.is_local_machine`),
            runs it here as an ordinary call.  The SEA is still
            resolved and loaded here (its settings are checked here)
            but runs as the machine's copy (:func:`remote_sea`): a
            registered command by its name (the run's
            ``sea_command``), the default agent as the machine's plain
            run, a ``.py`` path at the same path there.  The
            ``work_dir`` option is a path on the machine,
            as given (empty: the machine's configured folder, or what
            its copy of the SEA sets); the SEA's ``work_dir`` setting
            read here is a path here and is not sent; the task's paths
            are not rewritten.

    Returns:
        The keyword arguments of :func:`dispatch_result` plus
        ``name`` and ``alias``, or an error message (an empty task, bad
        options, an unknown agent or machine, a broken SEA, or a
        locked key the call contradicts).
    """
    if not task.strip():
        return EMPTY_TASK_ERROR
    try:
        run_options = parse_run_options(options, tool_profile, model, max_budget, timeout)
    except ValueError as e:
        return f"Error: {e}"
    machine_name = ""
    if machine.strip() and not is_local_machine(machine):
        found = find_machine(machine)
        if found is None:
            return _unknown_machine_error(machine)
        machine_name = found.name
    resolved = resolve_agent(agent, parent_work_dir)
    if isinstance(resolved, str):
        return resolved
    sea_path, name = resolved
    from kiss.agents.sorcar.sea_commands import base_settings, is_channel, load_sea

    try:
        sea = load_sea(Path(sea_path))
        settings = base_settings([sea])
    except SeaError as e:
        return f"Error: {e}"
    seconds = resolve_timeout(run_options.timeout, settings)
    # sea_settings.PRECEDENCE_RULE: an explicit argument or option wins
    # over the SEA's ``settings()``, which win over what the calling
    # task passes on; a key the SEA locks may not be replaced.  A
    # channel (a ``ChannelSea``: a channel agent, cron) takes nothing
    # from the calling task — not its chat, model, budget share,
    # container or prompt suffixes — and runs in its own ``work_dir``
    # (CHANNEL_BEHAVIOURS "no inheritance", "scratch directory"); so
    # does a call with ``inherit: false``.  Every other SEA is a
    # sub-agent on the caller's project (or on the ``work_dir`` option
    # or setting, a relative one under the caller's directory): the
    # arguments left empty are inherited from the calling agent (see
    # ``inherit_from_parent``), and the task's references to the main
    # checkout are rewritten to the caller's worktree.
    asked = explicit_values(run_options.model, run_options.max_budget, run_options)
    conflict = locked_conflicts(settings, asked, parent_work_dir)
    if conflict:
        return f"Error: {name}: {conflict}"
    channel = is_channel([sea])
    if channel and run_options.inherit:
        return f"Error: {name}: a channel never inherits from the calling task"
    if run_options.workspace and not channel:
        return (
            f"Error: {name}: options['workspace'] applies to a channel only; "
            f"{name} is not a channel (its class does not derive from ChannelSea)"
        )
    inherit = not channel and run_options.inherit is not False
    base_dir = parent_work_dir or str(kiss_home() / "agent_work")
    sea_command = ""
    if machine_name:
        # The call's own option is a path on the machine; the SEA's
        # ``work_dir`` setting is a path here, and the machine's copy of
        # the SEA applies its own there.
        work_dir = run_options.work_dir
        sea_path, sea_command = remote_sea(sea_path)
    else:
        chosen = run_options.work_dir or str(settings.get("work_dir") or "")
        work_dir = anchored_work_dir(chosen, base_dir) if chosen else base_dir
    kwargs: dict[str, Any] = {
        "name": name, "prompt": sub_task_prompt(task, parent_work_dir, inherit, machine_name),
        "sea_path": sea_path, "work_dir": work_dir,
        "model_name": run_options.model, "budget": run_options.max_budget, "timeout": seconds,
        "parent_agent": parent_agent, "scope_work_dir": parent_work_dir,
        "options": run_options, "inherit": inherit, "settings": settings,
        "workspace": run_options.workspace, "timeout_explicit": run_options.timeout is not None,
        "siblings": siblings, "machine": machine_name, "sea_command": sea_command,
        "reserving": live_agent_jobs(parent_agent),
        "alias": command_alias(resolved[0]) if is_sea_path(agent.strip()) else "",
    }
    return kwargs


def sub_task_prompt(task: str, parent_work_dir: str, inherit: bool, machine: str) -> str:
    """Return the prompt a sub-task planned by :func:`plan_run_agent` gets for *task*.

    The one part of a plan that depends on the task text: a sub-task
    that inherits from the caller and runs on this daemon gets the
    caller's main-repository paths rewritten to its worktree
    (:func:`rewrite_parent_repo_paths`, when
    ``dispatch_path_rewrite`` is on); a channel, an ``inherit: false``
    call or a run on another *machine* (whose paths mean nothing
    here) gets *task* as given.

    Args:
        task: The task text.
        parent_work_dir: Work directory of the calling task.
        inherit: The plan's ``inherit`` flag.
        machine: The plan's ``machine`` name; empty for this daemon.
    """
    if inherit and not machine and DEFAULT_CONFIG.dispatch_path_rewrite:
        return rewrite_parent_repo_paths(task, parent_work_dir)
    return task


def remote_sea(sea_path: str) -> tuple[str, str]:
    """Return how a run on another machine names the SEA at *sea_path*.

    The daemon of a remote machine has its own copy of every bundled
    and registered SEA, under its own paths: the default agent
    (:data:`DEFAULT_AGENT_PATH`, which pins nothing) is sent as no SEA
    at all, a SEA that is a registered command by that name (the
    ``seaCommand`` of the run, which the remote daemon resolves to its
    own copy and refuses when it has none), and any other path as it
    is (the file must exist there).

    Returns:
        ``(sea_path, sea_command)`` to send; at most one is non-empty.
    """
    if sea_path == DEFAULT_AGENT_PATH:
        return "", ""
    alias = command_alias(sea_path)
    if alias:
        return "", alias
    return sea_path, ""


def _unknown_machine_error(machine: str) -> str:
    """Return the ``run_agent`` error for a ``machine`` that is not registered."""
    names = ", ".join(m.name for m in load_machines()) or "none"
    return (
        f"Error: {machine.strip()!r} is not a registered machine (registered: {names}); "
        "register it with /machine add <name> user@host or /machine add <name> "
        "https://host:port <password>, or pass a registered name or host."
    )


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
    machine: str = "",
) -> str:
    """Run a SEA on a task immediately.

    The implementation behind the per-task ``run_agent`` tool built by
    :func:`make_run_agent_tool`, which captures *parent_work_dir*; the
    remaining arguments are the tool's (see its docstring).  Starts
    the job (:func:`start_run_agent`) and blocks for its ``timeout``
    unless *wait* is ``"false"``.

    Returns:
        The sub-task's YAML result ("success" and "summary" keys), the
        job notice (``wait="false"``, or the ``timeout`` expired), or
        an error message.
    """
    try:
        blocking = _parse_bool("wait", wait) is not False
    except ValueError as e:
        return f"Error: {e}"
    job = start_run_agent(
        parent_work_dir, task, agent, model, tool_profile, max_budget, timeout, options,
        parent_agent, machine=machine,
    )
    if isinstance(job, str):
        return job
    try:
        if blocking:
            join_agent_job(job, job.timeout)
        else:
            wait_until_started(job)
            if not job.finished:
                return job_notice(job, None)
            # The dispatch already failed (no daemon, a refused run):
            # there is no tab and nothing to wait for, so the error is
            # the answer, as for a blocking call.
    except BaseException:
        _abandon_job(job)
        raise
    return _collect_job(job)


def _abandon_job(job: AgentJob) -> None:
    """Cancel *job* when the call that started it was interrupted.

    The tool call's Stop button, or the calling task stopping, whether
    while the sub-task was starting or while it ran: the sub-task is
    cancelled too, as a blocking call's sub-task always has been.  Not
    joined here, so the Stop stays prompt: the job thread stops the
    sub-task on its own and the run's end (:func:`kill_jobs_of`)
    collects what is left.  A job without an owner has no run end, so
    it is dropped here.
    """
    job.cancel.set()
    if job.owner is None:
        forget_agent_job(job)


def _collect_job(job: AgentJob) -> str:
    """Return *job*'s result text, dropping it from the registry, or its notice if unfinished."""
    if job.finished:
        forget_agent_job(job)
        return job.result
    return job_notice(job, job.timeout)


def run_agents_parallel(
    parent_work_dir: str,
    tasks: list[str],
    agent: str = "",
    model: str = "",
    tool_profile: str = "",
    max_budget: str = "",
    timeout: str = "",
    options: str = "",
    parent_agent: Any = None,
) -> str:
    """Run *tasks* as N ``run_agent`` sub-tasks at once and wait for them together.

    The implementation behind the ``run_parallel`` tool
    (:func:`make_run_parallel_tool`): one :func:`plan_run_agent` for
    the call (the agent and arguments are the same for every task, so
    the SEA is resolved, loaded and checked once) and one job per task
    from that plan — so a child is exactly what ``run_agent`` would
    start, with the caller's remaining budget shared among the N
    children plus the caller — then one wait of the call's ``timeout``
    for all of them.  A child still running when the wait ends is not
    stopped: its entry is the job notice, and ``agent_job`` collects
    or kills it.

    Args:
        parent_work_dir: Work directory of the calling task (see
            :func:`start_run_agent`).
        tasks: The non-empty task texts (:func:`~kiss.agents.sorcar.fanout_guard.parse_tasks_json`).
        agent: The SEA every child runs as.
        model: LLM model of the children; empty inherits the caller's.
        tool_profile: Tool profile of the children; empty lets the
            daemon choose.
        max_budget: Per-child USD budget as a number string; empty
            shares the caller's remaining budget.
        timeout: Seconds the call blocks, as a number string; empty
            takes the SEA's ``timeout`` setting, else 3600.
        options: JSON object of further run settings.
        parent_agent: The agent calling the tool.

    Returns:
        A YAML list with one entry per task, in order: the child's
        result text (``ran``, ``success``, ``summary``) or its job
        notice; or the error message alone when the children cannot
        start (the checks are the same for every task, so nothing is
        started then).
    """
    started: list[AgentJob] = []
    plan: dict[str, Any] | None = None
    try:
        for task in tasks:
            if plan is None:
                planned = plan_run_agent(
                    parent_work_dir, task, agent, model, tool_profile, max_budget, timeout,
                    options, parent_agent, siblings=len(tasks),
                )
                if isinstance(planned, str):
                    return planned
                plan = kwargs = planned  # the plan's prompt is this first task's
            else:
                prompt = sub_task_prompt(task, parent_work_dir, plan["inherit"], plan["machine"])
                kwargs = {**plan, "prompt": prompt}
            started.append(start_agent_job(plan["name"], kwargs, parent_agent))
        deadline = time.monotonic() + started[0].timeout
        for job in started:
            join_agent_job(job, max(0.0, deadline - time.monotonic()))
    except BaseException:
        for job in started:
            _abandon_job(job)
        raise
    return str(yaml.safe_dump([_collect_job(job) for job in started], sort_keys=False))


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
            (its initial ``status running=true``) or, after ``result``
            is recorded, when the dispatch ended without one.
        thread: The thread running :func:`dispatch_result`, assigned
            (already started) by :func:`start_agent_job` after the job
            exists, since the thread's arguments name the job.
        started: ``time.monotonic()`` when the thread was started.
        timeout: The seconds the starting call blocks for the result
            (:func:`resolve_timeout`), shown on the ``ran`` line.
        workspace: The channel workspace the sub-task holds while it
            runs (channel SEAs), else ``""``.
        budget: The USD budget the sub-task was given (the call's own
            or its inherited share), recorded by :func:`dispatch_result`
            once resolved and before the sub-task runs; ``None`` until
            then or when the daemon default applies.  While the job is
            unfinished it is reserved against the owner's remainder
            (:func:`reserved_budget`); once finished the sub-task's
            real spend is on the owner's ledger instead.
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
    thread: threading.Thread = dataclasses.field(init=False)
    started: float = 0.0
    timeout: float = DEFAULT_DISPATCH_TIMEOUT_SECONDS
    workspace: str = ""
    budget: float | None = None
    outcome: TaskResult | str | None = None
    result: str = ""

    @property
    def finished(self) -> bool:
        """Whether the dispatch has recorded its result.

        The result is written before the thread's own ``running.set()``
        and before the thread exits, so this — not the thread's
        liveness — is what ``run_agent`` and ``agent_job`` report on: a
        caller woken by the event without a tab finds the result, and
        one that polls never sees a finished job as still running.
        """
        return bool(self.result)


_AGENT_JOBS: dict[str, AgentJob] = {}
"""Every unresolved sub-task dispatched in this process, by job id.

A job leaves the registry when its blocking call takes its result or
when its owner's run ends (:func:`kill_jobs_of`).  Guarded by
:data:`_AGENT_JOBS_LOCK`.
"""

_AGENT_JOBS_LOCK = threading.RLock()


def start_agent_job(name: str, kwargs: dict[str, Any], owner: Any) -> AgentJob:
    """Start :func:`dispatch_result` with *kwargs* in a thread and register the job.

    The job is registered before its thread starts, so a Stop that
    interrupts this call never leaves a running sub-task that
    :func:`kill_jobs_of` cannot find.  The function does
    not wait for the sub-task to start: a blocking ``run_agent`` call
    joins the thread under its one ``timeout`` (startup included) and
    a ``wait="false"`` call waits for the tab with
    :func:`wait_until_started`.

    Args:
        name: The agent's display name.
        kwargs: The keyword arguments of :func:`dispatch_result`, plus
            the ``alias`` :func:`format_dispatch_result` shows; the
            job itself is added as ``job``.
        owner: The calling agent (``None`` for standalone use).

    Returns:
        The registered job.
    """
    job_id = f"agent-{uuid.uuid4().hex[:8]}"
    job = AgentJob(
        job_id, name, owner, threading.Event(), threading.Event(),
        timeout=float(kwargs["timeout"]), workspace=str(kwargs.get("workspace") or ""),
    )
    job.thread = threading.Thread(
        target=_finish_agent_job, args=(job, {**kwargs, "job": job}),
        name=f"agent-job-{job_id}", daemon=True,
    )
    job.started = time.monotonic()
    # Registered first, started under the same lock: a task's Stop is a
    # KeyboardInterrupt injected between any two bytecodes, and a job
    # it found started but not yet registered would escape
    # kill_jobs_of and outlive its caller.  Thread.start() itself is
    # not atomic either: the native thread exists before the parent
    # observes it alive, so a Stop landing inside it leaves a job that
    # looks unstarted yet runs later.  kill_jobs_of therefore cancels
    # every registered job, and _finish_agent_job dispatches nothing
    # once its cancel is set.
    locked_call(_AGENT_JOBS_LOCK, _register_and_start, job)
    return job


def _register_and_start(job: AgentJob) -> None:
    """Enter *job* in the registry and start its thread; caller holds ``_AGENT_JOBS_LOCK``."""
    _AGENT_JOBS[job.job_id] = job
    job.thread.start()


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
        if job.cancel.is_set():
            # Killed before this thread ran (its owner's run ended while
            # Thread.start() was still in flight): nothing is dispatched.
            job.outcome = f"Error: the {job.name} agent task was stopped before it started."
            job.result = job.outcome
            return
        job.outcome = dispatch_result(**kwargs)
        job.result = format_dispatch_result(
            job.outcome, kwargs["timeout"], alias, kwargs.get("machine", ""),
        )
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
    joins the thread for :data:`_JOB_KILL_GRACE_SECONDS` through
    :func:`join_agent_job`, so a Stop of the killing call (or of its
    task) lands during the wait and propagates with the cancel already
    sent.  A finished job is left as it is.

    Returns:
        The job's result text (the stop error, or the result it had
        already produced), or a still-running notice when the daemon
        did not answer within the grace.
    """
    job.cancel.set()
    join_agent_job(job, _JOB_KILL_GRACE_SECONDS)
    if not job.finished:
        return f"Job {job.job_id} ({job.name} agent task) is still running."
    return job.result


def forget_agent_job(job: AgentJob) -> None:
    """Drop *job* from the registry (its result has been taken)."""
    locked_call(_AGENT_JOBS_LOCK, _AGENT_JOBS.pop, job.job_id, None)


@locked(_AGENT_JOBS_LOCK)
def agent_jobs_of(owner: Any) -> dict[str, AgentJob]:
    """Return the registered jobs *owner*'s ``run_agent`` started, by job id."""
    return {job_id: job for job_id, job in _AGENT_JOBS.items() if job.owner is owner}


def live_agent_jobs(owner: Any) -> list[AgentJob]:
    """Return *owner*'s jobs whose sub-task is still running (:attr:`AgentJob.finished` unset)."""
    return [job for job in agent_jobs_of(owner).values() if not job.finished]


def reserved_budget(jobs: Sequence[AgentJob]) -> float:
    """Return the USD the unfinished ones of *jobs* were given (:attr:`AgentJob.budget`).

    *jobs* are the caller's live jobs :func:`plan_run_agent` read once
    per ``run_agent`` / ``run_parallel`` call, on the calling thread:
    the sub-tasks that call starts are not registered yet, so siblings
    of one call never subtract each other, while a job of an earlier
    call has recorded its budget before its ``running`` event (and so
    before the earlier call returned).  Summed again when a job
    thread resolves its budget (:func:`_inherit_from_parent`), so a
    sub-task that finished in between — its spend is on the caller's
    ledger by then — no longer holds anything.
    """
    return sum(job.budget or 0.0 for job in jobs if not job.finished)


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
    own thread.  Every job is cancelled, finished or not: a job whose
    start was interrupted may have a thread that is not yet observably
    alive but runs later, and it then sees its cancel and dispatches
    nothing (:func:`_finish_agent_job`).

    Returns:
        The ids of the jobs that had not finished.
    """
    jobs = agent_jobs_of(owner)
    # Registry and cancels first: they must survive an interrupt that
    # lands in the bounded join below.
    for job in jobs.values():
        forget_agent_job(job)
        job.cancel.set()
    live = [job for job in jobs.values() if not job.finished]
    deadline = time.monotonic() + _JOB_END_GRACE_SECONDS
    for job in live:
        # A thread not yet observably started cannot be joined; it
        # exits on its own as soon as it runs.
        if job.thread.is_alive():
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
        if not job.finished:
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
``worker`` is not one of them: it names a base class (a tool-bound
run), so ``agent="worker"`` gets the usual "no such command" error
instead of silently running a session; nor are the
:data:`_REVIEWER_NAMES`, which ask for a toolset, not an agent."""

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
    from kiss.agents.sorcar.daemon_client import resolve_sea_path

    requested = agent.strip()
    squashed = _squash(requested)
    if not requested or squashed in _GENERIC_AGENT_NAMES:
        requested = DEFAULT_AGENT_PATH
    if is_sea_path(requested):
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute() and parent_work_dir:
            candidate = Path(parent_work_dir) / candidate
        try:
            sea_path = resolve_sea_path(str(candidate))
        except ValueError as e:
            return f"Error: {e}"
        return sea_path, script_name(sea_path)
    commands = sea_commands.list_commands()
    for name in commands:
        if _squash(name) == squashed:
            command_path = sea_commands.get_command(name)
            if command_path is not None:
                return str(command_path), name
    return _unknown_agent_error(agent, squashed, commands)


def is_sea_path(agent: str) -> bool:
    """Return whether a ``run_agent`` ``agent`` argument is a path (rule 2 of ``resolve_agent``)."""
    return agent.endswith(".py") or "/" in agent or "\\" in agent


def command_alias(sea_path: str) -> str:
    """Return the registered command name whose script is *sea_path*, or ``""``.

    Lets a call that reached a bundled or ``SEAS.md`` SEA by its path
    learn the name it could have used instead
    (``run_agent(agent="review_paper", ...)``).
    """
    from kiss.agents.sorcar import sea_commands

    for name in sea_commands.list_commands():
        if str(sea_commands.get_command(name)) == sea_path:
            return name
    return ""


def resolve_timeout(timeout: float | None, settings: dict[str, Any]) -> float:
    """Resolve how long a ``run_agent`` call waits for its sub-task, in seconds.

    The call's ``timeout`` wins; ``None`` takes the SEA's own
    ``timeout`` setting (``settings()["timeout"]``, see
    :mod:`kiss.agents.sorcar.sea_settings`), else
    :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`.

    Args:
        timeout: The call's parsed ``timeout`` argument
            (:attr:`RunOptions.timeout`), ``None`` when not passed.
        settings: The resolved settings of the script the sub-task
            runs (:func:`~kiss.agents.sorcar.sea_settings.resolve_settings`
            has already refused a non-positive or non-numeric
            ``timeout``).

    Returns:
        The seconds to wait.
    """
    if timeout is not None:
        return timeout
    return float(settings.get("timeout") or DEFAULT_DISPATCH_TIMEOUT_SECONDS)


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


def current_work_dir(parent_agent: Any, work_dir: str) -> str:
    """The calling task's work directory at call time.

    A task's work directory moves when it calls ``create_worktree``,
    so a dispatch reads it from the agent rather than the value the
    tool was built with.

    Args:
        parent_agent: The agent the tool was built for, or ``None``.
        work_dir: The work directory the tool was built with.

    Returns:
        ``parent_agent.work_dir`` when set, else *work_dir*.
    """
    current = getattr(parent_agent, "work_dir", None) if parent_agent is not None else None
    return str(current) if current else work_dir


def _isolated_repos_of(parent_agent: Any) -> dict[str, str]:
    """Repository root -> worktree directory of the calling agent's worktrees.

    Read at call time like :func:`current_work_dir`: the set grows as
    the task calls ``create_worktree`` for repositories it discovers.
    A sub-task edits these worktrees instead of isolating the same
    repositories again (:meth:`SorcarAgent.inherit_worktrees`).

    Args:
        parent_agent: The agent the tool was built for, or ``None``.

    Returns:
        String paths for the wire; ``{}`` for no agent or a non-worktree one.
    """
    isolated = getattr(parent_agent, "isolated_repos", None)
    if not callable(isolated):
        return {}
    repos = isolated()
    if not isinstance(repos, dict):
        return {}
    return {str(repo): str(wt_dir) for repo, wt_dir in repos.items()}


def make_run_agent_tool(
    work_dir: str, parent_agent: Any = None,
) -> Callable[..., str]:
    """Build the ``run_agent`` tool for the agent running in *work_dir*.

    The tool executes in the daemon process, whose own working
    directory is unrelated to the user's project, so the calling
    task's work directory must be captured here (exactly like
    ``make_skill_tool``): it anchors relative SEA paths and
    is the work directory the dispatched path-mode sub-task runs in.
    When *parent_agent* is given its ``work_dir`` at call time wins
    (see :func:`current_work_dir`).

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
        machine: str = "",
        description: str = "",
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
        of its remaining budget, chat, prompt suffixes, extra tools and
        container.  A channel or cron sub-task inherits none of these
        and runs in the Sorcar
        home's ``channel_work`` directory.  The call blocks until the
        task finishes or ``timeout`` expires; at ``timeout`` the task
        keeps running as an ``agent_job`` and the call returns its job
        id (``agent_job(id, "wait")`` collects the result, ``"kill"``
        stops it; a job still running when this task ends is killed).
        ``run_parallel`` is this call made once per task.

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
                keys are the SEA settings vocabulary except ``model``,
                ``tool_profile``, ``max_budget`` and ``timeout``, which
                are the arguments above and are refused here:
                ``work_dir`` (relative to this task's, as is a SEA's
                own ``work_dir`` setting),
                ``chat_id``,
                ``workspace`` (the account of a multi-account channel),
                ``add_to_system_prompt`` / ``add_to_prompt`` (appended
                text), ``model_config`` (JSON object), ``docker_image``,
                and the booleans ``inherit`` (``false`` = take nothing
                from this task), ``auto_classify``, ``use_web_tools``,
                ``use_memory``.
            wait: ``"false"`` = return a job id at once; then
                ``agent_job(job_id, "wait")`` returns the result,
                ``"tail"`` its status, ``"kill"`` stops it.  Use it to
                run several agents at once or to keep working while a
                long sub-task runs; wait for or kill every job before
                finishing.  Empty (default) blocks until done.
            machine: The name or host/IP address of a registered
                remote machine (``/machine add``) to run the agent on;
                empty (default), or this machine's own name or address
                (``localhost``, its hostname or IP), = this machine, an
                ordinary call.  Otherwise the sub-task runs
                on that machine's daemon, with its tools, files and
                models: ``work_dir`` is a path there (empty = its
                configured folder), a command runs its copy of the
                SEA, a ``.py`` path must exist at the same path there,
                and the task's paths are not rewritten.  It still
                inherits this task's model, budget share and prompt
                suffixes, but not its chat, container or extra tools.
                Its tab opens under this task's on every surface like
                a local sub-agent's.
            description: A brief description of the sub-task, shown as
                the title of this call in the chat.

        Returns:
            The sub-task's YAML result: a ``ran`` line first (the SEA
            it ran as, its model, tool profile, budget and
            timeout, which values it inherited from this task, which
            inherited or default values the SEA pinned to its own, e.g.
            ``pinned=use_web_tools(True->False)``; a SEA reached by
            path that is also a command ends with
            ``(also agent="name")``; a run on a remote machine with
            ``machine=<name>``),
            then ``success`` and ``summary``; the job notice
            (``wait="false"``, or ``timeout`` expired with the task
            still running); or an error message (unknown agent — naming
            the closest command — unknown machine — naming the
            registered ones — or a locked key the call contradicts).
        """
        del description
        return _run_agent(
            current_work_dir(parent_agent, work_dir), task, agent, model, tool_profile,
            max_budget, timeout, options, parent_agent, wait, machine,
        )

    run_agent.__doc__ = (
        (run_agent.__doc__ or "")
        .replace("{channels}", ", ".join(available_channels()) or "none installed")
        .replace("{precedence}", PRECEDENCE_RULE)
    )
    run_agent.unknown_arguments_hint = options_keyword_hint  # type: ignore[attr-defined]
    return run_agent


def make_run_parallel_tool(
    work_dir: str, parent_agent: Any = None,
) -> Callable[..., str]:
    """Build the ``run_parallel`` tool for the agent running in *work_dir*.

    ``run_parallel`` is :func:`make_run_agent_tool`'s ``run_agent``
    made once per task (:func:`run_agents_parallel`): same agent
    resolution, same inheritance, same precedence, same ``timeout``.

    Args:
        work_dir: The calling task's work directory.
        parent_agent: The agent this tool is built for, when there is
            one (see :func:`make_run_agent_tool`).

    Returns:
        The ``run_parallel`` tool callable.
    """

    def run_parallel(
        tasks: str, agent: str = "", model: str = "", tool_profile: str = "",
        max_budget: str = "", timeout: str = "", options: str = "", description: str = "",
    ) -> str:
        """Run multiple independent tasks concurrently using parallel agents.

        One ``run_agent`` call per task, started together and waited
        for together: each child is a sub-task in its own tab with the
        same agent, model, tool profile and options, inheriting this
        task's chat, prompt suffixes, extra tools and container
        exactly as a ``run_agent`` sub-task does, with this task's
        remaining budget shared among
        the children (and this task).  {precedence}

        **When to call run_parallel:**
        - Multi-source / multi-topic research ("research these 5
          companies", "summarize each of these N PDFs").
        - Codebase exploration across unrelated modules ("look at the
          frontend, backend, db layer, and auth in parallel").
        - Multi-perspective review of one artifact (correctness
          reviewer + security reviewer + style reviewer +
          architecture reviewer, each looking at the same diff with
          a different lens).
        - Generating N alternative candidates for the same problem
          so the orchestrator can pick the best.
        - Independent test suites or validations on disjoint targets.
        - Bulk file generation when each file is independent and the
          API contract between them is already pinned down in a
          spec.

        **When NOT to call run_parallel:** when each task is just a
        shell command whose output you need (test splits, builds,
        lints).  Use ``run_commands_parallel`` for those: it runs the
        commands concurrently without spawning LLM sub-agents.

        **Hard limits (enforced, not advisory):**
        - ``tasks`` must be a literal JSON array; shell substitutions
          such as ``"$(cat tasks.json)"`` are not expanded and are
          rejected.

        Args:
            tasks: A JSON-encoded list of task description strings,
                e.g. ``'["Read src/foo.py and summarize its purpose",
                "Read src/bar.py and summarize its purpose", "Find the
                current weather in San Francisco"]'``.
            agent: The SEA every child runs as, exactly as
                ``run_agent``'s ``agent``: empty (default) = a plain
                Sorcar sub-agent; a slash-command name
                (``"write_paper"``, ``"slack"``) = that command; a
                ``.py`` path (relative to this task's work directory)
                = that SEA file; its ``prompt(task)`` shapes each
                child's prompt.
            model: LLM model for the sub-agents (e.g. a cheaper or a
                different reviewer model).  Empty (default) uses this
                task's model.  Prefer this over asking the sub-agent
                to call ``set_model`` itself, which costs a whole step
                on the wrong model.
            tool_profile: ``"review"`` gives the sub-agents the
                read-only toolset (Bash, bash_job, Read,
                run_commands_parallel, memory reads, browser tools,
                talk, decide, summary);
                ``"shell"`` just Bash, bash_job, Read and
                run_commands_parallel; ``"assistant"`` the shell set
                plus ask_user_question, talk, decide, summary and
                set_model.  Tool groups ``"edit"`` (Edit, Write),
                ``"browser"``, ``"memory"``, ``"agents"``, ``"mcp"``,
                ``"skills"``, ``"user"``, ``"decide"`` and
                ``"control"`` (summary, set_model) can be joined
                with ``+`` for the union of their tools, e.g.
                ``"shell+edit+memory"``; ``"readonly"`` is accepted
                for ``"review"``.  Empty (default): a child of a
                reviewer, or one whose task reads as a review and
                asks for no changes, gets ``"review"`` (its ``ran``
                line says ``tools=review(inferred)``), others the
                full toolset.
            max_budget: Per-child USD budget as a number string;
                empty shares this task's remaining budget among the
                children and this task.
            timeout: Seconds this call blocks, as a number string;
                empty = the SEA's ``timeout`` setting, else 3600 — as
                for ``run_agent``.  When it expires no child is
                stopped: a child still running is returned as its
                ``agent_job`` notice (``agent_job(id, "wait")``
                collects its result, ``"kill"`` stops it; a job still
                running when this task ends is killed).
            options: JSON object of run settings to override, as for
                ``run_agent`` (``model``, ``tool_profile``,
                ``max_budget`` and ``timeout`` are the arguments above
                and are refused here); usually empty.
            description: A brief description of the fan-out, shown as
                the title of this call in the chat.

        Returns:
            A YAML list with one entry per task, in the input order:
            the child's result (a ``ran`` line — the SEA it ran as,
            its model, tool profile and budget, which values it
            inherited from this task and which the SEA pinned —
            then ``success`` and ``summary``) or, for a child still
            running at ``timeout``, its job notice.  A string
            starting with ``Error:`` when the call was refused:
            ``tasks`` is no JSON array of non-empty strings, ``agent``
            names no usable SEA, the options are malformed, or a
            locked key is contradicted.
        """
        del description
        try:
            task_list = parse_tasks_json(tasks)
        except ValueError as e:
            return f"Error: {e.args[0]}"
        return run_agents_parallel(
            current_work_dir(parent_agent, work_dir), task_list, agent, model, tool_profile,
            max_budget, timeout, options, parent_agent,
        )

    run_parallel.__doc__ = (run_parallel.__doc__ or "").replace("{precedence}", PRECEDENCE_RULE)
    run_parallel.unknown_arguments_hint = options_keyword_hint  # type: ignore[attr-defined]
    return run_parallel
