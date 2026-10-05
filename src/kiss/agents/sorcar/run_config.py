# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The effective configuration a sub-task ran with, as a record and as one line.

Every task's ``task_settings`` event (the persisted display event
:meth:`kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent._task_settings_payload`
builds) carries, besides the model, work directory and budget, the
run-configuration keys of :data:`RUN_CONFIG_KEYS`: which agent script
(SEA) ran, its kind, the tool profile, the caller's timeout, which
values were inherited from the calling agent and which of the
caller's values the script replaced.  :func:`run_config_line` renders
that record as the ``ran:`` line every ``run_agent`` / ``run_parallel``
result starts with, so the calling model sees what its sub-task
actually ran with (and self-corrects an argument the script silently
replaced), and ``rsi7d`` can mine the persisted events for the
configuration causes of failures (``reports/sea-run-agent-semantics-
and-automation-2026-10-04.md``, P3 / A4).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

RUN_CONFIG_KEYS = (
    "sea",
    "kind",
    "tool_profile",
    "timeout",
    "inherited",
    "overridden",
)
"""The ``task_settings`` keys :func:`run_config_line` reads besides ``model``, ``work_dir``
and ``max_budget``.

``sea``: the script's name (its file stem), ``""`` for a plain sub-agent;
``kind``: the script's ``kind`` setting (``session``, ``worker`` or ``channel``);
``tool_profile``: the effective profile, ``""`` for the full toolset;
``timeout``: the caller's wait in seconds, or ``None``;
``inherited``: the setting keys filled from the calling agent;
``overridden``: ``{key: [asked, forced]}`` for every non-empty value the script replaced.
"""

PROVENANCE_EXPLICIT = "explicit"
PROVENANCE_INHERITED = "inherited"
"""The two values of a ``run`` command's ``provenance`` wire field (``{setting key: value}``).

The keys are the ``run()`` keyword / SEA setting names (``model``,
``chat_id``, ``add_to_prompt``, ...).  A key with no entry carries a
persisted setting or the daemon's default.
"""


def note_override(overridden: dict[str, list[Any]], key: str, asked: Any, forced: Any) -> None:
    """Record that an agent script replaced the value *asked* for *key* with *forced*.

    Nothing is recorded when the caller asked for nothing (``None`` or
    ``""``) or for the same value.  Dict values (``model_config``, which
    may hold an API key) are reduced to their key names, so the record
    can be persisted, broadcast and echoed to the calling model.

    Args:
        overridden: The ``overridden`` record to extend in place.
        key: The setting key.
        asked: The value the command carried.
        forced: The value the script's settings write.
    """
    if asked is None or asked == "" or asked == forced:
        return
    overridden[key] = [_recordable(asked), _recordable(forced)]


def _recordable(value: Any) -> Any:
    """Return *value* for the record: a dict as ``dict(key, key, ...)``, else unchanged."""
    if isinstance(value, Mapping):
        return "dict(" + ", ".join(sorted(str(k) for k in value)) + ")"
    return value


def sea_overrides(
    before: Mapping[str, Any],
    staged: Mapping[str, Any],
    fields: Mapping[str, str],
) -> dict[str, list[Any]]:
    """Return which non-empty command values an agent script replaced.

    Prompt getters (``system_prompt`` and the two suffixes) are not
    settings and never count as overrides: a script with a
    ``system_prompt()`` replaces the inherited one by design.

    Args:
        before: The ``run`` command as the caller sent it.
        staged: The wire fields the script's settings write
            (``{wire field: value}``).
        fields: ``{setting key: wire field}`` (``SETTING_FIELDS``).

    Returns:
        ``{setting key: [asked, forced]}`` for every field in *fields*
        whose command value was neither absent, ``None`` nor ``""`` and
        differs from the staged value (see :func:`note_override`).
        Empty when nothing was replaced.
    """
    overridden: dict[str, list[Any]] = {}
    for key, field in fields.items():
        if field in staged:
            note_override(overridden, key, before.get(field), staged[field])
    return overridden


def inherited_keys(provenance: Any) -> list[str]:
    """Return the setting keys a ``run`` command's ``provenance`` marks as inherited.

    Args:
        provenance: The command's ``provenance`` wire field (anything
            but a dict counts as empty).
    """
    if not isinstance(provenance, Mapping):
        return []
    return [str(key) for key, mark in provenance.items() if mark == PROVENANCE_INHERITED]


def run_config_line(settings: Mapping[str, Any]) -> str:
    """Render a task's effective configuration as one line.

    Example::

        sh (worker) model=gpt-5 tools=bash budget=$1.00 timeout=3600s
        inherited=model,chat_id,max_budget overridden=tool_profile(review->bash)

    Args:
        settings: A ``task_settings`` payload (see :data:`RUN_CONFIG_KEYS`);
            missing keys render as unknown / none.

    Returns:
        The line, without a trailing newline.
    """
    sea = str(settings.get("sea") or "") or "sub-agent"
    kind = str(settings.get("kind") or "session")
    model = str(settings.get("model") or "") or "default"
    tools = str(settings.get("tool_profile") or "") or "full"
    budget = settings.get("max_budget")
    budget_text = f"${budget:.2f}" if isinstance(budget, int | float) else "none"
    timeout = settings.get("timeout")
    timeout_text = f"{timeout:g}s" if isinstance(timeout, int | float) else "none"
    inherited = settings.get("inherited")
    inherited_text = ",".join(str(k) for k in inherited) if inherited else "none"
    overridden = settings.get("overridden")
    if isinstance(overridden, Mapping) and overridden:
        overridden_text = ",".join(
            f"{key}({_short(asked)}->{_short(forced)})"
            for key, (asked, forced) in overridden.items()
        )
    else:
        overridden_text = "none"
    return (
        f"{sea} ({kind}) model={model} tools={tools} budget={budget_text} "
        f"timeout={timeout_text} inherited={inherited_text} overridden={overridden_text}"
    )


def _short(value: Any) -> str:
    """Return *value* for the ``overridden`` list: ``""`` as ``empty``, long text cut."""
    text = "empty" if value == "" else str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def with_run_config(result: str, settings: Mapping[str, Any]) -> str:
    """Return a sub-agent's YAML *result* with the ``ran:`` line first.

    Args:
        result: The sub-agent's result text, normally a YAML mapping
            with ``success`` and ``summary`` keys.
        settings: Its ``task_settings`` payload.

    Returns:
        The YAML mapping re-dumped with ``ran`` as its first key; a
        result that is not a mapping is prefixed with a ``ran:`` line.
    """
    line = run_config_line(settings)
    try:
        parsed = yaml.safe_load(result)
    except yaml.YAMLError:
        parsed = None
    if not isinstance(parsed, dict):
        return f"ran: {line}\n{result}"
    ordered: dict[str, Any] = {"ran": line}
    ordered.update((key, value) for key, value in parsed.items() if key != "ran")
    dumped: str = yaml.safe_dump(ordered, sort_keys=False)
    return dumped
