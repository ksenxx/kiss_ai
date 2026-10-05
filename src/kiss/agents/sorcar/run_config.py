# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The effective configuration a sub-task ran with, as a record and as one line.

Every task's ``task_settings`` event (the persisted display event
:meth:`kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent._task_settings_payload`
builds) carries, besides the model, work directory and budget, the
run-configuration keys of :data:`RUN_CONFIG_KEYS`: which SEA ran, its
kind, the tool profile, the caller's timeout, which values were
inherited from the calling agent and which inherited or default values
the SEA pinned to its own.  :func:`run_config_line` renders that record
as the ``ran:`` line every ``run_agent`` / ``run_parallel`` result
starts with, so the calling model sees what its sub-task actually ran
with, and ``rsi7d`` can mine the persisted events for the configuration
causes of failures.

An explicit argument of the call is never pinned over: it wins, or (for
a ``locked`` key) the call is refused before anything runs
(:data:`kiss.agents.sorcar.sea_settings.PRECEDENCE_RULE`).  So
``pinned`` only ever lists values the caller did not choose itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

RUN_CONFIG_KEYS = (
    "sea",
    "kind",
    "tool_profile",
    "tool_profile_inferred",
    "timeout",
    "inherited",
    "pinned",
    "classified",
)
"""The ``task_settings`` keys :func:`run_config_line` reads besides ``model``, ``work_dir``
and ``max_budget``.

``sea``: the SEA's name (its file stem), ``""`` for a plain sub-agent;
``kind``: the SEA's ``kind`` setting (``session``, ``worker`` or ``channel``);
``tool_profile``: the effective profile, ``""`` for the full toolset;
``tool_profile_inferred``: ``True`` when nobody named the profile and the run got
``review`` because it is a reviewer sub-agent (``tools=review(inferred)`` on the line);
``timeout``: the caller's wait in seconds, or ``None``;
``inherited``: the setting keys filled from the calling agent;
``pinned``: ``{key: [before, pinned]}`` for every inherited or default value the SEA's
``settings()`` replaced;
``classified``: ``{key: [before, after]}`` for every default the daemon's pre-run
classifier changed (today only ``use_worktree``, demoted for a non-development task).
"""

PROVENANCE_EXPLICIT = "explicit"
PROVENANCE_INHERITED = "inherited"
"""The two values of a ``run`` command's ``provenance`` wire field (``{setting key: value}``).

The keys are the ``run()`` keyword / SEA setting names (``model``,
``chat_id``, ``add_to_prompt``, ...).  A key with no entry carries a
persisted setting or the daemon's default.
"""


def note_pinned(pinned: dict[str, list[Any]], key: str, before: Any, value: Any) -> None:
    """Record that a SEA pinned *key* to *value* in place of the inherited or default *before*.

    Nothing is recorded when there was nothing to replace (``None`` or
    ``""``) or the values agree.  Dict values (``model_config``, which
    may hold an API key) are reduced to their key names, so the record
    can be persisted, broadcast and echoed to the calling model.

    Args:
        pinned: The ``pinned`` record to extend in place.
        key: The setting key.
        before: The value the run would have had without the SEA.
        value: The value the SEA's settings write.
    """
    if before is None or before == "" or before == value:
        return
    pinned[key] = [_recordable(before), _recordable(value)]


def _recordable(value: Any) -> Any:
    """Return *value* for the record: a dict as ``dict(key, key, ...)``, else unchanged."""
    if isinstance(value, Mapping):
        return "dict(" + ", ".join(sorted(str(k) for k in value)) + ")"
    return value


def sea_pinned(
    before: Mapping[str, Any],
    staged: Mapping[str, Any],
    fields: Mapping[str, str],
) -> dict[str, list[Any]]:
    """Return which non-empty command values a SEA pinned to its own.

    Prompt getters (``system_prompt`` and the two suffixes) are not
    settings and never count: a SEA with a ``system_prompt()`` replaces
    the inherited one by design.

    Args:
        before: The ``run`` command as the caller sent it.
        staged: The wire fields the SEA's settings write
            (``{wire field: value}``).
        fields: ``{setting key: wire field}`` (``SETTING_FIELDS``).

    Returns:
        ``{setting key: [before, pinned]}`` for every field in *fields*
        whose command value was neither absent, ``None`` nor ``""`` and
        differs from the staged value (see :func:`note_pinned`).
        Empty when nothing was replaced.
    """
    pinned: dict[str, list[Any]] = {}
    for key, field in fields.items():
        if field in staged:
            note_pinned(pinned, key, before.get(field), staged[field])
    return pinned


def inherited_keys(provenance: Any) -> list[str]:
    """Return the setting keys a ``run`` command's ``provenance`` marks as inherited.

    Args:
        provenance: The command's ``provenance`` wire field (anything
            but a dict counts as empty).
    """
    if not isinstance(provenance, Mapping):
        return []
    return [str(key) for key, mark in provenance.items() if mark == PROVENANCE_INHERITED]


def is_explicit(provenance: Any, key: str) -> bool:
    """Return whether a ``run`` command's ``provenance`` marks *key* as passed explicitly.

    Args:
        provenance: The command's ``provenance`` wire field (anything
            but a dict counts as empty).
        key: The setting key (``use_worktree``, ``model``, ...).
    """
    return isinstance(provenance, Mapping) and provenance.get(key) == PROVENANCE_EXPLICIT


def run_config_line(settings: Mapping[str, Any], alias: str = "") -> str:
    """Render a task's effective configuration as one line.

    Example (``/sh`` run from a task that uses a worktree)::

        sh (worker) model=gpt-5 tools=bash budget=$1.00 timeout=3600s
        inherited=model,chat_id,max_budget pinned=use_worktree(True->False)

    A plain sub-agent whose worktree default the classifier dropped
    ends with ``classified=use_worktree(True->False)`` instead; the
    entry is absent when the classifier changed nothing.

    Args:
        settings: A ``task_settings`` payload (see :data:`RUN_CONFIG_KEYS`);
            missing keys render as unknown / none.
        alias: The registered command name of a SEA the call reached
            by path, appended as ``(also agent="name")`` so the caller
            learns the shorter spelling; empty for none.

    Returns:
        The line, without a trailing newline.
    """
    sea = str(settings.get("sea") or "") or "sub-agent"
    kind = str(settings.get("kind") or "session")
    model = str(settings.get("model") or "") or "default"
    tools = str(settings.get("tool_profile") or "") or "full"
    if settings.get("tool_profile_inferred"):
        tools += "(inferred)"
    budget = settings.get("max_budget")
    budget_text = f"${budget:.2f}" if isinstance(budget, int | float) else "none"
    timeout = settings.get("timeout")
    timeout_text = f"{timeout:g}s" if isinstance(timeout, int | float) else "none"
    inherited = settings.get("inherited")
    inherited_text = ",".join(str(k) for k in inherited) if inherited else "none"
    line = (
        f"{sea} ({kind}) model={model} tools={tools} budget={budget_text} "
        f"timeout={timeout_text} inherited={inherited_text} "
        f"pinned={_changes_text(settings.get('pinned')) or 'none'}"
    )
    classified = _changes_text(settings.get("classified"))
    if classified:
        line += f" classified={classified}"
    return f'{line} (also agent="{alias}")' if alias else line


def _changes_text(changes: Any) -> str:
    """Render a ``{key: [before, after]}`` record as ``key(before->after),...``, ``""`` if empty."""
    if not isinstance(changes, Mapping) or not changes:
        return ""
    return ",".join(
        f"{key}({_short(before)}->{_short(value)})" for key, (before, value) in changes.items()
    )


def _short(value: Any) -> str:
    """Return *value* for the ``pinned`` list: ``""`` as ``empty``, long text cut."""
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
