# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Projects: named groups of chats whose settings override the product's.

A project is a row of the ``projects`` table of ``history.db`` and a chat
is in at most one (``chat_summaries.project_id``); the rows are read and
written by ``persistence``.  This module holds what a project's settings
mean:

* ``settings`` keeps only what differs from the settings of the product
  (``config.json`` and the key store), so a key the project does not
  name follows the product's setting, also when that changes later
  (:func:`settings_overrides`);
* a task of a chat in a project runs with those overrides on top of the
  configuration (:func:`config_with`) and reads the project's API keys
  and switches instead of the daemon's (:func:`overlay_fields`, the
  values of ``kiss.core.config.run_overlay``).

The settings apply to tasks that run on this machine.
"""

from __future__ import annotations

import math
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.core.secret_redaction import add_secrets
from kiss.core.user_config import API_KEY_ENV_VARS, DEFAULTS

NAME_MAX = 200
"""Characters kept of a project's name."""

DESCRIPTION_MAX = 2000
"""Characters kept of a project's description."""

HISTORY_PROJECT_PAGE = 200
"""Rows in a ``history`` page of one project (``getHistory`` with ``project``).

Larger than a page of the whole history: unfolding a project lists all
its chats, and most projects fit in one page.
"""

CONFIG_KEYS: tuple[str, ...] = (
    "default_work_dir",
    "max_budget",
    "use_web_browser",
    "use_memory",
    "compaction",
    "chrome_bridge",
    "claude_code_use_subscription",
    "codex_use_subscription",
)
"""The ``config.json`` keys a project can override."""

_OVERLAY_FIELDS = {
    "compaction": "tool_output_compaction",
    "claude_code_use_subscription": "claude_code_use_subscription",
    "codex_use_subscription": "codex_use_subscription",
}
"""The ``kiss.core.config.Config`` field a run reads each of these settings from."""


def _typed(key: str, value: object) -> object | None:
    """Return *value* when it has the type config key *key* holds, else ``None``.

    Stricter than ``sanitize_config``, which turns junk into the
    default: a project stores a value only when the user gave one.

    Args:
        key: A key of :data:`CONFIG_KEYS`.
        value: The value a client sent for it.

    Returns:
        A bool for a switch, a finite float for a number, a stripped
        string for a text; ``None`` for anything else.
    """
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return value if isinstance(value, bool) else None
    if isinstance(default, int | float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None
    return value.strip() if isinstance(value, str) else None


def settings_overrides(
    raw: object, config: dict[str, Any], api_keys: dict[str, str],
) -> dict[str, Any]:
    """Reduce a project's settings form to what differs from the product's settings.

    The page sends what the project overrides already and what the user
    set in its panel; a setting the user left alone is absent
    (``media/projects.js`` ``collectSettings``).  A config key or an API key
    whose value equals the product's is dropped, and so is an empty
    text, an unknown key and a value of the wrong type; what is left is
    what the project overrides.

    Args:
        raw: The ``settings`` object a client sent.
        config: The configuration (``load_config()``).
        api_keys: The key store's values (``get_current_api_keys()``).

    Returns:
        The settings to store: a subset of :data:`CONFIG_KEYS`,
        ``victory_sound`` (a preference of the page, kept as sent) and
        ``api_keys`` (name to value) when the project has keys of its own.
    """
    if not isinstance(raw, dict):
        return {}
    kept: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        value = _typed(key, raw.get(key))
        if value is not None and value != "" and value != config.get(key):
            kept[key] = value
    if isinstance(raw.get("victory_sound"), bool):
        kept["victory_sound"] = raw["victory_sound"]
    sent_keys = raw.get("api_keys")
    own_keys = {
        name: value.strip()
        for name, value in (sent_keys.items() if isinstance(sent_keys, dict) else ())
        if name in API_KEY_ENV_VARS
        and isinstance(value, str)
        and value.strip() not in ("", api_keys.get(name, ""))
    }
    if own_keys:
        kept["api_keys"] = own_keys
    return kept


def chat_settings(chat_id: str) -> dict[str, Any]:
    """Return what the project of chat *chat_id* overrides.

    Args:
        chat_id: The chat session id.

    Returns:
        The project's stored settings, ``{}`` for a chat in no project.
        Typed again on the way out (the stored JSON may have been
        edited by hand): a value of the wrong type is left out.
    """
    project = persistence._get_project(persistence._chat_project(chat_id))
    return settings_overrides(project["settings"], {}, {}) if project else {}


def config_with(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Return *config* with the config keys project *settings* override.

    Args:
        config: The configuration (``load_config()``).
        settings: A project's stored settings.

    Returns:
        A new dict; *config* is not modified.
    """
    return config | {key: settings[key] for key in CONFIG_KEYS if key in settings}


def _api_keys(settings: dict[str, Any]) -> dict[str, str]:
    """Return the API keys project *settings* hold (name to value)."""
    keys = settings.get("api_keys")
    return {
        name: value
        for name, value in (keys.items() if isinstance(keys, dict) else ())
        if name in API_KEY_ENV_VARS and isinstance(value, str) and value
    }


def overlay_fields(settings: dict[str, Any]) -> dict[str, Any]:
    """Return the ``Config`` fields a run of a project reads with the project's values.

    Args:
        settings: A project's stored settings.

    Returns:
        The project's API keys under the name of their variable, and the
        switches of :data:`_OVERLAY_FIELDS` it overrides.
    """
    fields: dict[str, Any] = dict(_api_keys(settings))
    for key, field in _OVERLAY_FIELDS.items():
        if isinstance(settings.get(key), bool):
            fields[field] = settings[key]
    return fields


def register_keys(settings: dict[str, Any]) -> None:
    """Have the API keys of project *settings* replaced in tool output from now on.

    They are stored in ``history.db``, which a task can query, and are
    in the environment of the commands of the project's tasks; the
    redaction of tool output otherwise knows the key store's values only.

    Args:
        settings: A project's settings, stored or about to be.
    """
    add_secrets(_api_keys(settings))


def register_secrets() -> None:
    """Have the API keys of every stored project replaced in tool output (:func:`register_keys`)."""
    for project in persistence._list_projects():
        register_keys(project["settings"])


def projects_event(with_chats: bool) -> dict[str, Any]:
    """Build the ``projects`` event: every project, oldest first.

    Args:
        with_chats: Whether to add ``chatProjects``, the project of every
            chat that is in one (the reply to ``getProjects``; the
            broadcast after a change carries the list alone).

    Returns:
        The event.
    """
    event: dict[str, Any] = {"type": "projects", "projects": persistence._list_projects()}
    if with_chats:
        event["chatProjects"] = persistence._chat_projects()
    return event
