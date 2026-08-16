# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Home Assistant Agent — channel agent for the Home Assistant REST API.

Provides access to a Home Assistant instance via its REST API using a
long-lived access token (sent as ``Authorization: Bearer`` on every
call).  Stores config in
``~/.kiss/third_party_agents/homeassistant/config.json``.

Home Assistant's plain REST API has no meaningful inbound message
stream, so this adapter is outbound-only: ``poll_messages`` always
returns no messages and the ``--channel`` poll mode is disabled
(``main`` passes ``make_backend=None`` to ``channel_main``).
``send_message`` delivers text as a Home Assistant persistent
notification.

Usage::

    agent = HomeAssistantAgent()
    agent.run(prompt_template="Turn off all the lights in the kitchen")
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 30


def _bad_segment(value: str, name: str) -> str | None:
    """Reject *value* if it cannot safely form a single URL path segment.

    Values containing a path separator or a ``..`` sequence could
    traverse out of the intended API endpoint, so they are refused up
    front (defense in depth on top of ``quote(value, safe="")``).

    Args:
        value: Caller-supplied identifier destined for a URL path segment.
        name: Parameter name used in the error message.

    Returns:
        An ``{"ok": false, "error": ...}`` JSON string if *value* is
        unsafe, otherwise None.
    """
    if "/" in value or "\\" in value or ".." in value:
        return json.dumps(
            {"ok": False, "error": f"invalid {name}: must not contain path separators or '..'"}
        )
    return None


_HOMEASSISTANT_DIR = Path.home() / ".kiss" / "third_party_agents" / "homeassistant"
_config = ChannelConfig(_HOMEASSISTANT_DIR, ("base_url", "token"))


class HomeAssistantChannelBackend(ToolMethodBackend):
    """Channel backend for the Home Assistant REST API.

    Talks to a Home Assistant instance over HTTP with a long-lived
    access token.  Outbound-only: there is no inbound message stream
    over plain REST, so :meth:`poll_messages` always returns no
    messages.
    """

    def __init__(self) -> None:
        self._base_url: str = ""
        self._token: str = ""
        self._request_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the Home Assistant config from disk.

        Returns:
            True if a valid config with ``base_url`` and ``token`` was loaded.
        """
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No Home Assistant config found."
            return False
        self._base_url = cfg["base_url"]
        self._token = cfg["token"]
        self._connection_info = f"Home Assistant configured at {self._base_url}"
        return True

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> str:
        """Issue an authenticated Home Assistant REST request.

        Args:
            method: HTTP method (``"GET"`` or ``"POST"``).
            path: API path starting with ``/api/``.
            payload: Optional JSON body for POST requests.

        Returns:
            JSON string ``{"ok": true, "result": ...}`` on success or
            ``{"ok": false, "error": ...}`` on an HTTP error status.
        """
        url = self._base_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        with self._request_lock:
            resp = requests.request(method, url, headers=headers, json=payload, timeout=_TIMEOUT)
        if resp.status_code >= 400:
            return json.dumps({"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"})
        try:
            result: Any = resp.json()
        except ValueError:
            result = resp.text
        return json.dumps({"ok": True, "result": result})

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return no messages: Home Assistant REST has no inbound message stream.

        Args:
            channel_id: Ignored.
            oldest: Cursor, returned unchanged.
            limit: Ignored.

        Returns:
            ``([], oldest)``.
        """
        return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Deliver *text* as a Home Assistant persistent notification.

        Args:
            channel_id: Used as the notification title; defaults to
                ``"KISS Sorcar"`` when empty.
            text: Notification message body.
            thread_ts: Ignored (Home Assistant has no threads).

        Raises:
            RuntimeError: If the notification service call fails.
        """
        payload = {"message": text, "title": channel_id or "KISS Sorcar"}
        result = json.loads(
            self._request("POST", "/api/services/persistent_notification/create", payload)
        )
        if not result.get("ok"):
            raise RuntimeError(f"Home Assistant notification failed: {result.get('error')}")

    def ha_get_states(self, entity_id: str = "") -> str:
        """Get the state of one entity or of all entities.

        Args:
            entity_id: Entity ID such as ``"light.kitchen"``.  Empty
                returns the states of all entities.

        Returns:
            JSON string with ok status and the state object(s).
        """
        try:
            if entity_id:
                err = _bad_segment(entity_id, "entity_id")
                if err:
                    return err
            path = f"/api/states/{quote(entity_id, safe='')}" if entity_id else "/api/states"
            return self._request("GET", path)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def ha_call_service(
        self, domain: str, service: str, entity_id: str = "", data_json: str = ""
    ) -> str:
        """Call a Home Assistant service (e.g. turn a light on).

        Args:
            domain: Service domain such as ``"light"`` or ``"switch"``.
            service: Service name such as ``"turn_on"``.
            entity_id: Optional target entity ID, merged into the
                service data as ``entity_id``.
            data_json: Optional JSON object string with extra service
                data (e.g. ``'{"brightness": 128}'``).

        Returns:
            JSON string with ok status and the states changed by the call.
        """
        try:
            err = _bad_segment(domain, "domain") or _bad_segment(service, "service")
            if err:
                return err
            data: Any = json.loads(data_json) if data_json else {}
            if not isinstance(data, dict):
                return json.dumps({"ok": False, "error": "data_json must be a JSON object"})
            if entity_id:
                data["entity_id"] = entity_id
            return self._request(
                "POST", f"/api/services/{quote(domain, safe='')}/{quote(service, safe='')}", data
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def ha_list_services(self) -> str:
        """List all available Home Assistant services grouped by domain.

        Returns:
            JSON string with ok status and the service catalog.
        """
        try:
            return self._request("GET", "/api/services")
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def ha_get_history(self, entity_id: str, hours: int = 24) -> str:
        """Get the recent state history of an entity.

        Args:
            entity_id: Entity ID such as ``"sensor.temperature"``.
            hours: How many hours back to fetch (default 24).

        Returns:
            JSON string with ok status and the state history.
        """
        try:
            err = _bad_segment(entity_id, "entity_id")
            if err:
                return err
            start = datetime.now(UTC) - timedelta(hours=hours)
            iso_start = start.isoformat(timespec="seconds").replace("+00:00", "Z")
            path = (
                f"/api/history/period/{quote(iso_start, safe=':')}"
                f"?filter_entity_id={quote(entity_id, safe='')}"
            )
            return self._request("GET", path)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def ha_render_template(self, template: str) -> str:
        """Render a Home Assistant Jinja2 template.

        Args:
            template: Template string, e.g.
                ``"{{ states('sensor.temperature') }}"``.

        Returns:
            JSON string with ok status and the rendered text.
        """
        try:
            return self._request("POST", "/api/template", {"template": template})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def ha_fire_event(self, event_type: str, data_json: str = "") -> str:
        """Fire a custom event on the Home Assistant event bus.

        Args:
            event_type: Event type name, e.g. ``"kiss_sorcar_alert"``.
            data_json: Optional JSON object string with event data.

        Returns:
            JSON string with ok status and the API response message.
        """
        try:
            err = _bad_segment(event_type, "event_type")
            if err:
                return err
            data: Any = json.loads(data_json) if data_json else {}
            if not isinstance(data, dict):
                return json.dumps({"ok": False, "error": "data_json must be a JSON object"})
            return self._request("POST", f"/api/events/{quote(event_type, safe='')}", data)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class HomeAssistantAgent(BaseChannelAgent):
    """Channel agent with Home Assistant REST API tools."""

    channel_system_prompt = (
        "You are operating a Home Assistant smart-home instance through its "
        "REST API. Use ha_get_states to inspect entities, ha_call_service to "
        "act on devices (e.g. domain 'light', service 'turn_on'), "
        "ha_list_services to discover services, ha_get_history for past "
        "states, ha_render_template for Jinja2 templates, and ha_fire_event "
        "to fire events. There is no inbound chat stream; notify the user "
        "with a persistent notification when asked."
    )

    def __init__(self) -> None:
        super().__init__("Home Assistant Agent")
        self._backend = HomeAssistantChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._base_url = cfg["base_url"]
            self._backend._token = cfg["token"]

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._base_url and self._backend._token)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_homeassistant_auth() -> str:
            """Check if Home Assistant is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for Home Assistant. Use "
                    "authenticate_homeassistant() to configure.\n"
                    "You need the instance base URL (e.g. "
                    "http://homeassistant.local:8123) and a long-lived access "
                    "token from your Home Assistant profile > Security > "
                    "Long-lived access tokens."
                )
            return json.dumps({"ok": True, "base_url": agent._backend._base_url})

        def authenticate_homeassistant(base_url: str, token: str) -> str:
            """Configure the Home Assistant base URL and access token.

            Args:
                base_url: Home Assistant base URL, e.g.
                    ``http://homeassistant.local:8123``.
                token: Long-lived access token.

            Returns:
                Configuration result or error message.
            """
            if not base_url.strip() or not token.strip():
                return "base_url and token cannot be empty."
            agent._backend._base_url = base_url.strip()
            agent._backend._token = token.strip()
            _config.save({"base_url": base_url.strip(), "token": token.strip()})
            return json.dumps({"ok": True, "message": "Home Assistant configured."})

        def clear_homeassistant_auth() -> str:
            """Clear the stored Home Assistant configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._base_url = ""
            agent._backend._token = ""
            return "Home Assistant configuration cleared."

        return [check_homeassistant_auth, authenticate_homeassistant, clear_homeassistant_auth]


def main() -> None:
    """Run the HomeAssistantAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): Home Assistant's REST
    API has no inbound message stream to poll.
    """
    channel_main(
        HomeAssistantAgent,
        "kiss-ha",
        channel_name="Home Assistant",
        make_backend=None,
    )


def get_tools() -> list:
    """Return the Home Assistant channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return HomeAssistantAgent()._get_tools()


if __name__ == "__main__":
    main()
