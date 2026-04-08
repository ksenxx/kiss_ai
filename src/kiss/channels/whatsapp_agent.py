"""WhatsApp Agent — StatefulSorcarAgent extension with WhatsApp Business Cloud API tools.

Provides authenticated access to WhatsApp via the Meta Graph API.
Handles authentication (reading config from disk or prompting the user
via the browser), stores the access token and phone number ID securely
in ``~/.kiss/channels/whatsapp/config.json``, and exposes a focused set
of WhatsApp Business API tools that give the agent full control over
messaging, media, templates, and business profile management.

Usage::

    agent = WhatsAppAgent()
    agent.run(prompt_template="Send 'Hello!' to +1234567890")
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import requests

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
    wait_for_matching_message,
)
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_PORT = 18080

_WHATSAPP_DIR = Path.home() / ".kiss" / "channels" / "whatsapp"
_GRAPH_API_BASE = "https://graph.facebook.com/v21.0"
_config = ChannelConfig(_WHATSAPP_DIR, ("access_token", "phone_number_id"))


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------


def _api_request(
    method: str,
    url: str,
    access_token: str,
    json_body: dict | None = None,  # type: ignore[type-arg]
    data: dict | None = None,  # type: ignore[type-arg]
    files: dict | None = None,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Make an authenticated request to the Meta Graph API.

    Args:
        method: HTTP method (GET, POST, DELETE).
        url: Full URL to request.
        access_token: Bearer token for authorization.
        json_body: JSON body for POST requests.
        data: Form data for multipart requests.
        files: File data for multipart uploads.

    Returns:
        Parsed JSON response dict.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    kwargs: dict[str, Any] = {"headers": headers, "timeout": 30}
    if json_body is not None:
        kwargs["json"] = json_body
    if data is not None:
        kwargs["data"] = data
    if files is not None:
        kwargs["files"] = files
    resp = requests.request(method, url, **kwargs)
    try:
        return resp.json()  # type: ignore[no-any-return]
    except ValueError:  # pragma: no cover – Graph API always returns JSON
        return {"error": {"message": resp.text, "code": resp.status_code}}


# ---------------------------------------------------------------------------
# WhatsAppChannelBackend
# ---------------------------------------------------------------------------


class WhatsAppChannelBackend(ToolMethodBackend):
    """Channel backend for WhatsApp Business Cloud API.

    Provides channel monitoring via webhook queue, message sending,
    and reply waiting for the channel poller and interactive agent.

    For message polling, uses a webhook queue pattern: an embedded HTTP
    server receives POST events from the WhatsApp platform and buffers
    them; ``poll_messages()`` drains this buffer.
    """

    def __init__(self) -> None:
        self._access_token: str = ""
        self._phone_number_id: str = ""
        self._waba_id: str = ""
        self._connection_info: str = ""
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None

    def connect(self) -> bool:
        """Authenticate with WhatsApp using stored config and start webhook server.

        Returns:
            True on success, False on failure.
        """
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No WhatsApp config found. Please authenticate first."
            return False
        self._access_token = cfg["access_token"]
        self._phone_number_id = cfg["phone_number_id"]
        self._waba_id = cfg.get("waba_id", "")

        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}?fields=verified_name,display_phone_number"
        result = _api_request("GET", url, self._access_token)
        if "error" in result:  # pragma: no branch
            self._connection_info = f"WhatsApp auth failed: {result['error']}"
            return False

        self._connection_info = (
            f"Authenticated as {result.get('verified_name', '')} "
            f"({result.get('display_phone_number', '')})"
        )
        if not self._start_webhook_server():  # pragma: no branch
            return False
        return True

    def _start_webhook_server(self, port: int = _DEFAULT_WEBHOOK_PORT) -> bool:
        """Start the webhook HTTP server in a background thread.

        Args:
            port: Port to listen on. Default: 18080.

        Returns:
            True if the server started successfully, False otherwise.
        """
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                # Handle webhook verification challenge
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                challenge = params.get("hub.challenge", [""])[0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(challenge.encode())

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                    for entry in data.get("entry", []):  # pragma: no branch
                        for change in entry.get("changes", []):  # pragma: no branch
                            value = change.get("value", {})
                            for msg in value.get("messages", []):  # pragma: no branch
                                backend._message_queue.put(msg)
                except Exception:
                    pass
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass  # Suppress access log

        self.disconnect()
        try:
            self._webhook_server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
            self._webhook_thread = threading.Thread(
                target=self._webhook_server.serve_forever, daemon=True
            )
            self._webhook_thread.start()
            logger.info("WhatsApp webhook server started on port %d", port)
            return True
        except OSError as e:
            self._connection_info = f"WhatsApp webhook bind failed: {e}"
            logger.warning("Could not start webhook server: %s", e)
            self._webhook_server = None
            self._webhook_thread = None
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue and return new messages.

        Args:
            channel_id: Recipient phone number (unused — all messages returned).
            oldest: Unused for push-mode channels.
            limit: Maximum messages to return.

        Returns:
            Tuple of (messages, oldest). Each message dict has at minimum:
            ts, user (from), text.
        """
        messages: list[dict[str, Any]] = []
        while not self._message_queue.empty() and len(messages) < limit:  # pragma: no branch
            raw = self._message_queue.get_nowait()
            msg_type = raw.get("type", "")
            if msg_type == "text":  # pragma: no branch
                text = raw.get("text", {}).get("body", "")
            else:
                text = f"[{msg_type} message]"
            messages.append(
                {
                    "ts": raw.get("timestamp", ""),
                    "user": raw.get("from", ""),
                    "text": text,
                    "id": raw.get("id", ""),
                }
            )
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a text message to a WhatsApp number.

        Args:
            channel_id: Recipient phone number in E.164 format.
            text: Message text.
            thread_ts: Unused for WhatsApp.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        _api_request(
            "POST",
            url,
            self._access_token,
            json_body={
                "messaging_product": "whatsapp",
                "to": channel_id,
                "type": "text",
                "text": {"body": text},
            },
        )

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
    ) -> str | None:
        """Block until a message from a specific user is received.

        Args:
            channel_id: Unused for WhatsApp.
            thread_ts: Unused for WhatsApp.
            user_id: Phone number to wait for.

        Returns:
            The text of the user's reply.
        """

        def poll() -> list[dict[str, Any]]:
            matches: list[dict[str, Any]] = []
            others: list[dict[str, Any]] = []
            while not self._message_queue.empty():  # pragma: no branch
                raw = self._message_queue.get_nowait()
                if raw.get("from") == user_id:  # pragma: no branch
                    matches.append(raw)
                else:
                    others.append(raw)
            for item in others:  # pragma: no branch
                self._message_queue.put(item)
            return matches

        return wait_for_matching_message(
            poll=poll,
            matches=lambda raw: raw.get("from") == user_id,
            extract_text=lambda raw: str(raw.get("text", {}).get("body", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=2.0,
        )

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

    # -------------------------------------------------------------------
    # WhatsApp API tool methods (return JSON strings for LLM agent use)
    # -------------------------------------------------------------------

    def send_text_message(self, to: str, body: str, preview_url: bool = False) -> str:
        """Send a text message to a WhatsApp number.

        Args:
            to: Recipient phone number in E.164 format (e.g. "+14155238886").
                Include country code, no spaces or dashes.
            body: Message text (up to 4096 characters).
            preview_url: If True, URLs in the body will show a preview.
                Default: False.

        Returns:
            JSON string with ok status and message_id.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"preview_url": preview_url, "body": body},
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            messages = result.get("messages", [])  # pragma: no cover
            msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
            return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_template_message(
        self,
        to: str,
        template_name: str,
        language_code: str = "en_US",
        components: str = "",
    ) -> str:
        """Send a pre-approved template message.

        Template messages are required to initiate conversations outside
        the 24-hour customer service window.

        Args:
            to: Recipient phone number in E.164 format.
            template_name: Name of the approved message template.
            language_code: Template language code (e.g. "en_US").
                Default: "en_US".
            components: Optional JSON string of template components
                (header, body, button parameters).

        Returns:
            JSON string with ok status and message_id.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            template: dict[str, Any] = {
                "name": template_name,
                "language": {"code": language_code},
            }
            if components:
                template["components"] = json.loads(components)
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "template",
                    "template": template,
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            messages = result.get("messages", [])  # pragma: no cover
            msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
            return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_media_message(
        self,
        to: str,
        media_type: str,
        media_id: str = "",
        link: str = "",
        caption: str = "",
        filename: str = "",
    ) -> str:
        """Send a media message (image, document, audio, video, sticker).

        Provide either media_id (from upload_media) or link (public URL).

        Args:
            to: Recipient phone number in E.164 format.
            media_type: Type of media. Options: "image", "document",
                "audio", "video", "sticker".
            media_id: Media ID from a previous upload_media call.
            link: Public URL of the media file. Used if media_id is empty.
            caption: Optional caption (supported for image, video, document).
            filename: Optional filename (for document type).

        Returns:
            JSON string with ok status and message_id.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            media_obj: dict[str, Any] = {}
            if media_id:
                media_obj["id"] = media_id
            elif link:
                media_obj["link"] = link
            if caption and media_type in ("image", "video", "document"):
                media_obj["caption"] = caption
            if filename and media_type == "document":
                media_obj["filename"] = filename
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": media_type,
                    media_type: media_obj,
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            messages = result.get("messages", [])  # pragma: no cover
            msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
            return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_reaction(self, to: str, message_id: str, emoji: str) -> str:
        """React to a message with an emoji.

        Args:
            to: Phone number of the message recipient.
            message_id: ID of the message to react to.
            emoji: Emoji character (e.g. "👍", "❤️", "😂").

        Returns:
            JSON string with ok status and message_id.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "reaction",
                    "reaction": {"message_id": message_id, "emoji": emoji},
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            messages = result.get("messages", [])  # pragma: no cover
            msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
            return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_location_message(
        self,
        to: str,
        latitude: str,
        longitude: str,
        name: str = "",
        address: str = "",
    ) -> str:
        """Send a location message.

        Args:
            to: Recipient phone number in E.164 format.
            latitude: Latitude of the location (e.g. "37.7749").
            longitude: Longitude of the location (e.g. "-122.4194").
            name: Optional name of the location.
            address: Optional address of the location.

        Returns:
            JSON string with ok status and message_id.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            location: dict[str, Any] = {
                "latitude": latitude,
                "longitude": longitude,
            }
            if name:
                location["name"] = name
            if address:
                location["address"] = address
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "location",
                    "location": location,
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            messages = result.get("messages", [])  # pragma: no cover
            msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
            return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_interactive_message(self, to: str, interactive_json: str) -> str:
        """Send an interactive message (buttons, lists, or product messages).

        Args:
            to: Recipient phone number in E.164 format.
            interactive_json: JSON string of the interactive object.

        Returns:
            JSON string with ok status and message_id.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "interactive",
                    "interactive": json.loads(interactive_json),
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            messages = result.get("messages", [])  # pragma: no cover
            msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
            return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_contact_message(self, to: str, contacts_json: str) -> str:
        """Send a contact card message.

        Args:
            to: Recipient phone number in E.164 format.
            contacts_json: JSON string of contacts array.

        Returns:
            JSON string with ok status and message_id.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "contacts",
                    "contacts": json.loads(contacts_json),
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            messages = result.get("messages", [])  # pragma: no cover
            msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
            return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def mark_as_read(self, message_id: str) -> str:
        """Mark a received message as read.

        Args:
            message_id: ID of the message to mark as read.

        Returns:
            JSON string with ok status.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        try:
            result = _api_request(
                "POST",
                url,
                self._access_token,
                json_body={
                    "messaging_product": "whatsapp",
                    "status": "read",
                    "message_id": message_id,
                },
            )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            return json.dumps(  # pragma: no cover
                {"ok": True, "success": result.get("success", False)}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_business_profile(self) -> str:
        """Get the WhatsApp Business profile information.

        Returns:
            JSON string with business profile data (about, address,
            description, email, websites, profile_picture_url).
        """
        url = (
            f"{_GRAPH_API_BASE}/{self._phone_number_id}/whatsapp_business_profile"
            "?fields=about,address,description,email,websites,profile_picture_url,vertical"
        )
        try:
            result = _api_request("GET", url, self._access_token)
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            data_list = result.get("data", [])  # pragma: no cover
            profile = data_list[0] if data_list else {}  # pragma: no cover
            return json.dumps({"ok": True, "profile": profile}, indent=2)  # pragma: no cover
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def update_business_profile(
        self,
        about: str = "",
        address: str = "",
        description: str = "",
        email: str = "",
        websites: str = "",
        vertical: str = "",
    ) -> str:
        """Update the WhatsApp Business profile.

        Args:
            about: Short description (max 139 characters).
            address: Business address.
            description: Full business description (max 512 characters).
            email: Business email address.
            websites: Comma-separated list of website URLs (max 2).
            vertical: Business category (e.g. "RETAIL", "FOOD", "HEALTH").

        Returns:
            JSON string with ok status.
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/whatsapp_business_profile"
        try:
            body: dict[str, Any] = {"messaging_product": "whatsapp"}
            if about:
                body["about"] = about
            if address:
                body["address"] = address
            if description:
                body["description"] = description
            if email:
                body["email"] = email
            if websites:
                body["websites"] = [w.strip() for w in websites.split(",")]
            if vertical:
                body["vertical"] = vertical
            result = _api_request("POST", url, self._access_token, json_body=body)
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            return json.dumps(  # pragma: no cover
                {"ok": True, "success": result.get("success", False)}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def upload_media(self, file_path: str, mime_type: str) -> str:
        """Upload a media file for later sending.

        Args:
            file_path: Local path to the file to upload.
            mime_type: MIME type of the file (e.g. "image/jpeg",
                "application/pdf", "video/mp4", "audio/ogg").

        Returns:
            JSON string with ok status and media_id (use in
            send_media_message).
        """
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/media"
        try:
            with open(file_path, "rb") as f:
                result = _api_request(
                    "POST",
                    url,
                    self._access_token,
                    data={"messaging_product": "whatsapp", "type": mime_type},
                    files={"file": (Path(file_path).name, f, mime_type)},
                )
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            return json.dumps(  # pragma: no cover
                {"ok": True, "media_id": result.get("id", "")}
            )
        except OSError as e:
            return json.dumps({"ok": False, "error": str(e)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_media_url(self, media_id: str) -> str:
        """Get the download URL for an uploaded media file.

        Args:
            media_id: Media ID from upload_media or a received message.

        Returns:
            JSON string with ok status, url, mime_type, and file_size.
        """
        url = f"{_GRAPH_API_BASE}/{media_id}"
        try:
            result = _api_request("GET", url, self._access_token)
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            return json.dumps(
                {  # pragma: no cover
                    "ok": True,
                    "url": result.get("url", ""),
                    "mime_type": result.get("mime_type", ""),
                    "file_size": result.get("file_size", 0),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_media(self, media_id: str) -> str:
        """Delete an uploaded media file.

        Args:
            media_id: Media ID to delete.

        Returns:
            JSON string with ok status.
        """
        url = f"{_GRAPH_API_BASE}/{media_id}"
        try:
            result = _api_request("DELETE", url, self._access_token)
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            return json.dumps(  # pragma: no cover
                {"ok": True, "success": result.get("success", False)}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_message_templates(self, limit: int = 20, status: str = "") -> str:
        """List available message templates for the WhatsApp Business Account.

        Requires waba_id to be configured.

        Args:
            limit: Maximum number of templates to return. Default: 20.
            status: Filter by status ("APPROVED", "PENDING", "REJECTED").
                If empty, returns all statuses.

        Returns:
            JSON string with template list (name, status, category, language).
        """
        if not self._waba_id:
            return json.dumps(
                {
                    "ok": False,
                    "error": "waba_id not configured. Re-authenticate with "
                    "authenticate_whatsapp() and provide the WABA ID.",
                }
            )
        params = f"?limit={limit}"
        if status:
            params += f"&status={status}"
        url = f"{_GRAPH_API_BASE}/{self._waba_id}/message_templates{params}"
        try:
            result = _api_request("GET", url, self._access_token)
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            templates = [  # pragma: no cover
                {
                    "name": t.get("name", ""),
                    "status": t.get("status", ""),
                    "category": t.get("category", ""),
                    "language": t.get("language", ""),
                    "id": t.get("id", ""),
                }
                for t in result.get("data", [])
            ]
            # pragma: no cover
            return json.dumps({"ok": True, "templates": templates}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


# ---------------------------------------------------------------------------
# WhatsAppAgent
# ---------------------------------------------------------------------------


class WhatsAppAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with WhatsApp Business Cloud API tools.

    Inherits all standard SorcarAgent capabilities (bash, file editing,
    browser automation) and adds authenticated WhatsApp API tools for
    sending messages, media, templates, reactions, interactive messages,
    location/contact sharing, and business profile management.

    The agent checks for stored config on initialization. If no config
    is found, authentication tools guide the user through obtaining and
    storing their Meta access token and phone number ID.

    Example::

        agent = WhatsAppAgent()
        result = agent.run(
            prompt_template="Send 'Hello!' to +14155238886",
        )
    """

    def __init__(self) -> None:
        super().__init__("WhatsApp Agent")
        self._backend = WhatsAppChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._access_token = cfg["access_token"]
            self._backend._phone_number_id = cfg["phone_number_id"]
            self._backend._waba_id = cfg.get("waba_id", "")

    def _is_authenticated(self) -> bool:
        """Return True if WhatsApp credentials are configured."""
        return bool(self._backend._access_token)

    def _get_auth_tools(self) -> list:
        """Return WhatsApp authentication tool functions."""
        agent = self

        def check_whatsapp_auth() -> str:
            """Check if WhatsApp Business API credentials are configured.

            Tests the stored credentials against the Meta Graph API.

            Returns:
                Authentication status with phone number info, or
                instructions for how to authenticate.
            """
            if not agent._backend._access_token:
                return (
                    "Not authenticated with WhatsApp. Use "
                    "authenticate_whatsapp(access_token=..., phone_number_id=...) "
                    "to configure. To get these values:\n"
                    "1. Go to https://developers.facebook.com/apps/\n"
                    "2. Create or select a Business app with WhatsApp product\n"
                    "3. Under WhatsApp > API Setup, find:\n"
                    "   - Temporary access token (or create a System User token)\n"
                    "   - Phone number ID (shown under 'From' phone number)\n"
                    "4. Call authenticate_whatsapp(access_token='...', "
                    "phone_number_id='...')"
                )
            url = (
                f"{_GRAPH_API_BASE}/{agent._backend._phone_number_id}"
                "?fields=verified_name,display_phone_number"
            )
            result = _api_request("GET", url, agent._backend._access_token)
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            return json.dumps(
                {  # pragma: no cover
                    "ok": True,
                    "phone_number_id": agent._backend._phone_number_id,
                    "verified_name": result.get("verified_name", ""),
                    "display_phone_number": result.get("display_phone_number", ""),
                }
            )

        def authenticate_whatsapp(
            access_token: str,
            phone_number_id: str,
            waba_id: str = "",
        ) -> str:
            """Store and validate WhatsApp Business API credentials.

            Saves the credentials to ~/.kiss/channels/whatsapp/config.json
            and validates them against the Meta Graph API.

            Args:
                access_token: Meta Graph API access token.
                phone_number_id: WhatsApp Business phone number ID.
                waba_id: WhatsApp Business Account ID (optional).

            Returns:
                Validation result with phone number info, or error.
            """
            access_token = access_token.strip()
            phone_number_id = phone_number_id.strip()
            if not access_token or not phone_number_id:
                return "Both access_token and phone_number_id are required."
            url = f"{_GRAPH_API_BASE}/{phone_number_id}?fields=verified_name,display_phone_number"
            result = _api_request("GET", url, access_token)
            if "error" in result:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"Credential validation failed: {result['error']}",
                    }
                )
            _config.save(
                {
                    "access_token": access_token.strip(),
                    "phone_number_id": phone_number_id.strip(),
                    "waba_id": waba_id.strip(),
                }
            )  # pragma: no cover
            agent._backend._access_token = access_token  # pragma: no cover
            agent._backend._phone_number_id = phone_number_id  # pragma: no cover
            agent._backend._waba_id = waba_id.strip()  # pragma: no cover
            return json.dumps(
                {  # pragma: no cover
                    "ok": True,
                    "message": "WhatsApp credentials saved and validated.",
                    "verified_name": result.get("verified_name", ""),
                    "display_phone_number": result.get("display_phone_number", ""),
                }
            )

        def clear_whatsapp_auth() -> str:
            """Clear the stored WhatsApp authentication credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._access_token = ""
            agent._backend._phone_number_id = ""
            agent._backend._waba_id = ""
            return "WhatsApp authentication cleared."

        return [check_whatsapp_auth, authenticate_whatsapp, clear_whatsapp_auth]


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run the WhatsAppAgent from the command line with chat persistence."""
    channel_main(WhatsAppAgent, "kiss-whatsapp")


if __name__ == "__main__":
    main()
