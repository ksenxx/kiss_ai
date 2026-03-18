"""WhatsApp Agent — SorcarAgent extension with WhatsApp Business Cloud API tools.

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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from kiss.agents.sorcar.sorcar_agent import SorcarAgent

logger = logging.getLogger(__name__)

_WHATSAPP_DIR = Path.home() / ".kiss" / "channels" / "whatsapp"
_GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    """Return the path to the stored WhatsApp config file.

    Returns:
        Path to ``~/.kiss/channels/whatsapp/config.json``.
    """
    return _WHATSAPP_DIR / "config.json"


def _load_config() -> dict[str, str] | None:
    """Load stored WhatsApp config from disk.

    Returns:
        Dict with ``access_token`` and ``phone_number_id``, or None.
    """
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if (
            isinstance(data, dict)
            and data.get("access_token")
            and data.get("phone_number_id")
        ):
            return {
                "access_token": data["access_token"],
                "phone_number_id": data["phone_number_id"],
                "waba_id": data.get("waba_id", ""),
            }
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_config(access_token: str, phone_number_id: str, waba_id: str = "") -> None:
    """Save WhatsApp config to disk with restricted permissions.

    Args:
        access_token: Meta Graph API access token.
        phone_number_id: WhatsApp Business phone number ID.
        waba_id: WhatsApp Business Account ID (optional).
    """
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "access_token": access_token.strip(),
        "phone_number_id": phone_number_id.strip(),
        "waba_id": waba_id.strip(),
    }, indent=2))
    path.chmod(0o600)


def _clear_config() -> None:
    """Delete the stored WhatsApp config."""
    path = _config_path()
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------


def _api_request(
    method: str,
    url: str,
    access_token: str,
    json_body: dict | None = None,
    data: dict | None = None,
    files: dict | None = None,
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
# WhatsApp API tool functions
# ---------------------------------------------------------------------------


def _make_whatsapp_tools(access_token: str, phone_number_id: str, waba_id: str) -> list:
    """Create WhatsApp API tool functions bound to the given credentials.

    Args:
        access_token: Meta Graph API access token.
        phone_number_id: WhatsApp Business phone number ID.
        waba_id: WhatsApp Business Account ID (may be empty).

    Returns:
        List of callable tool functions for WhatsApp operations.
    """
    msg_url = f"{_GRAPH_API_BASE}/{phone_number_id}/messages"

    def _send(payload: dict) -> str:
        result = _api_request("POST", msg_url, access_token, json_body=payload)
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        messages = result.get("messages", [])  # pragma: no cover – needs live API
        msg_id = messages[0]["id"] if messages else ""  # pragma: no cover
        return json.dumps({"ok": True, "message_id": msg_id})  # pragma: no cover

    def send_text_message(to: str, body: str, preview_url: bool = False) -> str:
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
        return _send({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"preview_url": preview_url, "body": body},
        })

    def send_template_message(
        to: str, template_name: str, language_code: str = "en_US",
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
                (header, body, button parameters). Example:
                '[{"type":"body","parameters":[{"type":"text","text":"John"}]}]'

        Returns:
            JSON string with ok status and message_id.
        """
        template: dict[str, Any] = {
            "name": template_name,
            "language": {"code": language_code},
        }
        if components:
            template["components"] = json.loads(components)
        return _send({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": template,
        })

    def send_media_message(
        to: str, media_type: str, media_id: str = "", link: str = "",
        caption: str = "", filename: str = "",
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
        media_obj: dict[str, Any] = {}
        if media_id:
            media_obj["id"] = media_id
        elif link:
            media_obj["link"] = link
        if caption and media_type in ("image", "video", "document"):
            media_obj["caption"] = caption
        if filename and media_type == "document":
            media_obj["filename"] = filename
        return _send({
            "messaging_product": "whatsapp",
            "to": to,
            "type": media_type,
            media_type: media_obj,
        })

    def send_reaction(to: str, message_id: str, emoji: str) -> str:
        """React to a message with an emoji.

        Args:
            to: Phone number of the message recipient.
            message_id: ID of the message to react to.
            emoji: Emoji character (e.g. "👍", "❤️", "😂").

        Returns:
            JSON string with ok status and message_id.
        """
        return _send({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "reaction",
            "reaction": {"message_id": message_id, "emoji": emoji},
        })

    def send_location_message(
        to: str, latitude: str, longitude: str,
        name: str = "", address: str = "",
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
        location: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
        }
        if name:
            location["name"] = name
        if address:
            location["address"] = address
        return _send({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "location",
            "location": location,
        })

    def send_interactive_message(to: str, interactive_json: str) -> str:
        """Send an interactive message (buttons, lists, or product messages).

        Args:
            to: Recipient phone number in E.164 format.
            interactive_json: JSON string of the interactive object.
                For buttons example:
                '{"type":"button","body":{"text":"Choose:"},
                  "action":{"buttons":[
                    {"type":"reply","reply":{"id":"1","title":"Yes"}},
                    {"type":"reply","reply":{"id":"2","title":"No"}}
                  ]}}'
                For list example:
                '{"type":"list","body":{"text":"Pick one:"},
                  "action":{"button":"Menu","sections":[
                    {"title":"Options","rows":[
                      {"id":"1","title":"Option A"},
                      {"id":"2","title":"Option B"}
                    ]}
                  ]}}'

        Returns:
            JSON string with ok status and message_id.
        """
        return _send({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": json.loads(interactive_json),
        })

    def send_contact_message(to: str, contacts_json: str) -> str:
        """Send a contact card message.

        Args:
            to: Recipient phone number in E.164 format.
            contacts_json: JSON string of contacts array. Example:
                '[{"name":{"formatted_name":"John Doe","first_name":"John",
                  "last_name":"Doe"},"phones":[{"phone":"+14155238886",
                  "type":"CELL"}]}]'

        Returns:
            JSON string with ok status and message_id.
        """
        return _send({
            "messaging_product": "whatsapp",
            "to": to,
            "type": "contacts",
            "contacts": json.loads(contacts_json),
        })

    def mark_as_read(message_id: str) -> str:
        """Mark a received message as read.

        Args:
            message_id: ID of the message to mark as read.

        Returns:
            JSON string with ok status.
        """
        result = _api_request("POST", msg_url, access_token, json_body={
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        })
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        return json.dumps(  # pragma: no cover – needs live API
            {"ok": True, "success": result.get("success", False)})

    def get_business_profile() -> str:
        """Get the WhatsApp Business profile information.

        Returns:
            JSON string with business profile data (about, address,
            description, email, websites, profile_picture_url).
        """
        url = f"{_GRAPH_API_BASE}/{phone_number_id}/whatsapp_business_profile"
        result = _api_request("GET", f"{url}?fields=about,address,description,"
                              "email,websites,profile_picture_url,vertical",
                              access_token)
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        data_list = result.get("data", [])  # pragma: no cover – needs live API
        profile = data_list[0] if data_list else {}  # pragma: no cover
        return json.dumps({"ok": True, "profile": profile}, indent=2)  # pragma: no cover

    def update_business_profile(
        about: str = "", address: str = "", description: str = "",
        email: str = "", websites: str = "", vertical: str = "",
    ) -> str:
        """Update the WhatsApp Business profile.

        Args:
            about: Short description (max 139 characters).
            address: Business address.
            description: Full business description (max 512 characters).
            email: Business email address.
            websites: Comma-separated list of website URLs (max 2).
            vertical: Business category (e.g. "RETAIL", "FOOD",
                "HEALTH", "TRAVEL", "EDU", "OTHER").

        Returns:
            JSON string with ok status.
        """
        url = f"{_GRAPH_API_BASE}/{phone_number_id}/whatsapp_business_profile"
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
        result = _api_request("POST", url, access_token, json_body=body)
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        return json.dumps(  # pragma: no cover – needs live API
            {"ok": True, "success": result.get("success", False)})

    def upload_media(file_path: str, mime_type: str) -> str:
        """Upload a media file for later sending.

        Args:
            file_path: Local path to the file to upload.
            mime_type: MIME type of the file (e.g. "image/jpeg",
                "application/pdf", "video/mp4", "audio/ogg").

        Returns:
            JSON string with ok status and media_id (use in
            send_media_message).
        """
        url = f"{_GRAPH_API_BASE}/{phone_number_id}/media"
        try:
            with open(file_path, "rb") as f:
                result = _api_request(
                    "POST", url, access_token,
                    data={"messaging_product": "whatsapp", "type": mime_type},
                    files={"file": (Path(file_path).name, f, mime_type)},
                )
        except OSError as e:
            return json.dumps({"ok": False, "error": str(e)})
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        return json.dumps(  # pragma: no cover – needs live API
            {"ok": True, "media_id": result.get("id", "")})

    def get_media_url(media_id: str) -> str:
        """Get the download URL for an uploaded media file.

        Args:
            media_id: Media ID from upload_media or a received message.

        Returns:
            JSON string with ok status, url, mime_type, and file_size.
        """
        url = f"{_GRAPH_API_BASE}/{media_id}"
        result = _api_request("GET", url, access_token)
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        return json.dumps({  # pragma: no cover – needs live API
            "ok": True,
            "url": result.get("url", ""),
            "mime_type": result.get("mime_type", ""),
            "file_size": result.get("file_size", 0),
        })

    def delete_media(media_id: str) -> str:
        """Delete an uploaded media file.

        Args:
            media_id: Media ID to delete.

        Returns:
            JSON string with ok status.
        """
        url = f"{_GRAPH_API_BASE}/{media_id}"
        result = _api_request("DELETE", url, access_token)
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        return json.dumps(  # pragma: no cover – needs live API
            {"ok": True, "success": result.get("success", False)})

    def list_message_templates(limit: int = 20, status: str = "") -> str:
        """List available message templates for the WhatsApp Business Account.

        Requires waba_id to be configured.

        Args:
            limit: Maximum number of templates to return. Default: 20.
            status: Filter by status ("APPROVED", "PENDING", "REJECTED").
                If empty, returns all statuses.

        Returns:
            JSON string with template list (name, status, category,
            language, components).
        """
        if not waba_id:
            return json.dumps({
                "ok": False,
                "error": "waba_id not configured. Re-authenticate with "
                         "authenticate_whatsapp() and provide the WABA ID.",
            })
        params = f"?limit={limit}"
        if status:
            params += f"&status={status}"
        url = f"{_GRAPH_API_BASE}/{waba_id}/message_templates{params}"
        result = _api_request("GET", url, access_token)
        if "error" in result:
            return json.dumps({"ok": False, "error": result["error"]})
        templates = [  # pragma: no cover – needs live API
            {
                "name": t.get("name", ""),
                "status": t.get("status", ""),
                "category": t.get("category", ""),
                "language": t.get("language", ""),
                "id": t.get("id", ""),
            }
            for t in result.get("data", [])
        ]
        return json.dumps({"ok": True, "templates": templates}, indent=2)[:8000]  # pragma: no cover

    return [
        send_text_message,
        send_template_message,
        send_media_message,
        send_reaction,
        send_location_message,
        send_interactive_message,
        send_contact_message,
        mark_as_read,
        get_business_profile,
        update_business_profile,
        upload_media,
        get_media_url,
        delete_media,
        list_message_templates,
    ]


# ---------------------------------------------------------------------------
# WhatsAppAgent
# ---------------------------------------------------------------------------


def _cli_wait_for_user(instruction: str, url: str) -> None:  # pragma: no cover
    """CLI callback for browser-action prompts (prints and waits for Enter).

    Args:
        instruction: What the user should do.
        url: Current browser URL (printed if non-empty).
    """
    print(f"\n>>> Browser action needed: {instruction}")
    if url:
        print(f"    Current URL: {url}")
    input("Press Enter when done... ")


def _cli_ask_user_question(question: str) -> str:  # pragma: no cover
    """CLI callback for agent questions (prints and reads from stdin).

    Args:
        question: The question to display to the user.

    Returns:
        The user's typed response text.
    """
    print(f"\n>>> Agent asks: {question}")
    return input("Your answer: ")


class WhatsAppAgent(SorcarAgent):
    def run(  # type: ignore[override]
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        printer: Any = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        headless: bool | None = None,
        verbose: bool | None = None,
        current_editor_file: str | None = None,
        attachments: list | None = None,
        wait_for_user_callback: Callable[[str, str], None] | None = None,
        ask_user_question_callback: Callable[[str], str] | None = None,
    ) -> str:
        """Run the WhatsApp agent with optional user-interaction callbacks."""
        return super().run(
            model_name=model_name,
            prompt_template=prompt_template,
            arguments=arguments,
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=work_dir,
            printer=printer,
            max_sub_sessions=max_sub_sessions,
            docker_image=docker_image,
            headless=headless,
            verbose=verbose,
            current_editor_file=current_editor_file,
            attachments=attachments,
            wait_for_user_callback=wait_for_user_callback,
            ask_user_question_callback=ask_user_question_callback,
        )

    """SorcarAgent extended with WhatsApp Business Cloud API tools.

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
            headless=False,
        )
    """

    def __init__(self) -> None:
        super().__init__("WhatsApp Agent")
        self._whatsapp_config: dict[str, str] | None = _load_config()

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + WhatsApp auth tools + WhatsApp API tools."""
        tools = super()._get_tools()
        agent = self

        def check_whatsapp_auth() -> str:
            """Check if WhatsApp Business API credentials are configured.

            Tests the stored credentials against the Meta Graph API.

            Returns:
                Authentication status with phone number info, or
                instructions for how to authenticate.
            """
            if agent._whatsapp_config is None:
                return (
                    "Not authenticated with WhatsApp. Use "
                    "authenticate_whatsapp(access_token=..., phone_number_id=...) "
                    "to configure. To get these values:\n"
                    "1. Go to https://developers.facebook.com/apps/\n"
                    "2. Create or select a Business app with WhatsApp product\n"
                    "3. Under WhatsApp > API Setup, find:\n"
                    "   - Temporary access token (or create a System User token "
                    "for permanent access)\n"
                    "   - Phone number ID (shown under 'From' phone number)\n"
                    "4. Call authenticate_whatsapp(access_token='...', "
                    "phone_number_id='...')"
                )
            token = agent._whatsapp_config["access_token"]
            pn_id = agent._whatsapp_config["phone_number_id"]
            url = f"{_GRAPH_API_BASE}/{pn_id}?fields=verified_name,display_phone_number"
            result = _api_request("GET", url, token)
            if "error" in result:
                return json.dumps({"ok": False, "error": result["error"]})
            return json.dumps({  # pragma: no cover – needs live API
                "ok": True,
                "phone_number_id": pn_id,
                "verified_name": result.get("verified_name", ""),
                "display_phone_number": result.get("display_phone_number", ""),
            })

        def authenticate_whatsapp(
            access_token: str, phone_number_id: str, waba_id: str = "",
        ) -> str:
            """Store and validate WhatsApp Business API credentials.

            Saves the credentials to ~/.kiss/channels/whatsapp/config.json
            and validates them against the Meta Graph API.

            Args:
                access_token: Meta Graph API access token. Get it from
                    https://developers.facebook.com/apps/ > your app >
                    WhatsApp > API Setup.
                phone_number_id: WhatsApp Business phone number ID.
                    Shown under the 'From' phone number in API Setup.
                waba_id: WhatsApp Business Account ID (optional, needed
                    for listing message templates). Found in WhatsApp >
                    API Setup or Business Settings.

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
                return json.dumps({
                    "ok": False,
                    "error": f"Credential validation failed: {result['error']}",
                })
            _save_config(  # pragma: no cover – needs live API
                access_token, phone_number_id, waba_id)
            agent._whatsapp_config = {  # pragma: no cover
                "access_token": access_token,
                "phone_number_id": phone_number_id,
                "waba_id": waba_id.strip(),
            }
            return json.dumps({  # pragma: no cover
                "ok": True,
                "message": "WhatsApp credentials saved and validated.",
                "verified_name": result.get("verified_name", ""),
                "display_phone_number": result.get("display_phone_number", ""),
            })

        def clear_whatsapp_auth() -> str:
            """Clear the stored WhatsApp authentication credentials.

            Returns:
                Status message.
            """
            _clear_config()
            agent._whatsapp_config = None
            return "WhatsApp authentication cleared."

        tools.extend([check_whatsapp_auth, authenticate_whatsapp, clear_whatsapp_auth])

        if agent._whatsapp_config is not None:
            tools.extend(_make_whatsapp_tools(
                agent._whatsapp_config["access_token"],
                agent._whatsapp_config["phone_number_id"],
                agent._whatsapp_config.get("waba_id", ""),
            ))

        return tools


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run the WhatsAppAgent from the command line with a --task argument."""
    import argparse
    import os
    import tempfile
    import time as time_mod

    import yaml

    parser = argparse.ArgumentParser(description="Run WhatsAppAgent on a task")
    parser.add_argument("--task", type=str, required=True, help="Task description for the agent")
    parser.add_argument("--model_name", type=str, default=None, help="LLM model name")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps")
    parser.add_argument("--max_budget", type=float, default=5.0, help="Maximum budget in USD")
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument(
        "--headless",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Run browser headless (true/false)",
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Print output to console (true/false)",
    )
    args = parser.parse_args()

    if args.work_dir is not None:
        work_dir = args.work_dir
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp()

    agent = WhatsAppAgent()
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        result = agent.run(
            prompt_template=args.task,
            model_name=args.model_name,
            max_steps=args.max_steps,
            max_budget=args.max_budget,
            work_dir=work_dir,
            headless=args.headless,
            verbose=args.verbose,
            wait_for_user_callback=_cli_wait_for_user,
            ask_user_question_callback=_cli_ask_user_question,
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    print("FINAL RESULT:")
    result_data = yaml.safe_load(result)
    print("Completed successfully: " + str(result_data["success"]))
    print(result_data["summary"])
    print("Work directory was: " + work_dir)
    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
