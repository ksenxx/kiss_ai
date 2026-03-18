"""Gmail Agent — SorcarAgent extension with Gmail API tools.

Provides authenticated access to a Gmail account via OAuth2.
Handles authentication (reading token from disk or prompting the user
via the browser), stores the token securely in
``~/.kiss/channels/gmail/token.json``, and exposes a focused set of
Gmail API tools that give the agent full control over email.

Usage::

    agent = GmailAgent()
    agent.run(prompt_template="List my 5 most recent emails")
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, cast

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from kiss.agents.sorcar.sorcar_agent import SorcarAgent

logger = logging.getLogger(__name__)

_GMAIL_DIR = Path.home() / ".kiss" / "channels" / "gmail"
_SCOPES = [
    "https://mail.google.com/",
]


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def _token_path() -> Path:
    """Return the path to the stored Gmail OAuth2 token file.

    Returns:
        Path to ``~/.kiss/channels/gmail/token.json``.
    """
    return _GMAIL_DIR / "token.json"


def _credentials_path() -> Path:
    """Return the path to the OAuth2 client credentials file.

    Returns:
        Path to ``~/.kiss/channels/gmail/credentials.json``.
    """
    return _GMAIL_DIR / "credentials.json"


def _load_credentials() -> Credentials | None:
    """Load stored OAuth2 credentials from disk.

    Returns:
        Valid Credentials object, or None if not found or expired.
    """
    path = _token_path()
    if not path.exists():
        return None
    try:
        creds: Credentials = Credentials.from_authorized_user_file(str(path), _SCOPES)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds)
            return creds
        except Exception:
            return None
    return None


def _save_credentials(creds: Credentials) -> None:
    """Save OAuth2 credentials to disk with restricted permissions.

    Args:
        creds: Google OAuth2 Credentials object.
    """
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds.to_json())
    path.chmod(0o600)


def _clear_credentials() -> None:
    """Delete the stored Gmail OAuth2 token."""
    path = _token_path()
    if path.exists():
        path.unlink()


def _run_oauth_flow() -> Credentials | None:
    """Run the OAuth2 installed-app flow to get new credentials.

    Requires ``~/.kiss/channels/gmail/credentials.json`` to exist
    (downloaded from Google Cloud Console).

    Returns:
        New Credentials object, or None if credentials.json not found.
    """
    creds_path = _credentials_path()
    if not creds_path.exists():
        return None
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), _SCOPES)
    creds = cast(Credentials, flow.run_local_server(port=0))
    _save_credentials(creds)
    return creds


def _build_service(creds: Credentials) -> Any:
    """Build a Gmail API service object.

    Args:
        creds: Valid OAuth2 Credentials.

    Returns:
        Gmail API service resource.
    """
    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Gmail API tool functions
# ---------------------------------------------------------------------------


def _make_gmail_tools(service: Any) -> list:
    """Create Gmail API tool functions bound to the given service.

    Args:
        service: Authenticated Gmail API service resource.

    Returns:
        List of callable tool functions for Gmail operations.
    """

    def get_profile() -> str:
        """Get the current user's Gmail profile.

        Returns:
            JSON string with email address, messages total, threads total,
            and history ID.
        """
        try:
            profile = service.users().getProfile(userId="me").execute()
            return json.dumps({
                "ok": True,
                "email": profile.get("emailAddress", ""),
                "messages_total": profile.get("messagesTotal", 0),
                "threads_total": profile.get("threadsTotal", 0),
                "history_id": profile.get("historyId", ""),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_messages(
        query: str = "",
        max_results: int = 20,
        page_token: str = "",
        label_ids: str = "",
    ) -> str:
        """List messages in the user's mailbox.

        Args:
            query: Gmail search query (same syntax as Gmail search box).
                Examples: "is:unread", "from:alice@example.com",
                "subject:meeting", "newer_than:1d", "has:attachment".
            max_results: Maximum number of messages to return (1-500).
                Default: 20.
            page_token: Page token for pagination from a previous response.
            label_ids: Comma-separated label IDs to filter by
                (e.g. "INBOX", "UNREAD", "STARRED").

        Returns:
            JSON string with message IDs, snippet, and pagination token.
            Use get_message() with the ID to read full content.
        """
        try:
            kwargs: dict[str, Any] = {
                "userId": "me",
                "maxResults": min(max_results, 500),
            }
            if query:
                kwargs["q"] = query
            if page_token:
                kwargs["pageToken"] = page_token
            if label_ids:
                kwargs["labelIds"] = [lid.strip() for lid in label_ids.split(",")]
            resp = service.users().messages().list(**kwargs).execute()
            messages = resp.get("messages", [])
            # Fetch snippets for each message
            results = []
            for msg_stub in messages[:max_results]:
                try:
                    msg = (
                        service.users()
                        .messages()
                        .get(userId="me", id=msg_stub["id"], format="metadata",
                             metadataHeaders=["Subject", "From", "To", "Date"])
                        .execute()
                    )
                    headers = {
                        h["name"]: h["value"]
                        for h in msg.get("payload", {}).get("headers", [])
                    }
                    results.append({
                        "id": msg["id"],
                        "thread_id": msg.get("threadId", ""),
                        "snippet": msg.get("snippet", ""),
                        "subject": headers.get("Subject", ""),
                        "from": headers.get("From", ""),
                        "to": headers.get("To", ""),
                        "date": headers.get("Date", ""),
                        "label_ids": msg.get("labelIds", []),
                    })
                except Exception:
                    results.append({"id": msg_stub["id"], "error": "failed to fetch"})
            result: dict[str, Any] = {"ok": True, "messages": results}
            next_page = resp.get("nextPageToken", "")
            if next_page:
                result["next_page_token"] = next_page
            result["result_size_estimate"] = resp.get("resultSizeEstimate", 0)
            return json.dumps(result, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_message(message_id: str, format: str = "full") -> str:
        """Get a specific message by ID.

        Args:
            message_id: The message ID (from list_messages).
            format: Response format. Options:
                "full" — full message with parsed payload (default).
                "metadata" — headers only (faster).
                "raw" — raw RFC 2822 message.
                "minimal" — just IDs, labels, snippet.

        Returns:
            JSON string with message headers, body text, labels, and
            attachment info.
        """
        try:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format=format)
                .execute()
            )
            headers = {
                h["name"]: h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            # Extract body text
            body_text = _extract_body(msg.get("payload", {}))
            # Extract attachment info
            attachments = _extract_attachments(msg.get("payload", {}))
            return json.dumps({
                "ok": True,
                "id": msg["id"],
                "thread_id": msg.get("threadId", ""),
                "label_ids": msg.get("labelIds", []),
                "snippet": msg.get("snippet", ""),
                "subject": headers.get("Subject", ""),
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "cc": headers.get("Cc", ""),
                "date": headers.get("Date", ""),
                "body": body_text[:4000],
                "attachments": attachments,
            }, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_message(
        to: str, subject: str, body: str, cc: str = "", bcc: str = "",
        html: bool = False,
    ) -> str:
        """Send an email message.

        Args:
            to: Recipient email address(es), comma-separated.
            subject: Email subject line.
            body: Email body text (plain text or HTML).
            cc: CC recipients, comma-separated. Optional.
            bcc: BCC recipients, comma-separated. Optional.
            html: If True, body is treated as HTML. Default: False.

        Returns:
            JSON string with ok status and the sent message ID.
        """
        try:
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            if bcc:
                message["bcc"] = bcc
            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            result = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw})
                .execute()
            )
            return json.dumps({
                "ok": True,
                "id": result.get("id", ""),
                "thread_id": result.get("threadId", ""),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def reply_to_message(
        message_id: str, body: str, reply_all: bool = False, html: bool = False,
    ) -> str:
        """Reply to an existing email message.

        Fetches the original message to get the thread ID, subject, and
        recipients, then sends a reply in the same thread.

        Args:
            message_id: ID of the message to reply to.
            body: Reply body text (plain text or HTML).
            reply_all: If True, reply to all recipients. Default: False.
            html: If True, body is treated as HTML. Default: False.

        Returns:
            JSON string with ok status and the reply message ID.
        """
        try:
            orig = (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="metadata",
                     metadataHeaders=["Subject", "From", "To", "Cc", "Message-ID"])
                .execute()
            )
            headers = {
                h["name"]: h["value"]
                for h in orig.get("payload", {}).get("headers", [])
            }
            thread_id = orig.get("threadId", "")
            subject = headers.get("Subject", "")
            if not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"

            to = headers.get("From", "")
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            message["In-Reply-To"] = headers.get("Message-ID", "")
            message["References"] = headers.get("Message-ID", "")
            if reply_all:
                orig_to = headers.get("To", "")
                orig_cc = headers.get("Cc", "")
                all_recipients = [r.strip() for r in f"{orig_to},{orig_cc}".split(",") if r.strip()]
                message["cc"] = ", ".join(all_recipients)

            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            result = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw, "threadId": thread_id})
                .execute()
            )
            return json.dumps({
                "ok": True,
                "id": result.get("id", ""),
                "thread_id": result.get("threadId", ""),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_draft(
        to: str, subject: str, body: str, cc: str = "", bcc: str = "",
        html: bool = False,
    ) -> str:
        """Create a draft email.

        Args:
            to: Recipient email address(es), comma-separated.
            subject: Email subject line.
            body: Email body text (plain text or HTML).
            cc: CC recipients, comma-separated. Optional.
            bcc: BCC recipients, comma-separated. Optional.
            html: If True, body is treated as HTML. Default: False.

        Returns:
            JSON string with ok status and draft ID.
        """
        try:
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            if bcc:
                message["bcc"] = bcc
            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            draft = (
                service.users()
                .drafts()
                .create(userId="me", body={"message": {"raw": raw}})
                .execute()
            )
            return json.dumps({
                "ok": True,
                "draft_id": draft.get("id", ""),
                "message_id": draft.get("message", {}).get("id", ""),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def trash_message(message_id: str) -> str:
        """Move a message to the trash.

        Args:
            message_id: ID of the message to trash.

        Returns:
            JSON string with ok status.
        """
        try:
            service.users().messages().trash(userId="me", id=message_id).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def untrash_message(message_id: str) -> str:
        """Remove a message from the trash.

        Args:
            message_id: ID of the message to untrash.

        Returns:
            JSON string with ok status.
        """
        try:
            service.users().messages().untrash(userId="me", id=message_id).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(message_id: str) -> str:
        """Permanently delete a message (cannot be undone).

        Args:
            message_id: ID of the message to permanently delete.

        Returns:
            JSON string with ok status.
        """
        try:
            service.users().messages().delete(userId="me", id=message_id).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def modify_labels(
        message_id: str,
        add_label_ids: str = "",
        remove_label_ids: str = "",
    ) -> str:
        """Modify labels on a message (star, archive, mark read/unread, etc.).

        Common label IDs: INBOX, UNREAD, STARRED, IMPORTANT, SPAM, TRASH,
        CATEGORY_PERSONAL, CATEGORY_SOCIAL, CATEGORY_PROMOTIONS.

        To archive: remove "INBOX".
        To mark as read: remove "UNREAD".
        To star: add "STARRED".

        Args:
            message_id: ID of the message to modify.
            add_label_ids: Comma-separated label IDs to add.
            remove_label_ids: Comma-separated label IDs to remove.

        Returns:
            JSON string with ok status and updated label list.
        """
        try:
            body: dict[str, Any] = {}
            if add_label_ids:
                body["addLabelIds"] = [lid.strip() for lid in add_label_ids.split(",")]
            if remove_label_ids:
                body["removeLabelIds"] = [lid.strip() for lid in remove_label_ids.split(",")]
            result = (
                service.users()
                .messages()
                .modify(userId="me", id=message_id, body=body)
                .execute()
            )
            return json.dumps({
                "ok": True,
                "id": result.get("id", ""),
                "label_ids": result.get("labelIds", []),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_labels() -> str:
        """List all labels in the user's mailbox.

        Returns:
            JSON string with label list (id, name, type).
        """
        try:
            resp = service.users().labels().list(userId="me").execute()
            labels = [
                {
                    "id": lbl.get("id", ""),
                    "name": lbl.get("name", ""),
                    "type": lbl.get("type", ""),
                }
                for lbl in resp.get("labels", [])
            ]
            return json.dumps({"ok": True, "labels": labels}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_label(name: str, text_color: str = "", background_color: str = "") -> str:
        """Create a new label.

        Args:
            name: Label name (e.g. "Projects/Important").
                Use "/" for nested labels.
            text_color: Optional hex text color (e.g. "#000000").
            background_color: Optional hex background color (e.g. "#16a765").

        Returns:
            JSON string with the new label's id and name.
        """
        try:
            body: dict[str, Any] = {
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            }
            if text_color and background_color:
                body["color"] = {
                    "textColor": text_color,
                    "backgroundColor": background_color,
                }
            result = service.users().labels().create(userId="me", body=body).execute()
            return json.dumps({
                "ok": True,
                "id": result.get("id", ""),
                "name": result.get("name", ""),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_attachment(message_id: str, attachment_id: str) -> str:
        """Download a message attachment.

        Args:
            message_id: ID of the message containing the attachment.
            attachment_id: Attachment ID (from get_message response).

        Returns:
            JSON string with base64-encoded attachment data and size.
        """
        try:
            result = (
                service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=message_id, id=attachment_id)
                .execute()
            )
            return json.dumps({
                "ok": True,
                "data": result.get("data", "")[:4000],
                "size": result.get("size", 0),
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_thread(thread_id: str) -> str:
        """Get all messages in an email thread/conversation.

        Args:
            thread_id: Thread ID (from list_messages or get_message).

        Returns:
            JSON string with all messages in the thread.
        """
        try:
            thread = (
                service.users()
                .threads()
                .get(userId="me", id=thread_id, format="metadata",
                     metadataHeaders=["Subject", "From", "To", "Date"])
                .execute()
            )
            messages = []
            for msg in thread.get("messages", []):
                headers = {
                    h["name"]: h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }
                messages.append({
                    "id": msg["id"],
                    "snippet": msg.get("snippet", ""),
                    "subject": headers.get("Subject", ""),
                    "from": headers.get("From", ""),
                    "to": headers.get("To", ""),
                    "date": headers.get("Date", ""),
                    "label_ids": msg.get("labelIds", []),
                })
            return json.dumps({
                "ok": True,
                "thread_id": thread.get("id", ""),
                "messages": messages,
            }, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    return [
        get_profile,
        list_messages,
        get_message,
        send_message,
        reply_to_message,
        create_draft,
        trash_message,
        untrash_message,
        delete_message,
        modify_labels,
        list_labels,
        create_label,
        get_attachment,
        get_thread,
    ]


# ---------------------------------------------------------------------------
# Body extraction helpers
# ---------------------------------------------------------------------------


def _extract_body(payload: dict) -> str:
    """Extract plain text body from a Gmail message payload.

    Args:
        payload: The message payload dict from the Gmail API.

    Returns:
        Decoded plain text body, or empty string.
    """
    # Simple message with body data directly
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    # Multipart: look through parts
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    # Fall back to HTML if no plain text
    if payload.get("mimeType") == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/html":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        # Nested multipart
        for subpart in part.get("parts", []):
            if subpart.get("mimeType") in ("text/plain", "text/html"):
                data = subpart.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    return ""


def _extract_attachments(payload: dict) -> list[dict[str, Any]]:
    """Extract attachment metadata from a Gmail message payload.

    Args:
        payload: The message payload dict from the Gmail API.

    Returns:
        List of dicts with filename, mimeType, size, and attachmentId.
    """
    attachments: list[dict[str, Any]] = []
    for part in payload.get("parts", []):
        if part.get("filename"):
            attachments.append({
                "filename": part["filename"],
                "mime_type": part.get("mimeType", ""),
                "size": part.get("body", {}).get("size", 0),
                "attachment_id": part.get("body", {}).get("attachmentId", ""),
            })
        # Nested multipart
        for subpart in part.get("parts", []):
            if subpart.get("filename"):
                attachments.append({
                    "filename": subpart["filename"],
                    "mime_type": subpart.get("mimeType", ""),
                    "size": subpart.get("body", {}).get("size", 0),
                    "attachment_id": subpart.get("body", {}).get("attachmentId", ""),
                })
    return attachments


# ---------------------------------------------------------------------------
# GmailAgent
# ---------------------------------------------------------------------------


def _cli_wait_for_user(instruction: str, url: str) -> None:
    """CLI callback for browser-action prompts (prints and waits for Enter).

    Args:
        instruction: What the user should do.
        url: Current browser URL (printed if non-empty).
    """
    print(f"\n>>> Browser action needed: {instruction}")
    if url:
        print(f"    Current URL: {url}")
    input("Press Enter when done... ")


def _cli_ask_user_question(question: str) -> str:
    """CLI callback for agent questions (prints and reads from stdin).

    Args:
        question: The question to display to the user.

    Returns:
        The user's typed response text.
    """
    print(f"\n>>> Agent asks: {question}")
    return input("Your answer: ")


class GmailAgent(SorcarAgent):
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
        """Run the Gmail agent with optional user-interaction callbacks."""
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

    """SorcarAgent extended with Gmail API tools.

    Inherits all standard SorcarAgent capabilities (bash, file editing,
    browser automation) and adds authenticated Gmail API tools for
    reading, sending, searching, labeling, and managing email.

    The agent checks for stored OAuth2 credentials on initialization.
    If no valid credentials are found, authentication tools guide the
    user through the OAuth2 flow.

    Example::

        agent = GmailAgent()
        result = agent.run(
            prompt_template="Show my 5 most recent unread emails",
            headless=False,
        )
    """

    def __init__(self) -> None:
        super().__init__("Gmail Agent")
        self._gmail_service: Any = None
        creds = _load_credentials()
        if creds:
            self._gmail_service = _build_service(creds)

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Gmail auth tools + Gmail API tools."""
        tools = super()._get_tools()
        agent = self

        def check_gmail_auth() -> str:
            """Check if Gmail OAuth2 credentials are configured and valid.

            Tests the stored credentials against the Gmail API.

            Returns:
                Authentication status with email address, or instructions
                for how to authenticate.
            """
            if agent._gmail_service is None:
                creds_exist = _credentials_path().exists()
                if creds_exist:
                    return (
                        "Not authenticated with Gmail. A credentials.json file exists. "
                        "Use authenticate_gmail() to start the OAuth2 flow — this will "
                        "open a browser window for you to authorize access."
                    )
                return (
                    "Not authenticated with Gmail. To set up:\n"
                    "1. Go to https://console.cloud.google.com/apis/credentials\n"
                    "2. Create an OAuth 2.0 Client ID (Desktop app type)\n"
                    "3. Download the JSON and save it as:\n"
                    f"   {_credentials_path()}\n"
                    "4. Then call authenticate_gmail() to start the OAuth2 flow."
                )
            try:
                profile = agent._gmail_service.users().getProfile(userId="me").execute()
                return json.dumps({
                    "ok": True,
                    "email": profile.get("emailAddress", ""),
                    "messages_total": profile.get("messagesTotal", 0),
                })
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_gmail() -> str:
            """Start the Gmail OAuth2 authentication flow.

            Opens a browser window for the user to authorize access.
            Requires credentials.json to exist at
            ~/.kiss/channels/gmail/credentials.json.

            Returns:
                Authentication result with email address, or error message.
            """
            creds = _run_oauth_flow()
            if creds is None:
                return (
                    f"credentials.json not found at {_credentials_path()}. "
                    "Download it from Google Cloud Console > APIs & Services > "
                    "Credentials > OAuth 2.0 Client IDs > Download JSON, "
                    f"then save it to {_credentials_path()}"
                )
            agent._gmail_service = _build_service(creds)
            try:
                profile = agent._gmail_service.users().getProfile(userId="me").execute()
                return json.dumps({
                    "ok": True,
                    "message": "Gmail authentication successful.",
                    "email": profile.get("emailAddress", ""),
                })
            except Exception as e:
                return json.dumps({
                    "ok": True,
                    "message": "Gmail token saved. Could not verify profile.",
                    "error": str(e),
                })

        def clear_gmail_auth() -> str:
            """Clear the stored Gmail authentication credentials.

            Returns:
                Status message.
            """
            _clear_credentials()
            agent._gmail_service = None
            return "Gmail authentication cleared."

        tools.extend([check_gmail_auth, authenticate_gmail, clear_gmail_auth])

        if agent._gmail_service is not None:
            tools.extend(_make_gmail_tools(agent._gmail_service))

        return tools


def main() -> None:
    """Run the GmailAgent from the command line with a --task argument."""
    import argparse
    import os
    import tempfile
    import time as time_mod

    import yaml

    parser = argparse.ArgumentParser(description="Run GmailAgent on a task")
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

    agent = GmailAgent()
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
