"""Gmail Agent — StatefulSorcarAgent extension with Gmail API tools.

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
import sys
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, cast

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import is_headless_environment, wait_for_matching_message
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
)

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
    if creds.valid:  # pragma: no branch
        return creds
    if creds.expired and creds.refresh_token:  # pragma: no branch
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
    if sys.platform != "win32":  # pragma: no branch
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

    In headless/Docker environments, falls back to ``run_console()`` which
    prints a URL and reads the auth code from stdin instead of opening a
    browser window.

    Returns:
        New Credentials object, or None if credentials.json not found.
    """
    creds_path = _credentials_path()
    if not creds_path.exists():  # pragma: no branch
        return None
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), _SCOPES)
    if is_headless_environment():  # pragma: no branch
        creds = cast(Credentials, flow.run_console())
    else:
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
# Body extraction helpers
# ---------------------------------------------------------------------------


def _extract_body(payload: dict) -> str:  # type: ignore[type-arg]
    """Extract plain text body from a Gmail message payload.

    Args:
        payload: The message payload dict from the Gmail API.

    Returns:
        Decoded plain text body, or empty string.
    """
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    if payload.get("mimeType") == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/html":
            data = part.get("body", {}).get("data", "")
            if data:  # pragma: no branch
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        for subpart in part.get("parts", []):
            if subpart.get("mimeType") in ("text/plain", "text/html"):  # pragma: no branch
                data = subpart.get("body", {}).get("data", "")
                if data:  # pragma: no branch
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    return ""


def _extract_attachments(payload: dict) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    """Extract attachment metadata from a Gmail message payload.

    Args:
        payload: The message payload dict from the Gmail API.

    Returns:
        List of dicts with filename, mimeType, size, and attachmentId.
    """
    attachments: list[dict[str, Any]] = []
    for part in payload.get("parts", []):
        if part.get("filename"):
            attachments.append(
                {
                    "filename": part["filename"],
                    "mime_type": part.get("mimeType", ""),
                    "size": part.get("body", {}).get("size", 0),
                    "attachment_id": part.get("body", {}).get("attachmentId", ""),
                }
            )
        for subpart in part.get("parts", []):
            if subpart.get("filename"):  # pragma: no branch
                attachments.append(
                    {
                        "filename": subpart["filename"],
                        "mime_type": subpart.get("mimeType", ""),
                        "size": subpart.get("body", {}).get("size", 0),
                        "attachment_id": subpart.get("body", {}).get("attachmentId", ""),
                    }
                )
    return attachments


# ---------------------------------------------------------------------------
# GmailChannelBackend
# ---------------------------------------------------------------------------


class GmailChannelBackend(ToolMethodBackend):
    """Channel backend for Gmail.

    Provides email monitoring, sending, and reply waiting for
    the channel poller and interactive agent.
    """

    def __init__(self) -> None:
        self._service: Any = None
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Gmail using stored OAuth2 credentials.

        Returns:
            True on success, False on failure.
        """
        creds = _load_credentials()
        if not creds:  # pragma: no branch
            self._connection_info = "No Gmail credentials found. Please authenticate first."
            return False
        self._service = _build_service(creds)
        try:
            profile = self._service.users().getProfile(userId="me").execute()
            self._connection_info = f"Authenticated as {profile.get('emailAddress', '')}"
            return True
        except Exception as e:
            self._connection_info = f"Gmail auth failed: {e}"
            return False

    def find_channel(self, name: str) -> str | None:
        """Find a Gmail label by name (used as channel ID).

        Args:
            name: Label name to search for.

        Returns:
            Label ID string, or None if not found.
        """
        assert self._service is not None
        try:
            resp = self._service.users().labels().list(userId="me").execute()
            for lbl in resp.get("labels", []):  # pragma: no branch
                if lbl.get("name") == name:  # pragma: no branch
                    return str(lbl["id"])
        except Exception:
            pass
        return None

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll Gmail inbox for new messages.

        Args:
            channel_id: Label ID to poll (use "INBOX" for inbox).
            oldest: History ID or timestamp string for incremental polling.
            limit: Maximum messages to return.

        Returns:
            Tuple of (messages, updated_oldest). Each message dict has:
            ts (date), user (from address), text (body).
        """
        assert self._service is not None
        try:
            kwargs: dict[str, Any] = {
                "userId": "me",
                "maxResults": limit,
                "q": "is:unread",
            }
            if channel_id and channel_id != "INBOX":  # pragma: no branch
                kwargs["labelIds"] = [channel_id]
            else:
                kwargs["labelIds"] = ["INBOX"]
            resp = self._service.users().messages().list(**kwargs).execute()
            messages = []
            for stub in resp.get("messages", []):  # pragma: no branch
                try:
                    msg = (
                        self._service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=stub["id"],
                            format="metadata",
                            metadataHeaders=["Subject", "From", "Date"],
                        )
                        .execute()
                    )
                    headers = {
                        h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])
                    }
                    messages.append(
                        {
                            "ts": headers.get("Date", ""),
                            "user": headers.get("From", ""),
                            "text": (
                                f"Subject: {headers.get('Subject', '')} | {msg.get('snippet', '')}"
                            ),
                            "id": stub["id"],
                            "thread_id": msg.get("threadId", ""),
                        }
                    )
                except Exception:
                    pass
            return messages, oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send an email (reply to a thread if thread_ts provided).

        Args:
            channel_id: Recipient email address.
            text: Email body text.
            thread_ts: Thread ID to reply to (optional).
        """
        assert self._service is not None
        msg = MIMEMultipart()
        msg["to"] = channel_id
        msg["subject"] = "Re: " if thread_ts else "Message from KISS Agent"
        msg.attach(MIMEText(text, "plain"))
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        body: dict[str, Any] = {"raw": raw}
        if thread_ts:  # pragma: no branch
            body["threadId"] = thread_ts
        self._service.users().messages().send(userId="me", body=body).execute()

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
        stop_event: threading.Event | None = None,
    ) -> str | None:
        """Poll a Gmail thread for a reply from a specific user.

        Args:
            channel_id: Label ID (unused for Gmail).
            thread_ts: Thread ID to poll.
            user_id: Email address of expected sender.

        Returns:
            The text of the user's reply.
        """
        assert self._service is not None
        seen: set[str] = set()

        def poll() -> list[dict[str, Any]]:
            try:
                thread = (
                    self._service.users()
                    .threads()
                    .get(
                        userId="me",
                        id=thread_ts,
                        format="metadata",
                        metadataHeaders=["From"],
                    )
                    .execute()
                )
            except Exception:
                return []
            messages: list[dict[str, Any]] = []
            for msg in thread.get("messages", []):  # pragma: no branch
                msg_id = msg["id"]
                if msg_id in seen:  # pragma: no branch
                    continue
                seen.add(msg_id)
                messages.append(msg)
            return messages

        return wait_for_matching_message(
            poll=poll,
            matches=lambda msg: (
                user_id.lower()
                in {
                    h["value"].lower()
                    for h in msg.get("payload", {}).get("headers", [])
                    if h.get("name") == "From"
                }
                or user_id.lower() in str(msg.get("payload", {})).lower()
            ),
            extract_text=lambda msg: str(msg.get("snippet", "")),
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
            poll_interval=5.0,
        )

    # -------------------------------------------------------------------
    # Gmail API tool methods (return JSON strings for LLM agent use)
    # -------------------------------------------------------------------

    def get_profile(self) -> str:
        """Get the current user's Gmail profile.

        Returns:
            JSON string with email address, messages total, threads total,
            and history ID.
        """
        assert self._service is not None
        try:
            profile = self._service.users().getProfile(userId="me").execute()
            return json.dumps(
                {
                    "ok": True,
                    "email": profile.get("emailAddress", ""),
                    "messages_total": profile.get("messagesTotal", 0),
                    "threads_total": profile.get("threadsTotal", 0),
                    "history_id": profile.get("historyId", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_messages(
        self,
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
        assert self._service is not None
        try:
            kwargs: dict[str, Any] = {
                "userId": "me",
                "maxResults": min(max_results, 500),
            }
            if query:  # pragma: no branch
                kwargs["q"] = query
            if page_token:  # pragma: no branch
                kwargs["pageToken"] = page_token
            if label_ids:  # pragma: no branch
                kwargs["labelIds"] = [lid.strip() for lid in label_ids.split(",")]
            resp = self._service.users().messages().list(**kwargs).execute()
            messages = []
            for msg_stub in resp.get("messages", [])[:max_results]:  # pragma: no branch
                try:
                    msg = (
                        self._service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=msg_stub["id"],
                            format="metadata",
                            metadataHeaders=["Subject", "From", "To", "Date"],
                        )
                        .execute()
                    )
                    headers = {
                        h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])
                    }
                    messages.append(
                        {
                            "id": msg["id"],
                            "thread_id": msg.get("threadId", ""),
                            "snippet": msg.get("snippet", ""),
                            "subject": headers.get("Subject", ""),
                            "from": headers.get("From", ""),
                            "to": headers.get("To", ""),
                            "date": headers.get("Date", ""),
                            "label_ids": msg.get("labelIds", []),
                        }
                    )
                except Exception:
                    messages.append({"id": msg_stub["id"], "error": "failed to fetch"})
            result: dict[str, Any] = {"ok": True, "messages": messages}
            next_page = resp.get("nextPageToken", "")
            if next_page:  # pragma: no branch
                result["next_page_token"] = next_page
            result["result_size_estimate"] = resp.get("resultSizeEstimate", 0)
            return json.dumps(result, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_message(self, message_id: str, format: str = "full") -> str:
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
        assert self._service is not None
        try:
            msg = (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format=format)
                .execute()
            )
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            body_text = _extract_body(msg.get("payload", {}))
            attachments = _extract_attachments(msg.get("payload", {}))
            return json.dumps(
                {
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
                },
                indent=2,
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
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
        assert self._service is not None
        try:
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            if cc:  # pragma: no branch
                message["cc"] = cc
            if bcc:  # pragma: no branch
                message["bcc"] = bcc
            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            result = self._service.users().messages().send(userId="me", body={"raw": raw}).execute()
            return json.dumps(
                {
                    "ok": True,
                    "id": result.get("id", ""),
                    "thread_id": result.get("threadId", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def reply_to_message(
        self,
        message_id: str,
        body: str,
        reply_all: bool = False,
        html: bool = False,
    ) -> str:
        """Reply to an existing email message.

        Args:
            message_id: ID of the message to reply to.
            body: Reply body text (plain text or HTML).
            reply_all: If True, reply to all recipients. Default: False.
            html: If True, body is treated as HTML. Default: False.

        Returns:
            JSON string with ok status and the reply message ID.
        """
        assert self._service is not None
        try:
            orig = (
                self._service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=["Subject", "From", "To", "Cc", "Message-ID"],
                )
                .execute()
            )
            headers = {h["name"]: h["value"] for h in orig.get("payload", {}).get("headers", [])}
            thread_id = orig.get("threadId", "")
            subject = headers.get("Subject", "")
            if not subject.lower().startswith("re:"):  # pragma: no branch
                subject = f"Re: {subject}"

            to = headers.get("From", "")
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            message["In-Reply-To"] = headers.get("Message-ID", "")
            message["References"] = headers.get("Message-ID", "")
            if reply_all:  # pragma: no branch
                orig_to = headers.get("To", "")
                orig_cc = headers.get("Cc", "")
                all_recipients = [r.strip() for r in f"{orig_to},{orig_cc}".split(",") if r.strip()]
                message["cc"] = ", ".join(all_recipients)

            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            result = (
                self._service.users()
                .messages()
                .send(userId="me", body={"raw": raw, "threadId": thread_id})
                .execute()
            )
            return json.dumps(
                {
                    "ok": True,
                    "id": result.get("id", ""),
                    "thread_id": result.get("threadId", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
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
        assert self._service is not None
        try:
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            if cc:  # pragma: no branch
                message["cc"] = cc
            if bcc:  # pragma: no branch
                message["bcc"] = bcc
            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            draft = (
                self._service.users()
                .drafts()
                .create(userId="me", body={"message": {"raw": raw}})
                .execute()
            )
            return json.dumps(
                {
                    "ok": True,
                    "draft_id": draft.get("id", ""),
                    "message_id": draft.get("message", {}).get("id", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def trash_message(self, message_id: str) -> str:
        """Move a message to the trash.

        Args:
            message_id: ID of the message to trash.

        Returns:
            JSON string with ok status.
        """
        assert self._service is not None
        try:
            self._service.users().messages().trash(userId="me", id=message_id).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def untrash_message(self, message_id: str) -> str:
        """Remove a message from the trash.

        Args:
            message_id: ID of the message to untrash.

        Returns:
            JSON string with ok status.
        """
        assert self._service is not None
        try:
            self._service.users().messages().untrash(userId="me", id=message_id).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(self, message_id: str) -> str:
        """Permanently delete a message (cannot be undone).

        Args:
            message_id: ID of the message to permanently delete.

        Returns:
            JSON string with ok status.
        """
        assert self._service is not None
        try:
            self._service.users().messages().delete(userId="me", id=message_id).execute()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def modify_labels(
        self,
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
        assert self._service is not None
        try:
            body: dict[str, Any] = {}
            if add_label_ids:  # pragma: no branch
                body["addLabelIds"] = [lid.strip() for lid in add_label_ids.split(",")]
            if remove_label_ids:  # pragma: no branch
                body["removeLabelIds"] = [lid.strip() for lid in remove_label_ids.split(",")]
            result = (
                self._service.users()
                .messages()
                .modify(userId="me", id=message_id, body=body)
                .execute()
            )
            return json.dumps(
                {
                    "ok": True,
                    "id": result.get("id", ""),
                    "label_ids": result.get("labelIds", []),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_labels(self) -> str:
        """List all labels in the user's mailbox.

        Returns:
            JSON string with label list (id, name, type).
        """
        assert self._service is not None
        try:
            resp = self._service.users().labels().list(userId="me").execute()
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

    def create_label(self, name: str, text_color: str = "", background_color: str = "") -> str:
        """Create a new label.

        Args:
            name: Label name (e.g. "Projects/Important").
                Use "/" for nested labels.
            text_color: Optional hex text color (e.g. "#000000").
            background_color: Optional hex background color (e.g. "#16a765").

        Returns:
            JSON string with the new label's id and name.
        """
        assert self._service is not None
        try:
            body: dict[str, Any] = {
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            }
            if text_color and background_color:  # pragma: no branch
                body["color"] = {
                    "textColor": text_color,
                    "backgroundColor": background_color,
                }
            result = self._service.users().labels().create(userId="me", body=body).execute()
            return json.dumps(
                {
                    "ok": True,
                    "id": result.get("id", ""),
                    "name": result.get("name", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_attachment(self, message_id: str, attachment_id: str) -> str:
        """Download a message attachment.

        Args:
            message_id: ID of the message containing the attachment.
            attachment_id: Attachment ID (from get_message response).

        Returns:
            JSON string with base64-encoded attachment data and size.
        """
        assert self._service is not None
        try:
            result = (
                self._service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=message_id, id=attachment_id)
                .execute()
            )
            return json.dumps(
                {
                    "ok": True,
                    "data": result.get("data", "")[:4000],
                    "size": result.get("size", 0),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_thread(self, thread_id: str) -> str:
        """Get all messages in an email thread/conversation.

        Args:
            thread_id: Thread ID (from list_messages or get_message).

        Returns:
            JSON string with all messages in the thread.
        """
        assert self._service is not None
        try:
            thread = (
                self._service.users()
                .threads()
                .get(
                    userId="me",
                    id=thread_id,
                    format="metadata",
                    metadataHeaders=["Subject", "From", "To", "Date"],
                )
                .execute()
            )
            messages = []
            for msg in thread.get("messages", []):  # pragma: no branch
                headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                messages.append(
                    {
                        "id": msg["id"],
                        "snippet": msg.get("snippet", ""),
                        "subject": headers.get("Subject", ""),
                        "from": headers.get("From", ""),
                        "to": headers.get("To", ""),
                        "date": headers.get("Date", ""),
                        "label_ids": msg.get("labelIds", []),
                    }
                )
            return json.dumps(
                {
                    "ok": True,
                    "thread_id": thread.get("id", ""),
                    "messages": messages,
                },
                indent=2,
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


# ---------------------------------------------------------------------------
# GmailAgent
# ---------------------------------------------------------------------------


class GmailAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Gmail API tools.

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
        )
    """

    def __init__(self) -> None:
        super().__init__("Gmail Agent")
        self._backend = GmailChannelBackend()
        creds = _load_credentials()
        if creds:  # pragma: no branch
            self._backend._service = _build_service(creds)

    def run(self, **kwargs: Any) -> str:  # type: ignore[override]
        """Run with Gmail-specific system prompt encouraging browser-based auth."""
        channel_prompt = (
            "\n\n## Gmail Authentication\n"
            "If credentials.json is missing, call start_gmail_browser_setup() to open "
            "Google Cloud Console, then use browser tools to create OAuth credentials "
            "autonomously. If credentials.json exists, call authenticate_gmail() directly. "
            "Use ask_user_browser_action() for any Google account login screens. "
            "Do NOT instruct the user to do these steps manually."
        )
        kwargs["system_prompt"] = (kwargs.get("system_prompt") or "") + channel_prompt
        return super().run(**kwargs)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._service is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_gmail_auth() -> str:
            """Check if Gmail OAuth2 credentials are configured and valid.

            Tests the stored credentials against the Gmail API.

            Returns:
                Authentication status with email address, or instructions
                for how to authenticate.
            """
            if agent._backend._service is None:
                creds_exist = _credentials_path().exists()
                if creds_exist:
                    return (
                        "Not authenticated with Gmail. A credentials.json file exists. "
                        "Call authenticate_gmail() to start the OAuth2 flow. "
                        "Use ask_user_browser_action() if a browser login is required."
                    )
                return (
                    "Not authenticated with Gmail. Call start_gmail_browser_setup() "
                    "to open Google Cloud Console in the browser and create OAuth "
                    "credentials autonomously, then call authenticate_gmail() to "
                    "complete the OAuth2 flow."
                )
            try:
                profile = agent._backend._service.users().getProfile(userId="me").execute()
                return json.dumps(
                    {
                        "ok": True,
                        "email": profile.get("emailAddress", ""),
                        "messages_total": profile.get("messagesTotal", 0),
                    }
                )
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
            if creds is None:  # pragma: no branch
                return (
                    f"credentials.json not found at {_credentials_path()}. "
                    "Download it from Google Cloud Console > APIs & Services > "
                    "Credentials > OAuth 2.0 Client IDs > Download JSON, "
                    f"then save it to {_credentials_path()}"
                )
            agent._backend._service = _build_service(creds)
            try:
                profile = agent._backend._service.users().getProfile(userId="me").execute()
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Gmail authentication successful.",
                        "email": profile.get("emailAddress", ""),
                    }
                )
            except Exception as e:
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Gmail token saved. Could not verify profile.",
                        "error": str(e),
                    }
                )

        def clear_gmail_auth() -> str:
            """Clear the stored Gmail authentication credentials.

            Returns:
                Status message.
            """
            _clear_credentials()
            agent._backend._service = None
            return "Gmail authentication cleared."

        def start_gmail_browser_setup() -> str:
            """Begin automated Gmail API credential setup via browser.

            Navigates to Google Cloud Console. Use your browser tools
            (go_to_url, click, type_text) to complete the following steps autonomously:
            1. Create or select a project.
            2. Enable the Gmail API (APIs & Services > Enable APIs > search "Gmail API").
            3. Go to Credentials > Create Credentials > OAuth client ID.
            4. Choose "Desktop app" as the application type, give it a name.
            5. Download the JSON file and save it to:
               ~/.kiss/channels/gmail/credentials.json
            6. Call authenticate_gmail() to complete the OAuth consent flow.
            Use ask_user_browser_action() for any Google account login screens.

            Returns:
                Page content of Google Cloud Console to begin navigation.
            """
            if agent.web_use_tool is None:  # pragma: no branch
                return (
                    "Browser not available. Manually download credentials.json from "
                    "https://console.cloud.google.com/apis/credentials and save it to "
                    f"{_credentials_path()}, then call authenticate_gmail()."
                )
            return agent.web_use_tool.go_to_url("https://console.cloud.google.com/apis/credentials")

        return [
            check_gmail_auth,
            authenticate_gmail,
            clear_gmail_auth,
            start_gmail_browser_setup,
        ]


def main() -> None:
    """Run the GmailAgent from the command line with chat persistence."""
    channel_main(GmailAgent, "kiss-gmail")


if __name__ == "__main__":
    main()
