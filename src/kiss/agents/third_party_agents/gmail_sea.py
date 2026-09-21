# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Gmail Agent — channel agent with Gmail API tools.

Provides authenticated access to a Gmail account via OAuth2.
Handles authentication (reading token from disk or prompting the user
via the browser), stores the token securely in
``~/.kiss/third_party_agents/gmail/token.json``, and exposes a focused set of
Gmail API tools that give the agent full control over email.

Usage::

    agent = GmailAgent()
    agent.run(prompt_template="List my 5 most recent emails")
"""

from __future__ import annotations

import base64
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, cast

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from kiss.agents.third_party_agents._browser_handoff import portal_handoff
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
    write_private_file,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    CLOUD_CONSOLE_URL,
    RemoteOAuthSession,
    google_consent_steps,
    start_google_consent,
)
from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled
from kiss.core.config import kiss_home

_SCOPES = [
    "https://mail.google.com/",
]


def _gmail_dir() -> Path:
    """Return the Gmail credential directory, honoring ``KISS_HOME``.

    Returns:
        Path to ``$KISS_HOME/third_party_agents/gmail`` (defaults to
        ``~/.kiss/third_party_agents/gmail``).
    """
    return kiss_home() / "third_party_agents" / "gmail"


def _token_path() -> Path:
    """Return the path to the stored Gmail OAuth2 token file.

    Returns:
        Path to ``token.json`` inside :func:`_gmail_dir`.
    """
    return _gmail_dir() / "token.json"


def _credentials_path() -> Path:
    """Return the path to the OAuth2 client credentials file.

    Returns:
        Path to ``credentials.json`` inside :func:`_gmail_dir`.
    """
    return _gmail_dir() / "credentials.json"


def _load_credentials() -> Credentials | None:
    """Load stored OAuth2 credentials from disk.

    In Muse-auth mode (the default) the real token stays in
    the daemon vault and a surrogate-bearing handle is returned instead.

    Returns:
        Valid Credentials object (or a surrogate handle in Muse-auth
        mode), or None if not found or expired.
    """
    if muse_auth_enabled():
        # A leftover legacy token.json (working install upgraded to the
        # Muse-auth default) is migrated into the vault and removed.
        from kiss.agents.third_party_agents.muse_auth.client import mint_surrogate_migrating

        return cast("Credentials | None", mint_surrogate_migrating("gmail", _token_path(), _SCOPES))
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
    """Save OAuth2 credentials to disk atomically with restricted permissions.

    In Muse-auth mode the credential goes straight into the daemon
    vault; no agent-readable ``token.json`` is written (surrogate
    handles are skipped — there is nothing real to persist).

    Args:
        creds: Google OAuth2 Credentials object (or a surrogate handle).
    """
    if muse_auth_enabled():
        from kiss.agents.third_party_agents.muse_auth.client import (
            SurrogateCredentials,
            store_credentials,
        )

        if not isinstance(creds, SurrogateCredentials):
            store_credentials("gmail", creds, list(getattr(creds, "scopes", None) or []))
        return
    write_private_file(_token_path(), creds.to_json())


def _clear_credentials() -> None:
    """Delete the stored Gmail OAuth2 token (legacy file and Muse vault)."""
    path = _token_path()
    if path.exists():
        path.unlink()
    if muse_auth_enabled():
        from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

        clear_credentials("gmail")


def _build_service(creds: Credentials) -> Any:
    """Build a Gmail API service object.

    With a surrogate handle (Muse-auth mode) the service routes every
    API call through the Muse-auth daemon via
    :class:`~kiss.agents.third_party_agents.muse_auth.client.MuseHttp`,
    so this process never signs requests with a real token.

    Args:
        creds: Valid OAuth2 Credentials or a Muse-auth surrogate handle.

    Returns:
        Gmail API service resource.
    """
    from kiss.agents.third_party_agents.muse_auth.client import MuseHttp, SurrogateCredentials

    if isinstance(creds, SurrogateCredentials):
        return build("gmail", "v1", http=MuseHttp("gmail", creds.token), static_discovery=True)
    return build("gmail", "v1", credentials=creds)


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


class GmailChannelBackend(ToolMethodBackend):
    """Channel backend for Gmail.

    Provides email monitoring and sending for the channel poller
    and interactive agent.
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

    def _thread_reply_headers(self, thread_id: str) -> dict[str, str]:
        """Return the From/Subject headers of the newest message in a thread.

        Args:
            thread_id: Gmail thread ID.

        Returns:
            Header name -> value dict, or empty dict if the thread cannot
            be fetched.
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
                    metadataHeaders=["From", "Subject"],
                )
                .execute()
            )
            msgs = thread.get("messages", [])
            if not msgs:  # pragma: no branch
                return {}
            headers = msgs[-1].get("payload", {}).get("headers", [])
            return {h["name"]: h["value"] for h in headers}
        except Exception:
            return {}

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send an email (reply to a thread if thread_ts provided).

        The recipient is ``channel_id`` when it is an email address
        (contains ``"@"``). When replying to a thread (``thread_ts`` given)
        and ``channel_id`` is not an address (e.g. a label ID such as
        ``"INBOX"``), the recipient and subject are resolved from the
        newest message in the thread instead.

        Args:
            channel_id: Recipient email address, or a label ID when replying
                to a thread.
            text: Email body text.
            thread_ts: Gmail thread ID to reply to (optional).

        Raises:
            ValueError: If no recipient email address can be determined.
        """
        assert self._service is not None
        to = channel_id if "@" in channel_id else ""
        subject = "" if thread_ts else "Message from KISS Agent"
        if thread_ts:
            headers = self._thread_reply_headers(thread_ts)
            if not to:
                to = headers.get("From", "")
            orig_subject = headers.get("Subject", "")
            if orig_subject:  # pragma: no branch
                subject = (
                    orig_subject
                    if orig_subject.lower().startswith("re:")
                    else f"Re: {orig_subject}"
                )
        if not to:
            raise ValueError(
                f"Cannot send Gmail message: {channel_id!r} is not an email "
                "address and no recipient could be resolved from the thread."
            )
        msg = MIMEMultipart()
        msg["to"] = to
        if subject:  # pragma: no branch
            msg["subject"] = subject
        msg.attach(MIMEText(text, "plain"))
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        body: dict[str, Any] = {"raw": raw}
        if thread_ts:  # pragma: no branch
            body["threadId"] = thread_ts
        self._service.users().messages().send(userId="me", body=body).execute()

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


class GmailAgent(BaseChannelAgent):
    """Channel agent with Gmail API tools.

    Tasks run on the kiss-web daemon's agent (which supplies bash,
    file editing, and browser automation) with authenticated Gmail API tools for
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

    channel_system_prompt = (
        "\n\n## Gmail Authentication\n"
        "Always call check_gmail_auth() first; if it returns ok, report the "
        "authenticated email and stop — never start an OAuth flow over valid "
        "credentials. If credentials.json is missing, call "
        "start_gmail_browser_setup(), which opens Google Cloud Console for the "
        "user to create an OAuth Desktop-app client; if credentials.json exists, "
        "call authenticate_gmail() directly.\n"
        + google_consent_steps("gmail")
        + " Finish by verifying with check_gmail_auth() and reporting the "
        "authenticated email address."
    )

    def __init__(self) -> None:
        super().__init__("Gmail Agent")
        self._backend = GmailChannelBackend()
        creds = _load_credentials()
        if creds:  # pragma: no branch
            self._backend._service = _build_service(creds)

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
                        "Use ask_user_question() if you need user help with a browser login."
                    )
                return (
                    "Not authenticated with Gmail. Call start_gmail_browser_setup() "
                    "to open Google Cloud Console in the user's default browser so "
                    "they can create OAuth credentials, then call "
                    "authenticate_gmail() to start the OAuth2 consent."
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
            """Start the Gmail OAuth2 consent flow.

            Starts the loopback consent server, opens the Google consent
            page in the user's default browser when this machine has one,
            and returns the auth_url for the user to open by hand
            otherwise. Requires credentials.json to exist at
            ~/.kiss/third_party_agents/gmail/credentials.json. Complete
            the flow with finish_gmail_auth().

            Returns:
                status 'consent_required' with auth_url, browser_opened and
                instructions; instructions when credentials.json is missing;
                or an error message.
            """
            answer = start_google_consent("gmail", "Gmail", _SCOPES)
            if answer is None:
                return (
                    f"credentials.json not found at {_credentials_path()}. "
                    f"Download it from Google Cloud Console ({CLOUD_CONSOLE_URL}) > "
                    "OAuth 2.0 Client IDs > Download JSON, then save it to "
                    f"{_credentials_path()}, or call start_gmail_browser_setup()."
                )
            return answer

        def clear_gmail_auth() -> str:
            """Clear the stored Gmail authentication credentials.

            Returns:
                Status message.
            """
            _clear_credentials()
            agent._backend._service = None
            return "Gmail authentication cleared."

        def start_gmail_browser_setup() -> str:
            """Open Google Cloud Console for the user to create Gmail OAuth credentials.

            Opens the Credentials page in the user's default browser when
            this machine has one and returns the steps to relay to the
            user with ask_user_question(). Do not drive Google Cloud
            Console or any Google sign-in page with your built-in browser,
            and never ask for the user's Google password or 2FA code.

            Returns:
                The console URL and step-by-step instructions for the user.
            """
            return (
                f"The user creates the OAuth client themselves. "
                f"{portal_handoff(CLOUD_CONSOLE_URL)} Ask them to: 1. Create or "
                "select a project. 2. Enable the Gmail API (APIs & Services > "
                "Enable APIs). 3. Credentials > Create Credentials > OAuth client "
                "ID > Desktop app. 4. Download the JSON and either paste its "
                f"content back or save it to {_credentials_path()}. Write pasted "
                "content to that path yourself, then call authenticate_gmail() to "
                "start the OAuth consent. Do not drive Google Cloud Console or any "
                "Google sign-in page with your built-in browser, and never ask for "
                "the user's Google password or 2FA code."
            )

        def finish_gmail_auth() -> str:
            """Complete the Gmail OAuth consent started by authenticate_gmail().

            Call after the user has approved consent in their own browser
            (and, when they did so on another machine, after the pasted
            redirect URL has been delivered to the local consent server
            with ``curl -s '<pasted redirect URL>'``).

            Returns:
                Authentication result, a pending status when consent is not
                finished, or an error message.
            """
            creds, status = RemoteOAuthSession.finish("gmail", _SCOPES)
            if status == "pending":
                return json.dumps(
                    {
                        "ok": False,
                        "status": "pending",
                        "error": "Consent is not completed yet; finish the flow in "
                                 "the browser, then call this tool again.",
                    }
                )
            if creds is None:
                return json.dumps({"ok": False, "error": f"OAuth flow failed: {status}"})
            agent._backend._service = _build_service(creds)
            return json.dumps({"ok": True, "message": "Gmail authentication successful."})

        return [
            check_gmail_auth,
            authenticate_gmail,
            clear_gmail_auth,
            start_gmail_browser_setup,
            finish_gmail_auth,
        ]


def main() -> None:
    """Run the GmailAgent from the command line with chat persistence."""
    channel_main(GmailAgent, "kiss-gmail")


def tools() -> list:
    """Return the Gmail channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return GmailAgent()._get_tools()


if __name__ == "__main__":
    main()
