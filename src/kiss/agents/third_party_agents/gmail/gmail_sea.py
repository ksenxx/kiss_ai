# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Gmail Agent — channel agent with Gmail API tools.

Provides access to a Gmail account through Composio (see
:mod:`._composio_google`): the user connects Gmail with a Composio
Connect Link, and every API call goes through Composio's proxy, which
holds the Google token.  Exposes a focused set of Gmail API tools for
reading, sending, labeling and trashing email.

Usage::

    agent = GmailAgent()
    agent.run(prompt_template="List my 5 most recent emails")
"""

from __future__ import annotations

import base64
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from googleapiclient.discovery import build

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
)
from kiss.agents.third_party_agents._composio_google import ComposioHttp, connected_account_id
from kiss.agents.third_party_agents._google_workspace_utils import (
    google_auth_prompt,
    make_google_auth_tools,
)

_SERVICE = "gmail"


class GmailSea(BaseSea):
    """The ``/gmail`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/gmail help``."""
        return (
            "Channel agent for Gmail that reads, searches, sends, labels and trashes email "
            "through a Composio-connected Google account (the user connects with a Composio "
            "Connect Link and Composio's proxy holds the token); use "
            '`run_agent(agent="gmail", task=...)` or the `kiss-gmail` CLI.'
        )

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return the Gmail channel tools (the SEA ``tools`` method).

        Called by the kiss-web daemon when this module's path is passed as
        the API's ``sea_path``: builds a fresh agent from the
        Composio connection recorded under ``~/.kiss`` and returns its
        authentication and backend tools.
        """
        return tools + GmailAgent()._get_tools()

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Run as a ``channel`` worker (``kiss.server.sorcar.run`` SEA contract).

        No git lifecycle, nothing inherited from the calling task, the
        channel preamble in the system prompt (see
        :mod:`kiss.agents.sorcar.sea_settings`).
        """
        return settings | {"channel": True}

    def system_prompt(self, system_prompt: str) -> str:
        """Return the channel guidance appended to the run's system prompt."""
        return system_prompt + "\n\n" + GmailAgent.channel_system_prompt


def _build_service() -> Any:
    """Build a Gmail API service whose requests go through Composio's proxy.

    Returns:
        Gmail API service resource.
    """
    return build("gmail", "v1", http=ComposioHttp(_SERVICE), static_discovery=True)


def _headers(message: dict) -> dict[str, str]:  # type: ignore[type-arg]
    """Map a message's header names (lower-cased) to their values.

    Header names are case-insensitive and arrive as the sender wrote
    them (``Message-ID`` or ``Message-Id``), so callers look them up
    in lower case.

    Args:
        message: Gmail API message dict with a ``payload.headers`` list.

    Returns:
        Lower-cased header name -> value.
    """
    return {h["name"].lower(): h["value"] for h in message.get("payload", {}).get("headers", [])}


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
        """Connect to Gmail through the recorded Composio connection.

        Returns:
            True on success, False on failure.
        """
        if not connected_account_id(_SERVICE):
            self._connection_info = "Gmail is not connected. Please authenticate first."
            return False
        self._service = _build_service()
        try:
            profile = self._service.users().getProfile(userId="me").execute()
            self._connection_info = f"Authenticated as {profile.get('emailAddress', '')}"
            return True
        except Exception as e:
            self._connection_info = f"Gmail auth failed: {e}"
            return False

    def _thread_reply_headers(self, thread_id: str) -> dict[str, str]:
        """Return the reply headers of the newest message in a thread.

        Args:
            thread_id: Gmail thread ID.

        Returns:
            Header name -> value dict (From, Subject, Message-ID,
            References), or empty dict if the thread cannot be fetched.
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
                    metadataHeaders=["From", "Subject", "Message-ID", "References"],
                )
                .execute()
            )
            msgs = thread.get("messages", [])
            if not msgs:  # pragma: no branch
                return {}
            return _headers(msgs[-1])
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
                to = headers.get("from", "")
            orig_subject = headers.get("subject", "")
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
                    headers = _headers(msg)
                    messages.append(
                        {
                            "id": msg["id"],
                            "thread_id": msg.get("threadId", ""),
                            "snippet": msg.get("snippet", ""),
                            "subject": headers.get("subject", ""),
                            "from": headers.get("from", ""),
                            "to": headers.get("to", ""),
                            "date": headers.get("date", ""),
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
            headers = _headers(msg)
            body_text = _extract_body(msg.get("payload", {}))
            attachments = _extract_attachments(msg.get("payload", {}))
            return json.dumps(
                {
                    "ok": True,
                    "id": msg["id"],
                    "thread_id": msg.get("threadId", ""),
                    "label_ids": msg.get("labelIds", []),
                    "snippet": msg.get("snippet", ""),
                    "subject": headers.get("subject", ""),
                    "from": headers.get("from", ""),
                    "to": headers.get("to", ""),
                    "cc": headers.get("cc", ""),
                    "date": headers.get("date", ""),
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
        draft_only: bool = False,
    ) -> str:
        """Reply to an existing email message, sending it or saving it as a draft.

        Args:
            message_id: ID of the message to reply to.
            body: Reply body text (plain text or HTML).
            reply_all: If True, reply to all recipients. Default: False.
            html: If True, body is treated as HTML. Default: False.
            draft_only: If True, the reply is saved as a draft inside the
                original thread and nothing is sent; the user reviews and
                sends it from Gmail. Use this for automated or unattended
                replies. Default: False.

        Returns:
            JSON string with ok status and the sent reply's ``id`` and
            ``thread_id``, or with ``draft_id``, ``message_id`` and
            ``thread_id`` when ``draft_only`` is True.
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
                    metadataHeaders=["Subject", "From", "To", "Cc", "Message-ID", "References"],
                )
                .execute()
            )
            headers = _headers(orig)
            thread_id = orig.get("threadId", "")
            subject = headers.get("subject", "")
            if not subject.lower().startswith("re:"):  # pragma: no branch
                subject = f"Re: {subject}"

            to = headers.get("from", "")
            orig_id = headers.get("message-id", "")
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            message["In-Reply-To"] = orig_id
            message["References"] = " ".join(
                v for v in (headers.get("references", ""), orig_id) if v
            )
            if reply_all:  # pragma: no branch
                orig_to = headers.get("to", "")
                orig_cc = headers.get("cc", "")
                all_recipients = [r.strip() for r in f"{orig_to},{orig_cc}".split(",") if r.strip()]
                message["cc"] = ", ".join(all_recipients)

            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            if draft_only:
                return json.dumps(self._save_draft(raw, thread_id))
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
        thread_id: str = "",
        in_reply_to: str = "",
        references: str = "",
    ) -> str:
        """Create a draft email, optionally as a reply inside an existing thread.

        To draft a reply that Gmail shows inside the original conversation,
        pass ``thread_id`` (from list_messages/get_message). The
        ``In-Reply-To``/``References`` headers, and ``to``/``subject`` when
        left empty, are then taken from the newest message of that thread,
        so ``create_draft("", "", body, thread_id=...)`` is enough.

        Args:
            to: Recipient email address(es), comma-separated. May be empty
                when ``thread_id`` is given (defaults to the thread's last
                sender).
            subject: Email subject line. May be empty when ``thread_id`` is
                given (defaults to "Re: <thread subject>").
            body: Email body text (plain text or HTML).
            cc: CC recipients, comma-separated. Optional.
            bcc: BCC recipients, comma-separated. Optional.
            html: If True, body is treated as HTML. Default: False.
            thread_id: Gmail thread ID the draft belongs to. Optional.
            in_reply_to: RFC 822 ``Message-ID`` of the message being
                answered (``In-Reply-To`` header). Optional; resolved from
                the thread when ``thread_id`` is given and this is empty.
            references: ``References`` header value. Optional; resolved
                from the thread when ``thread_id`` is given and this is
                empty.

        Returns:
            JSON string with ok status, draft ID, message ID and thread ID.
        """
        assert self._service is not None
        try:
            if thread_id:
                headers = self._thread_reply_headers(thread_id)
                to = to or headers.get("from", "")
                orig_subject = headers.get("subject", "")
                if not subject and orig_subject:
                    subject = (
                        orig_subject
                        if orig_subject.lower().startswith("re:")
                        else f"Re: {orig_subject}"
                    )
                in_reply_to = in_reply_to or headers.get("message-id", "")
                if not references:
                    references = " ".join(
                        v for v in (headers.get("references", ""), in_reply_to) if v
                    )
            if not to:
                raise ValueError("create_draft needs a recipient: pass `to` or a `thread_id`")
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            if bcc:
                message["bcc"] = bcc
            if in_reply_to:
                message["In-Reply-To"] = in_reply_to
            if references:
                message["References"] = references
            subtype = "html" if html else "plain"
            message.attach(MIMEText(body, subtype))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            return json.dumps(self._save_draft(raw, thread_id))
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def _save_draft(self, raw: str, thread_id: str) -> dict[str, Any]:
        """Store a raw RFC 822 message as a Gmail draft, inside ``thread_id`` when given.

        Args:
            raw: URL-safe base64-encoded MIME message.
            thread_id: Gmail thread the draft belongs to; empty for a new thread.

        Returns:
            Result dict with ok status, draft ID, message ID and thread ID.
        """
        assert self._service is not None
        draft_message: dict[str, Any] = {"raw": raw}
        if thread_id:
            draft_message["threadId"] = thread_id
        draft = (
            self._service.users()
            .drafts()
            .create(userId="me", body={"message": draft_message})
            .execute()
        )
        return {
            "ok": True,
            "draft_id": draft.get("id", ""),
            "message_id": draft.get("message", {}).get("id", ""),
            "thread_id": draft.get("message", {}).get("threadId", ""),
        }

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
                headers = _headers(msg)
                messages.append(
                    {
                        "id": msg["id"],
                        "snippet": msg.get("snippet", ""),
                        "subject": headers.get("subject", ""),
                        "from": headers.get("from", ""),
                        "to": headers.get("to", ""),
                        "date": headers.get("date", ""),
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

    When Gmail has no Composio connection yet, the authentication tools
    guide the user through the Composio Connect Link.

    Example::

        agent = GmailAgent()
        result = agent.run(
            prompt_template="Show my 5 most recent unread emails",
        )
    """

    channel_system_prompt = google_auth_prompt(_SERVICE, "Gmail")

    def __init__(self) -> None:
        super().__init__("Gmail Agent")
        self._backend = GmailChannelBackend()
        if connected_account_id(_SERVICE):
            self._backend._service = _build_service()

    def _is_authenticated(self) -> bool:
        """Return True if Gmail is connected through Composio."""
        return bool(connected_account_id(_SERVICE))

    def _get_auth_tools(self) -> list:
        """Return the Composio sign-in tool set for Gmail."""
        return make_google_auth_tools(self, _SERVICE, "Gmail", self._backend.connect)


def main() -> None:
    """Run the GmailAgent from the command line with chat persistence."""
    channel_main(GmailAgent, "kiss-gmail")


if __name__ == "__main__":
    main()
