"""SMS Agent — ChatSorcarAgent extension with Twilio SMS tools.

Provides SMS sending/receiving via Twilio. Stores config in
``~/.kiss/third_party_agents/sms/config.json``.

Usage::

    agent = SMSAgent()
    agent.run(prompt_template="Send 'Hello!' to +14155238886")
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.third_party_agents._backend_utils import wait_for_matching_message
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

_SMS_DIR = Path.home() / ".kiss" / "third_party_agents" / "sms"
_config = ChannelConfig(
    _SMS_DIR,
    (
        "account_sid",
        "auth_token",
    ),
)


class SMSChannelBackend(ToolMethodBackend):
    """Channel backend for Twilio SMS."""

    def __init__(self) -> None:
        self._client: Any = None
        self._from_number: str = ""
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Authenticate with Twilio using stored config."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Twilio config found."
            return False
        try:
            from twilio.rest import Client

            self._client = Client(cfg["account_sid"], cfg["auth_token"])
            self._from_number = cfg.get("from_number", "")
            self._client.api.accounts(cfg["account_sid"]).fetch()
            self._connection_info = f"Authenticated with Twilio account {cfg['account_sid']}"
            return True
        except Exception as e:
            self._connection_info = f"Twilio auth failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll Twilio for recent inbound messages."""
        if not self._client:  # pragma: no branch
            return [], oldest
        try:
            messages_list = self._client.messages.list(to=self._from_number, limit=limit)
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for msg in messages_list:  # pragma: no branch
                ts = str(msg.date_sent.timestamp() if msg.date_sent else "")
                if oldest and ts <= oldest:  # pragma: no branch
                    continue
                new_oldest = ts
                messages.append(
                    {
                        "ts": ts,
                        "user": msg.from_,
                        "text": msg.body,
                        "sid": msg.sid,
                    }
                )
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send an SMS."""
        if self._client:  # pragma: no branch
            self._client.messages.create(to=channel_id, from_=self._from_number, body=text)

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
    ) -> str | None:
        """Poll for a reply from a specific number."""
        oldest = str(time.time())

        def poll() -> list[dict[str, Any]]:
            nonlocal oldest
            msgs, oldest = self.poll_messages(channel_id, oldest)
            return msgs

        return wait_for_matching_message(
            poll=poll,
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=5.0,
        )

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Check if message is from the bot's number."""
        return bool(msg.get("user", "") == self._from_number)

    def send_sms(self, to: str, body: str) -> str:
        """Send an SMS message via Twilio.

        Args:
            to: Recipient phone number in E.164 format.
            body: Message text (up to 1600 characters).

        Returns:
            JSON string with ok status and message SID.
        """
        assert self._client is not None
        try:
            msg = self._client.messages.create(to=to, from_=self._from_number, body=body)
            return json.dumps({"ok": True, "sid": msg.sid, "status": msg.status})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_mms(self, to: str, body: str, media_url: str) -> str:
        """Send an MMS message with media via Twilio.

        Args:
            to: Recipient phone number in E.164 format.
            body: Message text.
            media_url: Publicly accessible URL of the media file.

        Returns:
            JSON string with ok status and message SID.
        """
        assert self._client is not None
        try:
            msg = self._client.messages.create(
                to=to, from_=self._from_number, body=body, media_url=[media_url]
            )
            return json.dumps({"ok": True, "sid": msg.sid})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_messages(
        self,
        to: str = "",
        from_: str = "",
        limit: int = 20,
        page_token: str = "",
    ) -> str:
        """List Twilio messages.

        Args:
            to: Filter by recipient phone number. Optional.
            from_: Filter by sender phone number. Optional.
            limit: Maximum messages to return. Default: 20.
            page_token: Pagination token. Optional.

        Returns:
            JSON string with message list.
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {"limit": limit}
            if to:  # pragma: no branch
                kwargs["to"] = to
            if from_:  # pragma: no branch
                kwargs["from_"] = from_
            messages = self._client.messages.list(**kwargs)
            result = [
                {
                    "sid": m.sid,
                    "from": m.from_,
                    "to": m.to,
                    "body": m.body,
                    "status": m.status,
                    "date_sent": str(m.date_sent) if m.date_sent else "",
                    "direction": m.direction,
                }
                for m in messages
            ]
            return json.dumps({"ok": True, "messages": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_message(self, message_sid: str) -> str:
        """Get details about a specific Twilio message.

        Args:
            message_sid: Message SID (e.g. "SM...").

        Returns:
            JSON string with message details.
        """
        assert self._client is not None
        try:
            msg = self._client.messages(message_sid).fetch()
            return json.dumps(
                {
                    "ok": True,
                    "sid": msg.sid,
                    "from": msg.from_,
                    "to": msg.to,
                    "body": msg.body,
                    "status": msg.status,
                    "date_sent": str(msg.date_sent) if msg.date_sent else "",
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_phone_numbers(self, limit: int = 20) -> str:
        """List Twilio phone numbers on the account.

        Args:
            limit: Maximum numbers to return. Default: 20.

        Returns:
            JSON string with phone number list.
        """
        assert self._client is not None
        try:
            numbers = self._client.incoming_phone_numbers.list(limit=limit)
            result = [
                {
                    "sid": n.sid,
                    "phone_number": n.phone_number,
                    "friendly_name": n.friendly_name,
                    "capabilities": {
                        "sms": n.capabilities.get("sms", False) if n.capabilities else False,
                        "voice": n.capabilities.get("voice", False) if n.capabilities else False,
                    },
                }
                for n in numbers
            ]
            return json.dumps({"ok": True, "numbers": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_account_info(self) -> str:
        """Get Twilio account information.

        Returns:
            JSON string with account details.
        """
        assert self._client is not None
        try:
            account = self._client.api.accounts(self._client.username).fetch()
            return json.dumps(
                {
                    "ok": True,
                    "sid": account.sid,
                    "friendly_name": account.friendly_name,
                    "status": account.status,
                    "type": account.type,
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_whatsapp_message(self, to: str, body: str) -> str:
        """Send a WhatsApp message via Twilio.

        Args:
            to: Recipient WhatsApp number in format "whatsapp:+14155238886".
            body: Message text.

        Returns:
            JSON string with ok status and message SID.
        """
        assert self._client is not None
        try:
            to_wa = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
            from_wa = f"whatsapp:{self._from_number}"
            msg = self._client.messages.create(to=to_wa, from_=from_wa, body=body)
            return json.dumps({"ok": True, "sid": msg.sid})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_call(self, to: str, url: str, method: str = "GET") -> str:
        """Create a Twilio voice call.

        Args:
            to: Phone number to call.
            url: TwiML URL for the call instructions.
            method: HTTP method for the URL. Default: "GET".

        Returns:
            JSON string with ok status and call SID.
        """
        assert self._client is not None
        try:
            call = self._client.calls.create(to=to, from_=self._from_number, url=url, method=method)
            return json.dumps({"ok": True, "sid": call.sid, "status": call.status})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_calls(self, to: str = "", from_: str = "", limit: int = 20) -> str:
        """List recent Twilio calls.

        Args:
            to: Filter by recipient phone number. Optional.
            from_: Filter by caller phone number. Optional.
            limit: Maximum calls to return. Default: 20.

        Returns:
            JSON string with call list.
        """
        assert self._client is not None
        try:
            kwargs: dict[str, Any] = {"limit": limit}
            if to:  # pragma: no branch
                kwargs["to"] = to
            if from_:  # pragma: no branch
                kwargs["from_"] = from_
            calls = self._client.calls.list(**kwargs)
            result = [
                {
                    "sid": c.sid,
                    "from": c.from_,
                    "to": c.to,
                    "status": c.status,
                    "direction": c.direction,
                    "duration": c.duration,
                    "start_time": str(c.start_time) if c.start_time else "",
                }
                for c in calls
            ]
            return json.dumps({"ok": True, "calls": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_call(self, call_sid: str) -> str:
        """Get details about a specific Twilio call.

        Args:
            call_sid: Call SID (e.g. "CA...").

        Returns:
            JSON string with call details.
        """
        assert self._client is not None
        try:
            call = self._client.calls(call_sid).fetch()
            return json.dumps(
                {
                    "ok": True,
                    "sid": call.sid,
                    "from": call.from_,
                    "to": call.to,
                    "status": call.status,
                    "duration": call.duration,
                    "direction": call.direction,
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def cancel_message(self, message_sid: str) -> str:
        """Cancel a queued or scheduled Twilio message.

        Args:
            message_sid: Message SID to cancel.

        Returns:
            JSON string with ok status.
        """
        assert self._client is not None
        try:
            self._client.messages(message_sid).update(status="canceled")
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class SMSAgent(BaseChannelAgent, ChatSorcarAgent):
    """ChatSorcarAgent extended with Twilio SMS tools."""

    def __init__(self) -> None:
        super().__init__("SMS Agent")
        self._backend = SMSChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            try:
                from twilio.rest import Client

                self._backend._client = Client(cfg["account_sid"], cfg["auth_token"])
                self._backend._from_number = cfg.get("from_number", "")
            except Exception:
                pass

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._client is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_sms_auth() -> str:
            """Check if Twilio credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._client is None:  # pragma: no branch
                return (
                    "Not authenticated with Twilio. Use authenticate_sms() to configure.\n"
                    "You need account_sid, auth_token, and from_number from "
                    "https://console.twilio.com/ (Dashboard > Account Info)."
                )
            try:
                result = json.loads(agent._backend.get_account_info())
                if result.get("ok"):  # pragma: no branch
                    return json.dumps(
                        {
                            "ok": True,
                            "account": result.get("friendly_name", ""),
                            "from_number": agent._backend._from_number,
                        }
                    )
                return json.dumps({"ok": False, "error": "Authentication failed."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_sms(account_sid: str, auth_token: str, from_number: str = "") -> str:
            """Store and validate Twilio credentials.

            Args:
                account_sid: Twilio account SID from console.
                auth_token: Twilio auth token from console.
                from_number: Default from phone number in E.164 format.

            Returns:
                Validation result or error message.
            """
            for val, name in [(account_sid, "account_sid"), (auth_token, "auth_token")]:
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            try:
                from twilio.rest import Client

                client = Client(account_sid.strip(), auth_token.strip())
                client.api.accounts(account_sid.strip()).fetch()
                agent._backend._client = client
                agent._backend._from_number = from_number.strip()
                _config.save(
                    {
                        "account_sid": account_sid.strip(),
                        "auth_token": auth_token.strip(),
                        "from_number": from_number.strip(),
                    }
                )
                return json.dumps({"ok": True, "message": "Twilio credentials saved."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_sms_auth() -> str:
            """Clear the stored Twilio credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._client = None
            agent._backend._from_number = ""
            return "SMS authentication cleared."

        return [check_sms_auth, authenticate_sms, clear_sms_auth]


def _make_backend() -> SMSChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = SMSChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-sms -t 'authenticate'")
        sys.exit(1)
    from twilio.rest import Client

    backend._client = Client(cfg["account_sid"], cfg["auth_token"])
    backend._from_number = cfg.get("from_number", "")
    return backend


def main() -> None:
    """Run the SMSAgent from the command line with chat persistence."""
    channel_main(
        SMSAgent,
        "kiss-sms",
        channel_name="SMS",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()
