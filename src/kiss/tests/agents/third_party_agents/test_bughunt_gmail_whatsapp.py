# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing a verified bug in gmail_sea.

No mocks, patches, or fakes of kiss classes: the tests use a real
googleapiclient service built from the bundled static discovery document.

Bug covered:
  (C) gmail: ``send_message`` addressed mail to a label ID (e.g. "INBOX").

The WhatsApp bugs this file once covered, (E) ``send_message`` must surface
bridge send failures and (G) ``poll_messages`` must honour
``channel_id``/limit/cursor, are pinned by ``test_whatsapp_agent.py``
(``TestChannelProtocol``) against the QR-paired bridge backend.
"""

from __future__ import annotations

import httplib2  # type: ignore[import-untyped]
import pytest
from googleapiclient.discovery import build

from kiss.agents.third_party_agents.gmail.gmail_sea import GmailChannelBackend


class TestGmailSendMessage:
    """Bug (C): send_message must not address mail to a non-email channel_id."""

    @staticmethod
    def _backend(refusing_port: int) -> GmailChannelBackend:
        """Real Gmail service whose API endpoint refuses every connection.

        ``send_message`` resolves ``thread_ts`` with a live
        ``threads().get`` call before validating the recipient; pointing
        the service at a refusing loopback port (with a short socket
        timeout) keeps that call off the network and instant.
        """
        backend = GmailChannelBackend()
        backend._service = build(
            "gmail",
            "v1",
            developerKey="test",
            static_discovery=True,
            client_options={"api_endpoint": f"http://127.0.0.1:{refusing_port}/"},
            http=httplib2.Http(timeout=2),
        )
        return backend

    def test_send_message_rejects_label_id_recipient(self, refusing_port: int) -> None:
        with pytest.raises(ValueError, match="email address"):
            self._backend(refusing_port).send_message("INBOX", "hello")

    def test_send_message_rejects_label_id_when_thread_unresolvable(
        self, refusing_port: int
    ) -> None:
        with pytest.raises(ValueError, match="email address"):
            self._backend(refusing_port).send_message(
                "INBOX", "hello", thread_ts="nonexistent-thread"
            )
