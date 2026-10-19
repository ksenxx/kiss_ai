# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a command whose catalog entry requires ``tabId`` is refused with
an ``error`` reply when the field is absent.

``ServerApi.dispatch`` canonicalises ``tabId`` before validation.  It
used to write ``""`` back for an ABSENT field too, so
``validate_command`` (which treats only ``None`` as missing) never
fired the ``required=("tabId",)`` declared for ``closeTab``, ``openTab``
and ``interruptTool``: the handler then silently ignored the command
and the client got no reply at all (findings H2-F2).

Real daemon over its local WSS endpoint, one real client connection.
"""

from __future__ import annotations

from kiss.tests.server.two_client_harness import TwoClientHarness


class RequiredTabIdTest(TwoClientHarness):
    """Missing ``tabId`` is answered; a present one still canonicalises."""

    def test_absent_tab_id_is_answered_with_an_error(self) -> None:
        client = self.open_client("a")
        client.ready()
        for name in ("closeTab", "openTab", "interruptTool"):
            client.send({"type": name})
            self.wait_event(
                client,
                "error",
                text=f"Invalid {name} command: missing tabId",
            )
        self.assertEqual([e.get("tabId") for e in client.of_type("error")], [None] * 3)

    def test_present_tab_id_is_stripped_and_accepted(self) -> None:
        client = self.open_client("a")
        client.ready()
        before = len(client.of_type("tabs_state"))
        client.send({"type": "openTab", "tabId": "  padded-tab  ", "title": "p"})
        snapshot = self.wait_for(
            lambda: [
                e
                for e in client.of_type("tabs_state")[before:]
                if any(t.get("tabId") == "padded-tab" for t in e.get("tabs", []))
            ],
            what="tabs_state with the stripped tab id",
        )
        self.assertTrue(snapshot)
        self.assertEqual(client.of_type("error"), [])
