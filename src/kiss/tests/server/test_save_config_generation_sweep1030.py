# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the model list a ``saveConfig`` answers with echoes the save's ``generation``.

The model picker tells a ``models`` reply asked before the user's
latest pick by its echoed ``generation`` (``getModels`` echoes it) and
keeps the pick.  ``saveConfig`` also answers with this daemon's model
list, and for this daemon that is the picker's only refresh after a
save; without the echo, a save served just before a pick named the old
default and the picker flipped back to it.

Real daemon over its local WSS endpoint, a real client connection.
"""

from __future__ import annotations

from kiss.tests.server.two_client_harness import TwoClientHarness


class SaveConfigGenerationTest(TwoClientHarness):
    """``saveConfig``'s ``models`` reply carries the command's ``generation``."""

    def test_the_models_reply_of_a_save_echoes_its_generation(self) -> None:
        a = self.open_client("a")
        a.ready()
        self.wait_event(a, "models")
        self.assertEqual(len(a.of_type("models")), 1)
        a.send({"type": "saveConfig", "config": {}, "apiKeys": {}, "generation": 7})
        self.wait_for(lambda: len(a.of_type("models")) == 2, what="the save's models reply")
        self.assertEqual(a.of_type("models")[1].get("generation"), 7)

        # A save without one (an older page) is answered as before.
        a.send({"type": "saveConfig", "config": {}, "apiKeys": {}})
        self.wait_for(lambda: len(a.of_type("models")) == 3, what="the second save's reply")
        self.assertNotIn("generation", a.of_type("models")[2])
