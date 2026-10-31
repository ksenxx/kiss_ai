# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the model picker of a registered machine.

The fixture of :mod:`kiss.tests.server.test_remote_machine_settings`: a
real local daemon, a real remote daemon registered as ``box``, a client
on the local connection.  The chat input's model picker shows the
machine the chat header's drop-down has picked:

* ``getModels`` stamped ``machine`` is answered by the machine's daemon
  -- its model list (its custom models too) and its default model --
  with the reply stamped ``machine`` and ``forwarded``;
* ``selectModel`` stamped ``machine`` makes the model the machine's
  default and leaves this daemon's default alone;
* a ``models`` reply echoes its ``getModels``' ``generation`` (the
  page's own, never the machine link's), so the page tells a reply
  asked before its latest pick;
* an unknown machine answers either with an empty ``models`` reply
  carrying the ``error``.
"""

from __future__ import annotations

from kiss.tests.server.test_remote_machine_settings import RemoteMachineCase


class TestRemoteMachineModels(RemoteMachineCase):
    """The model picker's commands stamped ``machine`` are the machine's."""

    async def test_a_pick_changes_the_machines_default_only(self) -> None:
        # A model only box lists: one of its custom models.
        await self._ask_box(
            {"type": "saveMyModel", "name": "box-model", "endpoint": "http://box:1/v1"},
            "myModelsData",
        )
        listed = await self._ask_box({"type": "getModels"}, "models")
        self.assertEqual(listed.get("error", ""), "")
        self.assertEqual(listed["machine"], "box")
        self.assertIn("box-model", [m["name"] for m in listed["models"]])
        await self._send({"type": "getModels"})
        mine = await self._next("models", forwarded=False)
        self.assertNotIn("box-model", [m["name"] for m in mine["models"]])

        # The pick has no reply of its own; box's next list shows it.
        await self._send(
            {"type": "selectModel", "model": "box-model", "tabId": "t1", "machine": "box"},
        )
        again = await self._ask_box({"type": "getModels"}, "models")
        self.assertEqual(again["selected"], "box-model")
        await self._send({"type": "getModels"})
        after = await self._next("models", forwarded=False)
        self.assertEqual(after["selected"], mine["selected"])

    async def test_a_reply_echoes_the_pages_generation(self) -> None:
        box = await self._ask_box({"type": "getModels", "generation": 9}, "models")
        self.assertEqual(box.get("error", ""), "")
        self.assertEqual(box["generation"], 9)
        await self._send({"type": "getModels", "generation": 7})
        mine = await self._next("models", forwarded=False)
        self.assertEqual(mine["generation"], 7)
        # Unnumbered (another client): the reply carries none.
        await self._send({"type": "getModels"})
        plain = await self._next("models", forwarded=False)
        self.assertNotIn("generation", plain)

    async def test_an_unknown_machine_answers_with_the_error(self) -> None:
        for cmd, generation in (
            ({"type": "getModels", "generation": 5}, 5),
            ({"type": "selectModel", "model": "box-model", "tabId": "t1"}, 0),
        ):
            await self._send({**cmd, "machine": "nowhere"})
            reply = await self._next("models", forwarded=True)
            self.assertEqual(reply["machine"], "nowhere")
            self.assertIn("no machine named 'nowhere'", reply["error"])
            self.assertEqual(reply["models"], [])
            self.assertEqual(reply["selected"], "")
            self.assertEqual(reply["generation"], generation)
