# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A malformed attachment entry is skipped, not fatal (sweep 2026-10-30, V4).

``decode_attachment`` promises to "skip malformed entries ... instead
of failing the whole run", and ``_well_formed_attachment`` rejects a
non-string ``mimeType`` — but only the lazy ``PendingAttachment``
reads used it.  ``decode_attachment`` itself checked ``isinstance(att,
dict)`` and guarded only ``b64decode``, so ``{"data": "aGk=",
"mimeType": 5}`` became ``Attachment(b"hi", 5)`` and the run then
failed on ``a.mime_type.startswith(...)`` in ``SorcarAgent.run``.

Reproduced end to end: a real run through ``_cmd_run`` against a local
stand-in model endpoint, submitted with that entry and one whose
base64 does not decode.  No mocks, patches or test doubles.
"""

from __future__ import annotations

from kiss.tests.server.test_chat_viewer_fanout_sweep1030 import _captured, _RunHarness


class TestMalformedAttachmentEntry(_RunHarness):
    """The run goes ahead without the entries it cannot use."""

    def test_entry_with_a_non_string_mime_type_does_not_fail_the_run(self) -> None:
        tab_id = "sweep1030-attachment-tab"
        entries = [{"data": "aGk=", "mimeType": 5}, {"data": "a", "mimeType": "text/plain"}]
        worker = self._start(
            tab_id, "chat-sweep1030-attachment", "sweep1030 attachment run", attachments=entries
        )
        self._finish(worker)
        # A run that fails inside ``SorcarAgent.run`` still ends with
        # ``task_done``, carrying the agent's ``success: False``.
        ends = [
            (e["type"], e.get("success"))
            for e in _captured(self.printer)
            if e.get("tabId") == tab_id and e.get("type") in ("task_done", "task_error")
        ]
        self.assertEqual(
            ends, [("task_done", True)], "BUG: one malformed attachment entry failed the whole run"
        )
