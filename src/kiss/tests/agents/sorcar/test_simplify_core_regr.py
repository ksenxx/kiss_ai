# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end regression tests locking in relentless_agent behavior.

Split out of ``tests/core/test_simplify_core_regr.py``: these methods
exercise ``kiss.agents.sorcar.relentless_agent`` for real (its ``finish``
output through the core ConsolePrinter's result panel, ``_str_to_bool``,
and ``RelentlessAgent._docker_bash``), so they belong in
``tests/agents/sorcar`` per the placement invariants.  The shared
``_make_printer`` helper stays in the lower-layer core file and is
imported from there.  Uses only real objects (no mocks/patches/fakes).
"""

import unittest
from typing import Any, cast

import yaml

from kiss.agents.sorcar.relentless_agent import RelentlessAgent, _str_to_bool
from kiss.agents.sorcar.relentless_agent import finish as relentless_finish
from kiss.tests.core.test_simplify_core_regr import _make_printer


class ConsolePrinterRegression(unittest.TestCase):
    def test_result_panel_success(self) -> None:
        p, buf = _make_printer()
        raw = relentless_finish(True, False, "everything done")
        p.print(raw, type="result", step_count=3, total_tokens=1234, cost="$0.5000")
        out = buf.getvalue()
        self.assertIn("Result", out)
        self.assertIn("everything done", out)
        self.assertNotIn("FAILED", out)
        self.assertNotIn("Status: Continue", out)
        self.assertIn("tokens=1,234", out)
        self.assertIn("cost=$0.5000", out)
        self.assertIn("steps=3", out)

    def test_result_panel_failed_and_continue(self) -> None:
        p, buf = _make_printer()
        p.print(relentless_finish(False, False, "it broke"), type="result")
        self.assertIn("Status: FAILED", buf.getvalue())
        p2, buf2 = _make_printer()
        p2.print(relentless_finish(False, True, "keep going"), type="result")
        self.assertIn("Status: Continue", buf2.getvalue())

    def test_result_panel_offsets_applied(self) -> None:
        p, buf = _make_printer()
        p.tokens_offset = 100
        p.budget_offset = 0.25
        p.steps_offset = 2
        p.print(
            relentless_finish(True, False, "done"),
            type="result",
            step_count=1,
            total_tokens=10,
            cost="$0.1000",
        )
        out = buf.getvalue()
        self.assertIn("tokens=110", out)
        self.assertIn("cost=$0.3500", out)
        self.assertIn("steps=3", out)


class RelentlessRegression(unittest.TestCase):
    def test_str_to_bool(self) -> None:
        for v in ("true", "TRUE", "1", "yes", True):
            self.assertTrue(_str_to_bool(v))
        for v in ("false", "0", "no", "", False):
            self.assertFalse(_str_to_bool(v))

    def test_finish_yaml_shape(self) -> None:
        data = yaml.safe_load(
            relentless_finish(cast(Any, "true"), cast(Any, "false"), "sum"),
        )
        self.assertEqual(data, {"success": True, "is_continue": False, "summary": "<p>sum</p>"})
        data2 = yaml.safe_load(relentless_finish(False))
        self.assertEqual(data2, {"success": False, "is_continue": False, "summary": ""})

    def test_docker_bash_without_manager_raises(self) -> None:
        from kiss.core.kiss_error import KISSError

        agent = RelentlessAgent("regr docker")
        agent.docker_manager = None
        with self.assertRaises(KISSError):
            agent._docker_bash("echo hi", "desc")


if __name__ == "__main__":
    unittest.main()
