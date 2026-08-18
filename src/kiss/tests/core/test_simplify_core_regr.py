# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end regression tests locking in behavior of core misc modules.

Covers code paths in utils, base, print_to_console and kiss_agent that
are touched by the simplification pass, using only real objects (no
mocks/patches/fakes).  The relentless_agent-dependent methods (the
result-panel printer tests and ``RelentlessRegression``) moved to
``tests/agents/sorcar/test_simplify_core_regr.py``, which imports the
shared ``_make_printer`` helper from this module.  The five
``escape_invalid_template_field_names`` methods moved to
``tests/agents/obsolete/gepa/test_simplify_core_regr.py`` because that
helper lives in ``kiss.agents.obsolete.gepa``, outside ``kiss.core``.
"""

import io
import unittest
from typing import Any, cast

import yaml

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.printer import parse_result_yaml
from kiss.core.utils import (
    config_to_dict,
)
from kiss.core.utils import (
    finish as utils_finish,
)


class UtilsRegression(unittest.TestCase):
    def test_finish_yaml_round_trip(self) -> None:
        raw = utils_finish(False, True, "my summary")
        data = yaml.safe_load(raw)
        self.assertEqual(
            data, {"success": False, "is_continue": True, "summary": "<p>my summary</p>"}
        )

    def test_finish_defaults(self) -> None:
        data = yaml.safe_load(utils_finish(True))
        self.assertEqual(
            data, {"success": True, "is_continue": False, "summary": ""}
        )

    def test_finish_coerces_string_booleans(self) -> None:
        data = yaml.safe_load(
            utils_finish(cast(Any, "true"), cast(Any, "no"), "s")
        )
        self.assertEqual(data, {"success": True, "is_continue": False, "summary": "<p>s</p>"})

    def test_finish_output_recognized_by_parse_result_yaml(self) -> None:
        raw = utils_finish(True, False, "structured summary")
        parsed = parse_result_yaml(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["summary"], "<p>structured summary</p>")
        self.assertTrue(parsed["success"])

    def test_config_to_dict_excludes_api_keys(self) -> None:
        d = config_to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("max_budget", d)
        self.assertFalse(any("API_KEY" in k for k in d))


class BaseSaveRegression(unittest.TestCase):
    def test_save_writes_to_trajectory_path(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # The runtime artifact-base setter was removed (it had no
            # production caller and could misroute a running agent's
            # trajectory), so redirect the process-wide directory here.
            original = config_module._artifact_dir
            config_module._artifact_dir = tmp
            try:
                agent = Base("regr agent/save")
                agent._add_message("user", "hello")
                agent._save()
                path = agent.get_trajectory_path()
                self.assertTrue(path.exists())
                data = yaml.safe_load(path.read_text())
                self.assertEqual(data["name"], "regr agent/save")
                self.assertEqual(data["messages"][0]["content"], "hello")
                self.assertEqual(data["max_tokens"], None)
                self.assertIn("trajectory_regr_agent_save_", path.name)
            finally:
                config_module._artifact_dir = original

    def test_get_trajectory_json(self) -> None:
        agent = Base("regr json")
        agent._add_message("user", "hi", timestamp=42)
        import json

        msgs = json.loads(agent.get_trajectory())
        self.assertEqual(msgs[0]["timestamp"], 42)


def _make_printer() -> tuple[ConsolePrinter, io.StringIO]:
    buf = io.StringIO()
    return ConsolePrinter(file=buf), buf


class ConsolePrinterRegression(unittest.TestCase):
    def test_text_and_empty_text(self) -> None:
        p, buf = _make_printer()
        p.print("   ", type="text")
        self.assertEqual(buf.getvalue(), "")
        p.print("hello world", type="text")
        self.assertIn("hello world", buf.getvalue())

    def test_prompt_and_system_prompt_panels(self) -> None:
        p, buf = _make_printer()
        p.print("do the thing", type="prompt")
        p.print("be nice", type="system_prompt")
        out = buf.getvalue()
        self.assertIn("Prompt", out)
        self.assertIn("System Prompt", out)
        self.assertIn("do the thing", out)
        self.assertIn("be nice", out)

    def test_tool_call_panel_full_inputs(self) -> None:
        p, buf = _make_printer()
        p.print(
            "Edit",
            type="tool_call",
            tool_input={
                "file_path": "a/b.py",
                "description": "edit desc",
                "command": "ls -la",
                "content": "print(1)",
                "old_string": "old",
                "new_string": "new",
                "extra_key": "extra_val",
            },
        )
        out = buf.getvalue()
        for expected in ("Edit", "a/b.py", "edit desc", "old:", "new:", "extra_key: extra_val"):
            self.assertIn(expected, out)

    def test_tool_call_no_arguments(self) -> None:
        p, buf = _make_printer()
        p.print("NoArgs", type="tool_call", tool_input={})
        self.assertIn("(no arguments)", buf.getvalue())

    def test_tool_result_success_and_error(self) -> None:
        p, buf = _make_printer()
        p.print("all good", type="tool_result", tool_name="Bash", tool_input={})
        self.assertIn("RESULT", buf.getvalue())
        self.assertIn("all good", buf.getvalue())
        p2, buf2 = _make_printer()
        p2.print("boom", type="tool_result", tool_name="Bash", tool_input={}, is_error=True)
        self.assertIn("FAILED", buf2.getvalue())

    def test_finish_tool_result_suppressed(self) -> None:
        p, buf = _make_printer()
        p.print("final", type="tool_result", tool_name="finish", tool_input={})
        self.assertEqual(buf.getvalue(), "")

    def test_read_tool_result_syntax_highlighted(self) -> None:
        p, buf = _make_printer()
        p.print(
            "import os\n",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "m.py", "start_line": 3},
        )
        self.assertIn("import", buf.getvalue())

    def test_read_tool_result_empty_sentinel_plain(self) -> None:
        p, buf = _make_printer()
        p.print(
            "(file is empty)",
            type="tool_result",
            tool_name="Read",
            tool_input={"file_path": "m.py"},
        )
        self.assertIn("(file is empty)", buf.getvalue())

    def test_bash_stream_then_result_closes_panel(self) -> None:
        p, buf = _make_printer()
        p.print("streamed line\n", type="bash_stream")
        out_mid = buf.getvalue()
        self.assertIn("RESULT", out_mid)
        self.assertIn("streamed line", out_mid)
        p.print("streamed line\n", type="tool_result", tool_name="Bash", tool_input={})
        self.assertEqual(buf.getvalue().count("RESULT"), 1)

    def test_bash_stream_then_error_result(self) -> None:
        p, buf = _make_printer()
        p.print("oops", type="bash_stream")
        p.print("oops", type="tool_result", tool_name="Bash", tool_input={}, is_error=True)
        self.assertIn("FAILED", buf.getvalue())

    def test_usage_info(self) -> None:
        p, buf = _make_printer()
        p.print("Steps: 1/10", type="usage_info")
        self.assertIn("Steps: 1/10", buf.getvalue())
        p.print("   ", type="usage_info")

    def test_notification_severities(self) -> None:
        for severity, label in (
            ("info", "INFO"),
            ("warning", "WARNING"),
            ("error", "ERROR"),
            ("unknown", "INFO"),
        ):
            p, buf = _make_printer()
            p.print("note body", type="notification", severity=severity,
                    progress_message="sub detail")
            out = buf.getvalue()
            self.assertIn(label, out)
            self.assertIn("note body", out)
            self.assertIn("sub detail", out)

    def test_result_panel_non_yaml_and_empty(self) -> None:
        p, buf = _make_printer()
        p.print("plain markdown result", type="result")
        self.assertIn("plain markdown result", buf.getvalue())
        p2, buf2 = _make_printer()
        p2.print("", type="result")
        self.assertIn("(no result)", buf2.getvalue())

    def test_apply_budget_offset_non_dollar(self) -> None:
        p, _ = _make_printer()
        p.budget_offset = 1.0
        self.assertEqual(p._apply_budget_offset("N/A"), "N/A")
        self.assertEqual(p._apply_budget_offset("$bad"), "$bad")
        self.assertEqual(p._apply_budget_offset("$1.0"), "$2.0000")

    def test_thinking_and_token_callbacks(self) -> None:
        p, buf = _make_printer()
        p.thinking_callback(True)
        p.token_callback("pondering")
        p.thinking_callback(False)
        p.token_callback("answer text")
        out = buf.getvalue()
        self.assertIn("Thinking", out)
        self.assertIn("pondering", out)
        self.assertIn("answer text", out)

    def test_reset_clears_state(self) -> None:
        p, _ = _make_printer()
        p.print("x", type="bash_stream")
        p.reset()
        self.assertFalse(p._bash_streamed)
        self.assertFalse(p._mid_line)
        self.assertEqual(p._current_block_type, "")

    def test_unknown_type_returns_empty(self) -> None:
        p, buf = _make_printer()
        self.assertEqual(p.print("x", type="does_not_exist"), "")
        self.assertEqual(buf.getvalue(), "")


def _echo_tool(x: str = "") -> str:
    """Echo the input back."""
    return f"echo:{x}"


def _boom_tool() -> str:
    """Always raises."""
    raise RuntimeError("kaput")


class KISSAgentToolRegression(unittest.TestCase):
    def _agent(self) -> KISSAgent:
        agent = KISSAgent("regr tool agent")
        agent.function_map = {"_echo_tool": _echo_tool, "_boom_tool": _boom_tool}
        return agent

    def test_execute_tool_success(self) -> None:
        name, resp = self._agent()._execute_tool(
            {"name": "_echo_tool", "arguments": {"x": "hi"}}
        )
        self.assertEqual((name, resp), ("_echo_tool", "echo:hi"))

    def test_execute_tool_non_dict_arguments(self) -> None:
        name, resp = self._agent()._execute_tool({"name": "_echo_tool", "arguments": None})
        self.assertEqual((name, resp), ("_echo_tool", "echo:"))

    def test_execute_tool_error_includes_signature(self) -> None:
        agent = self._agent()
        p, buf = _make_printer()
        agent.printer = p
        name, resp = agent._execute_tool({"name": "_boom_tool", "arguments": {}})
        self.assertEqual(name, "_boom_tool")
        self.assertIn("Failed to call _boom_tool", resp)
        self.assertIn("kaput", resp)
        self.assertIn("Expected signature: _boom_tool", resp)
        self.assertIn("FAILED", buf.getvalue())

    def test_check_limits(self) -> None:
        """The budget bound is enforced here; the step bound is not.

        The second half used to assert that ``step_count > max_steps``
        also raised from here.  That branch was unreachable — the
        agentic loop stops first — and it worded the same condition
        differently, so it was removed and the step bound now lives
        solely in ``_run_agentic_loop`` (covered end to end by
        ``test_artifact_dir_and_step_limit.py``).
        """
        from kiss.core.kiss_error import KISSError

        agent = self._agent()
        agent.max_budget = 1.0
        agent.max_steps = 5
        agent.budget_used = 2.0
        agent.step_count = 6
        with self.assertRaises(KISSError):
            agent._check_limits()

    def test_add_functions_duplicate_raises(self) -> None:
        from kiss.core.kiss_error import KISSError

        agent = self._agent()
        with self.assertRaises(KISSError):
            agent._add_functions([_echo_tool])

    def test_finish_returns_result(self) -> None:
        self.assertEqual(self._agent().finish("done"), "done")


if __name__ == "__main__":
    unittest.main()
