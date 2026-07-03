# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end regression tests locking in behavior of core misc modules.

Covers code paths in config_builder, utils, base, print_to_console,
kiss_agent and relentless_agent that are touched by the simplification
pass, using only real objects (no mocks/patches/fakes).
"""

import io
import sys
import unittest
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.config import Config, set_artifact_base_dir
from kiss.core.config_builder import add_config, build_config
from kiss.core.kiss_agent import KISSAgent
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.relentless_agent import RelentlessAgent, _str_to_bool
from kiss.core.relentless_agent import finish as relentless_finish
from kiss.core.utils import (
    add_prefix_to_each_line,
    config_to_dict,
    escape_invalid_template_field_names,
    is_subpath,
    resolve_path,
)
from kiss.core.utils import (
    finish as utils_finish,
)


class ConfigBuilderRegression(unittest.TestCase):
    def setUp(self) -> None:
        self._argv = sys.argv
        self._default_config = config_module.DEFAULT_CONFIG

    def tearDown(self) -> None:
        sys.argv = self._argv
        config_module.DEFAULT_CONFIG = self._default_config

    def test_build_config_no_args_keeps_defaults(self) -> None:
        sys.argv = ["prog"]
        config_module.DEFAULT_CONFIG = Config()
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 200.0)

    def test_build_config_cli_override(self) -> None:
        sys.argv = ["prog", "--max-budget", "333.5"]
        config_module.DEFAULT_CONFIG = Config()
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 333.5)

    def test_add_config_defaults_and_cli_override(self) -> None:
        class MyCfg(BaseModel):
            foo: str = "bar"
            num_val: int = 5
            flag: bool = False

        config_module.DEFAULT_CONFIG = Config()
        sys.argv = ["prog", "--my.num-val", "7", "--my.flag"]
        add_config("my", MyCfg)
        cfg: Any = config_module.DEFAULT_CONFIG
        self.assertEqual(cfg.my.num_val, 7)
        self.assertEqual(cfg.my.foo, "bar")
        self.assertTrue(cfg.my.flag)

    def test_add_config_accumulates_previous_configs(self) -> None:
        class FirstCfg(BaseModel):
            alpha: str = "a"

        class SecondCfg(BaseModel):
            beta: float = 1.5

        config_module.DEFAULT_CONFIG = Config()
        sys.argv = ["prog"]
        add_config("first", FirstCfg)
        cfg_first: Any = config_module.DEFAULT_CONFIG
        cfg_first.first.alpha = "changed"
        add_config("second", SecondCfg)
        cfg: Any = config_module.DEFAULT_CONFIG
        self.assertEqual(cfg.first.alpha, "changed")
        self.assertEqual(cfg.second.beta, 1.5)
        self.assertEqual(cfg.max_budget, 200.0)


class UtilsRegression(unittest.TestCase):
    def test_finish_yaml_round_trip(self) -> None:
        raw = utils_finish("failure", "my analysis", "my result")
        data = yaml.safe_load(raw)
        self.assertEqual(
            data, {"status": "failure", "analysis": "my analysis", "result": "my result"}
        )

    def test_finish_defaults(self) -> None:
        data = yaml.safe_load(utils_finish())
        self.assertEqual(data["status"], "success")

    def test_escape_keeps_valid_and_escapes_invalid(self) -> None:
        out = escape_invalid_template_field_names("hi {bad} {good}", {"good"})
        self.assertEqual(out.format(good="G"), "hi {bad} G")

    def test_escape_nested_spec_invalid(self) -> None:
        out = escape_invalid_template_field_names("{a:{b}}", {"a"})
        self.assertEqual(out.format(), "{a:{b}}")

    def test_escape_nested_spec_all_valid(self) -> None:
        out = escape_invalid_template_field_names("{a:{b}}", {"a", "b"})
        self.assertEqual(out.format(a=3, b=5), "    3")

    def test_escape_conversion_preserved(self) -> None:
        out = escape_invalid_template_field_names("{good!r}", {"good"})
        self.assertEqual(out.format(good="x"), "'x'")

    def test_escape_doubles_literal_braces(self) -> None:
        out = escape_invalid_template_field_names("a {{lit}} {good}", {"good"})
        self.assertEqual(out.format(good="G"), "a {lit} G")

    def test_add_prefix_to_each_line(self) -> None:
        self.assertEqual(add_prefix_to_each_line("a\nb", "> "), "> a\n> b")

    def test_resolve_path_relative_and_absolute(self) -> None:
        base = Path.cwd()
        self.assertEqual(resolve_path("x/y", str(base)), (base / "x/y").resolve())
        abs_p = str((base / "z").resolve())
        self.assertEqual(resolve_path(abs_p, "/nonexistent"), Path(abs_p))

    def test_is_subpath(self) -> None:
        base = Path.cwd().resolve()
        self.assertTrue(is_subpath(base / "sub" / "f.txt", [base]))
        self.assertFalse(is_subpath(Path("/"), [base / "sub"]))

    def test_config_to_dict_excludes_api_keys(self) -> None:
        d = config_to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("max_budget", d)
        self.assertFalse(any("API_KEY" in k for k in d))


class BaseSaveRegression(unittest.TestCase):
    def test_save_writes_to_trajectory_path(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            set_artifact_base_dir(tmp)
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
                set_artifact_base_dir(None)

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
        from kiss.core.kiss_error import KISSError

        agent = self._agent()
        agent.max_budget = 1.0
        agent.max_steps = 5
        agent.budget_used = 2.0
        agent.step_count = 1
        with self.assertRaises(KISSError):
            agent._check_limits()
        agent.budget_used = 0.0
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


class RelentlessRegression(unittest.TestCase):
    def test_str_to_bool(self) -> None:
        for v in ("true", "TRUE", "1", "yes", True):
            self.assertTrue(_str_to_bool(v))
        for v in ("false", "0", "no", "", False):
            self.assertFalse(_str_to_bool(v))

    def test_finish_yaml_shape(self) -> None:
        # ``finish`` tolerates string booleans at runtime; type them away.
        data = yaml.safe_load(
            relentless_finish(cast(Any, "true"), cast(Any, "false"), "sum"),
        )
        self.assertEqual(data, {"success": True, "is_continue": False, "summary": "sum"})
        data2 = yaml.safe_load(relentless_finish(False))
        self.assertEqual(data2, {"success": False, "is_continue": False, "summary": ""})

    def test_broadcast_final_result(self) -> None:
        agent = RelentlessAgent("regr relentless")
        p, buf = _make_printer()
        agent.printer = p
        agent.total_steps = 7
        agent.total_tokens_used = 999
        agent.budget_used = 0.1234
        agent._broadcast_final_result(
            {"success": False, "is_continue": False, "summary": "merged summary"}
        )
        out = buf.getvalue()
        self.assertIn("merged summary", out)
        self.assertIn("Status: FAILED", out)
        self.assertIn("tokens=999", out)
        self.assertIn("steps=7", out)
        self.assertIn("cost=$0.1234", out)

    def test_broadcast_final_result_no_printer(self) -> None:
        agent = RelentlessAgent("regr relentless2")
        agent.printer = None
        agent._broadcast_final_result({"success": True, "is_continue": False, "summary": "s"})

    def test_docker_bash_without_manager_raises(self) -> None:
        from kiss.core.kiss_error import KISSError

        agent = RelentlessAgent("regr docker")
        agent.docker_manager = None
        with self.assertRaises(KISSError):
            agent._docker_bash("echo hi", "desc")


if __name__ == "__main__":
    unittest.main()
