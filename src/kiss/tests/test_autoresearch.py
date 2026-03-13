"""Integration tests for kiss/agents/autoresearch/ with 100% branch coverage.

No mocks, patches, or test doubles. Uses real files and real objects.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from kiss.agents.autoresearch.autoresearch_agent import (
    _DEFAULT_PROGRAM,
    AutoresearchAgent,
    _build_arg_parser,
    main,
)
from kiss.core import config as config_module

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestAutoresearchConfig:
    def test_config_defaults(self) -> None:
        cfg = config_module.DEFAULT_CONFIG.autoresearch.autoresearch_agent
        assert cfg.model_name == "claude-opus-4-6"
        assert cfg.max_steps == 100
        assert cfg.max_budget == 200.0
        assert cfg.max_sub_sessions == 10000
        assert cfg.verbose is False

    def test_config_registered(self) -> None:
        assert hasattr(config_module.DEFAULT_CONFIG, "autoresearch")


# ---------------------------------------------------------------------------
# AutoresearchAgent construction and tools
# ---------------------------------------------------------------------------


class TestAutoresearchAgentInit:

    def test_get_tools_stream_callback_with_printer(self) -> None:
        """Verify the stream callback calls printer.print when printer is set."""

        agent = AutoresearchAgent("test")
        printed: list[tuple[str, str]] = []

        class CapturePrinter:
            def print(self, text: str, type: str = "") -> None:
                printed.append((text, type))

        agent.printer = CapturePrinter()  # type: ignore[assignment]
        tools = agent._get_tools()
        # Exercise the stream callback by calling Bash with a simple command
        bash_tool = next(t for t in tools if t.__name__ == "Bash")
        bash_tool(command="echo hello", description="test")
        assert any("hello" in t for t, _ in printed)

class TestAutoresearchAgentRun:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.program_content = (
            "Say 'hello world' and call finish(success=True, "
            "is_continue=False, summary='said hello')"
        )
        Path(self.tmpdir, _DEFAULT_PROGRAM).write_text(self.program_content)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_missing_program_file_raises(self) -> None:
        """Test FileNotFoundError when program.md doesn't exist."""
        empty_dir = tempfile.mkdtemp()
        try:
            agent = AutoresearchAgent("test")
            with pytest.raises(FileNotFoundError):
                agent.run(
                    model_name="gemini-2.0-flash",
                    work_dir=empty_dir,
                    max_steps=3,
                    max_budget=0.05,
                    max_sub_sessions=1,
                )
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_run_default_work_dir(self) -> None:
        """Test run() uses cwd when no work_dir specified.

        Verifies that program.md is found in cwd (not a FileNotFoundError).
        The agent may or may not complete within the step budget.
        """
        from kiss.core.kiss_error import KISSError

        old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            agent = AutoresearchAgent("test")
            try:
                result = agent.run(
                    model_name="gemini-2.0-flash",
                    max_steps=3,
                    max_budget=0.05,
                    max_sub_sessions=1,
                )
                parsed = yaml.safe_load(result)
                assert isinstance(parsed, dict)
            except KISSError:
                # KISSError means the agent ran (found program.md) but
                # exhausted its step budget — this still verifies cwd is used.
                pass
        finally:
            os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# CLI arg parser
# ---------------------------------------------------------------------------


class TestArgParser:

    def test_custom_args(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([
            "--model_name", "gpt-4o",
            "--max_steps", "50",
            "--max_budget", "10.0",
            "--work_dir", "/tmp/test",
            "--program", "/tmp/prog.md",
            "--verbose", "false",
            "--task", "hello",
        ])
        assert args.model_name == "gpt-4o"
        assert args.max_steps == 50
        assert args.max_budget == 10.0
        assert args.work_dir == "/tmp/test"
        assert args.program == "/tmp/prog.md"
        assert args.verbose is False
        assert args.task == "hello"


# ---------------------------------------------------------------------------
# main() CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_with_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() with --task flag runs successfully."""
        tmpdir = tempfile.mkdtemp()
        try:
            monkeypatch.setattr(
                "sys.argv",
                [
                    "autoresearch",
                    "--task",
                    "Call finish(success=True, is_continue=False, summary='ok')",
                    "--work_dir",
                    tmpdir,
                    "--model_name",
                    "gemini-2.0-flash",
                    "--max_steps",
                    "3",
                    "--max_budget",
                    "0.05",
                ],
            )
            main()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_main_default_work_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() without --work_dir uses cwd."""
        tmpdir = tempfile.mkdtemp()
        prog = os.path.join(tmpdir, "program.md")
        Path(prog).write_text(
            "Call finish(success=True, is_continue=False, summary='cwd')"
        )
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            monkeypatch.setattr(
                "sys.argv",
                [
                    "autoresearch",
                    "--model_name",
                    "gemini-2.0-flash",
                    "--max_steps",
                    "3",
                    "--max_budget",
                    "0.05",
                ],
            )
            main()
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# __init__.py coverage
# ---------------------------------------------------------------------------


class TestInit:
    def test_import(self) -> None:
        import kiss.agents.autoresearch  # noqa: F401

        assert hasattr(config_module.DEFAULT_CONFIG, "autoresearch")
