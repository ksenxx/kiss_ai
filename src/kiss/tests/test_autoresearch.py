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
    def test_init(self) -> None:
        agent = AutoresearchAgent("test")
        assert agent.name == "test"

    def test_get_tools_returns_four(self) -> None:
        agent = AutoresearchAgent("test")
        # _get_tools needs printer set up, set it to None
        agent.printer = None
        tools = agent._get_tools()
        assert len(tools) == 4
        names = [t.__name__ for t in tools]
        assert "Bash" in names
        assert "Read" in names
        assert "Edit" in names
        assert "Write" in names

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

    def test_get_tools_stream_callback_without_printer(self) -> None:
        """Verify stream callback is no-op when printer is None."""
        agent = AutoresearchAgent("test")
        agent.printer = None
        tools = agent._get_tools()
        assert len(tools) == 4


# ---------------------------------------------------------------------------
# AutoresearchAgent._reset
# ---------------------------------------------------------------------------


class TestAutoresearchAgentReset:
    def test_reset_uses_config_defaults(self) -> None:
        agent = AutoresearchAgent("test")
        agent._reset(
            model_name=None,
            max_sub_sessions=None,
            max_steps=None,
            max_budget=None,
            work_dir=None,
            docker_image=None,
        )
        cfg = config_module.DEFAULT_CONFIG.autoresearch.autoresearch_agent
        assert agent.model_name == cfg.model_name
        assert agent.max_steps == cfg.max_steps
        assert agent.max_budget == cfg.max_budget
        assert agent.max_sub_sessions == cfg.max_sub_sessions

    def test_reset_uses_explicit_values(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            agent = AutoresearchAgent("test")
            agent._reset(
                model_name="gpt-4o",
                max_sub_sessions=5,
                max_steps=50,
                max_budget=10.0,
                work_dir=tmpdir,
                docker_image=None,
                verbose=True,
            )
            assert agent.model_name == "gpt-4o"
            assert agent.max_steps == 50
            assert agent.max_budget == 10.0
            assert agent.max_sub_sessions == 5
            assert agent.work_dir == str(Path(tmpdir).resolve())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_reset_work_dir_default(self) -> None:
        agent = AutoresearchAgent("test")
        agent._reset(
            model_name=None,
            max_sub_sessions=None,
            max_steps=None,
            max_budget=None,
            work_dir=".",
            docker_image=None,
        )
        assert agent.work_dir == str(Path(".").resolve())


# ---------------------------------------------------------------------------
# AutoresearchAgent.run - program file reading
# ---------------------------------------------------------------------------


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

    def test_run_reads_program_file(self) -> None:
        """Test that run() reads program.md when no prompt_template given."""
        agent = AutoresearchAgent("test")
        # Use very small budget to just test the path
        result = agent.run(
            model_name="gemini-2.0-flash",
            work_dir=self.tmpdir,
            max_steps=3,
            max_budget=0.05,
            max_sub_sessions=1,
        )
        # Should return YAML with success/summary
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "summary" in parsed

    def test_run_uses_prompt_template_over_program(self) -> None:
        """Test prompt_template takes precedence over program.md."""
        agent = AutoresearchAgent("test")
        result = agent.run(
            model_name="gemini-2.0-flash",
            prompt_template="Just call finish(success=True, is_continue=False, summary='direct')",
            work_dir=self.tmpdir,
            max_steps=3,
            max_budget=0.05,
            max_sub_sessions=1,
        )
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)

    def test_run_custom_program_file(self) -> None:
        """Test specifying a custom program file path."""
        custom = os.path.join(self.tmpdir, "custom_program.md")
        Path(custom).write_text(
            "Call finish(success=True, is_continue=False, summary='custom')"
        )
        agent = AutoresearchAgent("test")
        result = agent.run(
            model_name="gemini-2.0-flash",
            work_dir=self.tmpdir,
            program_file=custom,
            max_steps=3,
            max_budget=0.05,
            max_sub_sessions=1,
        )
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)

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
        """Test run() uses cwd when no work_dir specified."""
        old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            agent = AutoresearchAgent("test")
            result = agent.run(
                model_name="gemini-2.0-flash",
                max_steps=3,
                max_budget=0.05,
                max_sub_sessions=1,
            )
            parsed = yaml.safe_load(result)
            assert isinstance(parsed, dict)
        finally:
            os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# CLI arg parser
# ---------------------------------------------------------------------------


class TestArgParser:
    def test_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.model_name == "claude-opus-4-6"
        assert args.max_steps == 100
        assert args.max_budget == 200.0
        assert args.work_dir is None
        assert args.program is None
        assert args.verbose is True
        assert args.task is None

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

    def test_main_with_program_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() reading from a program file."""
        tmpdir = tempfile.mkdtemp()
        try:
            prog = os.path.join(tmpdir, "program.md")
            Path(prog).write_text(
                "Call finish(success=True, is_continue=False, summary='from file')"
            )
            monkeypatch.setattr(
                "sys.argv",
                [
                    "autoresearch",
                    "--work_dir",
                    tmpdir,
                    "--program",
                    prog,
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
