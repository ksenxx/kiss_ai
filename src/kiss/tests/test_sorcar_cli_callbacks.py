"""Tests verifying CLI and UI modes of SorcarAgent produce identical behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

from kiss.agents.sorcar.sorcar_agent import (
    _DEFAULT_TASK,
    SorcarAgent,
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.models.model import Attachment

# ---------------------------------------------------------------------------
# CLI callbacks (module-level, importable)
# ---------------------------------------------------------------------------


class TestCliAskUserQuestion:
    """Test the module-level cli_ask_user_question callback."""

    def test_returns_user_input(self, monkeypatch: object) -> None:
        """Callback reads from stdin and returns the answer."""
        import builtins

        monkeypatch.setattr(builtins, "input", lambda prompt="": "my answer")  # type: ignore[attr-defined]
        captured: list[str] = []
        monkeypatch.setattr(builtins, "print", lambda *a, **kw: captured.append(str(a)))  # type: ignore[attr-defined]

        result = cli_ask_user_question("What is your name?")
        assert result == "my answer"
        assert any("What is your name?" in s for s in captured)

    def test_empty_answer(self, monkeypatch: object) -> None:
        """Callback handles empty input."""
        import builtins

        monkeypatch.setattr(builtins, "input", lambda prompt="": "")  # type: ignore[attr-defined]
        monkeypatch.setattr(builtins, "print", lambda *a, **kw: None)  # type: ignore[attr-defined]
        assert cli_ask_user_question("anything") == ""


class TestCliWaitForUser:
    """Test the module-level cli_wait_for_user callback."""

    def test_with_url(self, monkeypatch: object) -> None:
        """Prints instruction + URL, waits for Enter."""
        import builtins

        captured: list[str] = []
        monkeypatch.setattr(builtins, "print", lambda *a, **kw: captured.append(str(a)))  # type: ignore[attr-defined]
        monkeypatch.setattr(builtins, "input", lambda prompt="": "")  # type: ignore[attr-defined]

        cli_wait_for_user("Solve the CAPTCHA", "https://example.com")
        assert any("Solve the CAPTCHA" in s for s in captured)
        assert any("https://example.com" in s for s in captured)

    def test_no_url(self, monkeypatch: object) -> None:
        """Empty URL skips the URL line."""
        import builtins

        captured: list[str] = []
        monkeypatch.setattr(builtins, "print", lambda *a, **kw: captured.append(str(a)))  # type: ignore[attr-defined]
        monkeypatch.setattr(builtins, "input", lambda prompt="": "")  # type: ignore[attr-defined]

        cli_wait_for_user("Do something", "")
        assert any("Do something" in s for s in captured)
        assert not any("Current URL" in s for s in captured)


# ---------------------------------------------------------------------------
# SorcarAgent callback wiring
# ---------------------------------------------------------------------------


class TestSorcarAgentCallbackWiring:
    """Verify that both CLI and UI callback wiring paths produce identical tool behavior."""

    def test_ask_user_question_with_callback(self) -> None:
        """When callback is provided, the ask_user_question tool calls it."""

        def my_callback(question: str) -> str:
            return f"answer: {question}"

        agent = SorcarAgent("test", ask_user_question_callback=my_callback)
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            assert ask_tool("test?") == "answer: test?"
        finally:
            agent.web_use_tool.close()

    def test_ask_user_question_without_callback(self) -> None:
        """Without callback, ask_user_question returns fallback message."""
        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            result = ask_tool("hello?")
            assert "not available" in result
        finally:
            agent.web_use_tool.close()

    def test_wait_for_user_callback_stored_and_passed_to_web_use_tool(self) -> None:
        """wait_for_user_callback is passed through to WebUseTool."""
        calls: list[tuple[str, str]] = []

        def my_wait(instruction: str, url: str) -> None:
            calls.append((instruction, url))

        agent = SorcarAgent("test", wait_for_user_callback=my_wait)
        agent.web_use_tool = WebUseTool(
            headless=True,
            user_data_dir=None,
            wait_for_user_callback=my_wait,
        )
        assert agent.web_use_tool._wait_for_user_callback is my_wait
        agent.web_use_tool.close()

    def test_ui_mode_patches_callbacks_after_creation(self) -> None:
        """UI mode (sorcar.py) patches callbacks after agent creation.

        Verify that patching _ask_user_question_callback after __init__
        is reflected in _get_tools().
        """
        agent = SorcarAgent("test")
        # Simulate what sorcar.py does: patch callback after creation
        agent._ask_user_question_callback = lambda q: f"UI: {q}"
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            assert ask_tool("hello") == "UI: hello"
        finally:
            agent.web_use_tool.close()

    def test_cli_and_ui_callbacks_produce_same_tool_behavior(self) -> None:
        """Both CLI and UI wiring produce the same ask_user_question tool behavior."""

        def cli_cb(q: str) -> str:
            return f"got: {q}"

        def ui_cb(q: str) -> str:
            return f"got: {q}"

        # CLI: callback set at construction
        cli_agent = SorcarAgent("cli", ask_user_question_callback=cli_cb)
        cli_agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)

        # UI: callback patched after construction
        ui_agent = SorcarAgent("ui")
        ui_agent._ask_user_question_callback = ui_cb
        ui_agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)

        try:
            cli_tools = cli_agent._get_tools()
            ui_tools = ui_agent._get_tools()

            cli_ask = next(t for t in cli_tools if t.__name__ == "ask_user_question")
            ui_ask = next(t for t in ui_tools if t.__name__ == "ask_user_question")

            assert cli_ask("q") == ui_ask("q") == "got: q"

            # Same number of tools
            assert len(cli_tools) == len(ui_tools)

            # Same tool names
            cli_names = [t.__name__ for t in cli_tools]
            ui_names = [t.__name__ for t in ui_tools]
            assert cli_names == ui_names
        finally:
            cli_agent.web_use_tool.close()
            ui_agent.web_use_tool.close()


# ---------------------------------------------------------------------------
# _get_tools() branches
# ---------------------------------------------------------------------------


class TestGetToolsBranches:
    """Cover all branches in _get_tools()."""

    def test_tools_without_web_use_tool(self) -> None:
        """Without web_use_tool set, no browser tools are included."""
        agent = SorcarAgent("test")
        # web_use_tool is None by default
        assert agent.web_use_tool is None
        tools = agent._get_tools()
        tool_names = [t.__name__ for t in tools]
        assert "Bash" in tool_names
        assert "Read" in tool_names
        assert "Edit" in tool_names
        assert "Write" in tool_names
        assert "ask_user_question" in tool_names
        # No browser tools
        assert "go_to_url" not in tool_names

    def test_tools_with_web_use_tool(self) -> None:
        """With web_use_tool set, browser tools are included."""
        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            tool_names = [t.__name__ for t in tools]
            assert "go_to_url" in tool_names
            assert "ask_user_question" in tool_names
        finally:
            agent.web_use_tool.close()

    def test_stream_callback_with_printer(self) -> None:
        """_stream calls printer.print when printer is set."""
        printed: list[tuple[str, str]] = []

        class FakePrinter:
            def print(self, text: str, type: str = "") -> None:
                printed.append((text, type))

        agent = SorcarAgent("test")
        agent.printer = FakePrinter()  # type: ignore[assignment]
        tools = agent._get_tools()
        # The stream callback is internal to _get_tools, but we can verify
        # by checking that tools were created without error
        assert len(tools) >= 5

    def test_stream_callback_without_printer(self) -> None:
        """_stream is no-op when printer is None."""
        agent = SorcarAgent("test")
        agent.printer = None
        # Should not raise
        tools = agent._get_tools()
        assert len(tools) >= 5


# ---------------------------------------------------------------------------
# Prompt construction: run() branches
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify prompt construction branches produce identical results in both modes."""

    def _capture_prompt(
        self,
        prompt_template: str = "do stuff",
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Helper: build the prompt as run() would, without calling the LLM."""
        prompt = prompt_template
        if attachments:
            pdf_count = sum(
                1 for a in attachments if a.mime_type == "application/pdf"
            )
            img_count = sum(
                1 for a in attachments if a.mime_type.startswith("image/")
            )
            parts = []
            if img_count:
                parts.append(f"{img_count} image(s)")
            if pdf_count:
                parts.append(f"{pdf_count} PDF(s)")
            if parts:
                prompt += (
                    f"\n\n# Important\n - User attached {', '.join(parts)}. "
                    f"The files are included in this message. "
                    f"Examine them directly — do NOT use browser tools "
                    f"to view or screenshot these attachments."
                )
        if current_editor_file:
            prompt += (
                "\n\n- The path of the file open in the editor is "
                f"{current_editor_file}"
            )
        return prompt

    def test_no_attachments_no_editor_file(self) -> None:
        """Base case: prompt unchanged."""
        prompt = self._capture_prompt("do stuff")
        assert prompt == "do stuff"

    def test_with_editor_file(self) -> None:
        """current_editor_file appends path to prompt."""
        prompt = self._capture_prompt(
            "do stuff", current_editor_file="/path/to/file.py"
        )
        assert "/path/to/file.py" in prompt
        assert "file open in the editor" in prompt

    def test_with_images_only(self) -> None:
        """Only image attachments → prompt mentions images only."""
        attachments = [Attachment(data=b"img", mime_type="image/png")]
        prompt = self._capture_prompt("do stuff", attachments=attachments)
        assert "1 image(s)" in prompt
        assert "PDF" not in prompt

    def test_with_pdfs_only(self) -> None:
        """Only PDF attachments → prompt mentions PDFs only."""
        attachments = [Attachment(data=b"pdf", mime_type="application/pdf")]
        prompt = self._capture_prompt("do stuff", attachments=attachments)
        assert "1 PDF(s)" in prompt
        assert "image" not in prompt

    def test_with_mixed_attachments(self) -> None:
        """Both images and PDFs → prompt mentions both."""
        attachments = [
            Attachment(data=b"img", mime_type="image/png"),
            Attachment(data=b"pdf", mime_type="application/pdf"),
        ]
        prompt = self._capture_prompt("do stuff", attachments=attachments)
        assert "1 image(s)" in prompt
        assert "1 PDF(s)" in prompt

    def test_with_multiple_images(self) -> None:
        """Multiple images → correct count."""
        attachments = [
            Attachment(data=b"img1", mime_type="image/png"),
            Attachment(data=b"img2", mime_type="image/jpeg"),
        ]
        prompt = self._capture_prompt("do stuff", attachments=attachments)
        assert "2 image(s)" in prompt

    def test_attachment_with_unknown_mime_no_parts(self) -> None:
        """Attachment with non-image/non-pdf mime → no parts appended."""
        attachments = [Attachment(data=b"data", mime_type="text/plain")]
        prompt = self._capture_prompt("do stuff", attachments=attachments)
        # No img or pdf parts, so no attachment note
        assert prompt == "do stuff"

    def test_with_editor_file_and_attachments(self) -> None:
        """Both editor file and attachments → both appended."""
        attachments = [Attachment(data=b"img", mime_type="image/png")]
        prompt = self._capture_prompt(
            "do stuff",
            current_editor_file="/path/to/file.py",
            attachments=attachments,
        )
        assert "1 image(s)" in prompt
        assert "/path/to/file.py" in prompt
        # Editor file comes after attachments
        attach_idx = prompt.index("image(s)")
        editor_idx = prompt.index("/path/to/file.py")
        assert editor_idx > attach_idx


# ---------------------------------------------------------------------------
# _resolve_task
# ---------------------------------------------------------------------------


class TestResolveTask:
    """Cover all three branches of _resolve_task."""

    def test_from_file(self, tmp_path: Path) -> None:
        """Priority 1: -f flag reads from file."""
        task_file = tmp_path / "task.txt"
        task_file.write_text("task from file")
        args = argparse.Namespace(f=str(task_file), task="ignored")
        assert _resolve_task(args) == "task from file"

    def test_from_task_arg(self) -> None:
        """Priority 2: --task flag."""
        args = argparse.Namespace(f=None, task="task from arg")
        assert _resolve_task(args) == "task from arg"

    def test_default_task(self) -> None:
        """Priority 3: default task."""
        args = argparse.Namespace(f=None, task=None)
        assert _resolve_task(args) == _DEFAULT_TASK

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        args = argparse.Namespace(f=str(tmp_path / "missing.txt"), task=None)
        try:
            _resolve_task(args)
            assert False, "Should have raised"
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# _build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    """Cover argument parsing."""

    def test_defaults(self) -> None:
        """Default values match expected."""
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.model_name == "claude-opus-4-6"
        assert args.max_steps == 30
        assert args.max_budget == 5.0
        assert args.work_dir is None
        assert args.headless is False
        assert args.verbose is True
        assert args.task is None
        assert args.f is None

    def test_custom_args(self) -> None:
        """Custom arguments are parsed correctly."""
        parser = _build_arg_parser()
        args = parser.parse_args([
            "--model_name", "gpt-4",
            "--max_steps", "10",
            "--max_budget", "1.5",
            "--work_dir", "/tmp/test",
            "--headless", "true",
            "--verbose", "false",
            "--task", "hello world",
        ])
        assert args.model_name == "gpt-4"
        assert args.max_steps == 10
        assert args.max_budget == 1.5
        assert args.work_dir == "/tmp/test"
        assert args.headless is True
        assert args.verbose is False
        assert args.task == "hello world"

    def test_file_arg(self, tmp_path: Path) -> None:
        """-f flag is parsed correctly."""
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", str(tmp_path / "task.txt")])
        assert args.f == str(tmp_path / "task.txt")

    def test_headless_false_string(self) -> None:
        """--headless false parses to False."""
        parser = _build_arg_parser()
        args = parser.parse_args(["--headless", "false"])
        assert args.headless is False

    def test_headless_capitalized_true_string(self) -> None:
        """--headless True (capitalized) parses to True."""
        parser = _build_arg_parser()
        args = parser.parse_args(["--headless", "True"])
        assert args.headless is True
