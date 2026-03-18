"""Tests for kiss/agents/sorcar/sorcar_agent.py.

Covers CLI callbacks, callback wiring, prompt construction, argument parsing,
task resolution, and default task validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.sorcar_agent import (
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

    def test_ask_user_question_without_callback(self) -> None:
        """Without callback, ask_user_question returns fallback message."""
        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(user_data_dir=None)
        try:
            tools = agent._get_tools()
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            result = ask_tool("hello?")
            assert "not available" in result
        finally:
            agent.web_use_tool.close()

    def test_run_sets_callbacks_temporarily(self) -> None:
        """run() should wire callbacks for tool execution and clear them afterward."""
        agent = SorcarAgent("test")
        parent_class = cast(Any, agent.__class__.__mro__[1])
        original_run = parent_class.run
        captured: dict[str, object] = {}

        def wait_callback(instruction: str, url: str) -> None:
            del instruction, url

        def ask_callback(question: str) -> str:
            return f"UI: {question}"

        def fake_run(
            self: object, *args: object, **kwargs: object
        ) -> str:
            del self, args
            captured["wait"] = getattr(agent, "_wait_for_user_callback", None)
            captured["ask"] = getattr(agent, "_ask_user_question_callback", None)
            tools_obj = kwargs["tools"]
            assert isinstance(tools_obj, list)
            tools = [t for t in tools_obj if callable(t)]
            ask_tool = next(t for t in tools if t.__name__ == "ask_user_question")
            captured["answer"] = ask_tool("hello")
            return "success: true\nsummary: ok\n"

        parent_class.run = fake_run  # type: ignore[method-assign]
        try:
            result = agent.run(
                prompt_template="task",
                wait_for_user_callback=wait_callback,
                ask_user_question_callback=ask_callback,
            )
        finally:
            parent_class.run = original_run  # type: ignore[method-assign]

        assert "success: true" in result
        assert captured["wait"] is wait_callback
        assert captured["ask"] is ask_callback
        assert captured["answer"] == "UI: hello"
        assert getattr(agent, "_wait_for_user_callback", None) is None
        assert getattr(agent, "_ask_user_question_callback", None) is None
        assert agent.web_use_tool is None


# ---------------------------------------------------------------------------
# Prompt construction: run() branches
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify prompt construction branches produce identical results in both modes."""

    def _capture_prompt_and_system(
        self,
        prompt_template: str = "do stuff",
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Helper: build the prompt and system prompt as run() would, without calling the LLM.

        Returns:
            (prompt, system_instructions) tuple.
        """
        from kiss.core.base import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT

        system_instructions = BASE_SYSTEM_PROMPT + (system_prompt or "")
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
            system_instructions += (
                "\n\n- The path of the file open in the editor is "
                f"{current_editor_file}"
            )
        return prompt, system_instructions

    def test_no_attachments_no_editor_file(self) -> None:
        """Base case: prompt unchanged."""
        prompt, system = self._capture_prompt_and_system("do stuff")
        assert prompt == "do stuff"
        assert "file open in the editor" not in system

    def test_with_editor_file(self) -> None:
        """current_editor_file appends path to system prompt."""
        prompt, system = self._capture_prompt_and_system(
            "do stuff", current_editor_file="/path/to/file.py"
        )
        assert "/path/to/file.py" in system
        assert "file open in the editor" in system
        assert "/path/to/file.py" not in prompt

    def test_with_images_only(self) -> None:
        """Only image attachments → prompt mentions images only."""
        attachments = [Attachment(data=b"img", mime_type="image/png")]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "1 image(s)" in prompt
        assert "PDF" not in prompt

    def test_with_pdfs_only(self) -> None:
        """Only PDF attachments → prompt mentions PDFs only."""
        attachments = [Attachment(data=b"pdf", mime_type="application/pdf")]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "1 PDF(s)" in prompt
        assert "image" not in prompt

    def test_with_mixed_attachments(self) -> None:
        """Both images and PDFs → prompt mentions both."""
        attachments = [
            Attachment(data=b"img", mime_type="image/png"),
            Attachment(data=b"pdf", mime_type="application/pdf"),
        ]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "1 image(s)" in prompt
        assert "1 PDF(s)" in prompt

    def test_with_multiple_images(self) -> None:
        """Multiple images → correct count."""
        attachments = [
            Attachment(data=b"img1", mime_type="image/png"),
            Attachment(data=b"img2", mime_type="image/jpeg"),
        ]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "2 image(s)" in prompt

    def test_attachment_with_unknown_mime_no_parts(self) -> None:
        """Attachment with non-image/non-pdf mime → no parts appended."""
        attachments = [Attachment(data=b"data", mime_type="text/plain")]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        # No img or pdf parts, so no attachment note
        assert prompt == "do stuff"

    def test_with_editor_file_and_attachments(self) -> None:
        """Both editor file and attachments → attachments in prompt, editor file in system."""
        attachments = [Attachment(data=b"img", mime_type="image/png")]
        prompt, system = self._capture_prompt_and_system(
            "do stuff",
            current_editor_file="/path/to/file.py",
            attachments=attachments,
        )
        assert "1 image(s)" in prompt
        assert "/path/to/file.py" in system
        assert "/path/to/file.py" not in prompt


# ---------------------------------------------------------------------------
# _build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    """Cover argument parsing."""

    def test_custom_args(self) -> None:
        """Custom arguments are parsed correctly."""
        parser = _build_arg_parser()
        args = parser.parse_args([
            "--model_name", "gpt-4",
            "--max_budget", "1.5",
            "--work_dir", "/tmp/test",
            "--headless", "true",
            "--verbose", "false",
            "--task", "hello world",
        ])
        assert args.model_name == "gpt-4"
        assert args.max_budget == 1.5
        assert args.work_dir == "/tmp/test"
        assert args.headless is True
        assert args.verbose is False
        assert args.task == "hello world"


# ---------------------------------------------------------------------------
# _resolve_task
# ---------------------------------------------------------------------------


class TestResolveTask:
    """Cover _resolve_task branches."""

    def test_resolve_task_default(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        result = _resolve_task(args)
        assert "weather" in result.lower()

    def test_resolve_task_from_string(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--task", "Do something"])
        result = _resolve_task(args)
        assert result == "Do something"

    def test_resolve_task_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "task.txt"
        f.write_text("File task content")
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", str(f)])
        result = _resolve_task(args)
        assert result == "File task content"


# ---------------------------------------------------------------------------
# _DEFAULT_TASK validation
# ---------------------------------------------------------------------------


class TestDefaultTaskNoCredentials:
    """Test that _DEFAULT_TASK doesn't contain hardcoded credentials."""

    def test_no_password_in_default_task(self) -> None:
        """Default task should not contain passwords."""
        from kiss.agents.sorcar.sorcar_agent import _DEFAULT_TASK

        assert "password" not in _DEFAULT_TASK.lower()
        assert "kissagent" not in _DEFAULT_TASK.lower()
        assert "@gmail" not in _DEFAULT_TASK.lower()
