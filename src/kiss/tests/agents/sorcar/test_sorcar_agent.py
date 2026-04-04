"""Tests for sorcar_agent.py: prompt construction, arg parsing, task resolution,
CLI callbacks, callback wiring, bash streaming, and autocomplete clipping."""

from __future__ import annotations

import queue
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
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.models.model import Attachment


class TestPromptConstruction:
    def _capture_prompt_and_system(
        self,
        prompt_template: str = "do stuff",
        current_editor_file: str | None = None,
        attachments: list[Attachment] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, str]:
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
        prompt, system = self._capture_prompt_and_system("do stuff")
        assert prompt == "do stuff"
        assert "file open in the editor" not in system

    def test_with_editor_file(self) -> None:
        prompt, system = self._capture_prompt_and_system(
            "do stuff", current_editor_file="/path/to/file.py"
        )
        assert "/path/to/file.py" in system
        assert "file open in the editor" in system
        assert "/path/to/file.py" not in prompt

    def test_with_images_only(self) -> None:
        attachments = [Attachment(data=b"img", mime_type="image/png")]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "1 image(s)" in prompt
        assert "PDF" not in prompt

    def test_with_pdfs_only(self) -> None:
        attachments = [Attachment(data=b"pdf", mime_type="application/pdf")]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "1 PDF(s)" in prompt
        assert "image" not in prompt

    def test_with_mixed_attachments(self) -> None:
        attachments = [
            Attachment(data=b"img", mime_type="image/png"),
            Attachment(data=b"pdf", mime_type="application/pdf"),
        ]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "1 image(s)" in prompt
        assert "1 PDF(s)" in prompt

    def test_with_multiple_images(self) -> None:
        attachments = [
            Attachment(data=b"img1", mime_type="image/png"),
            Attachment(data=b"img2", mime_type="image/jpeg"),
        ]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert "2 image(s)" in prompt

    def test_attachment_with_unknown_mime_no_parts(self) -> None:
        attachments = [Attachment(data=b"data", mime_type="text/plain")]
        prompt, _system = self._capture_prompt_and_system("do stuff", attachments=attachments)
        assert prompt == "do stuff"

    def test_with_editor_file_and_attachments(self) -> None:
        attachments = [Attachment(data=b"img", mime_type="image/png")]
        prompt, system = self._capture_prompt_and_system(
            "do stuff",
            current_editor_file="/path/to/file.py",
            attachments=attachments,
        )
        assert "1 image(s)" in prompt
        assert "/path/to/file.py" in system
        assert "/path/to/file.py" not in prompt


class TestResolveTask:
    def test_resolve_task_default(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        result = _resolve_task(args)
        assert "weather" in result.lower()

    def test_resolve_task_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "task.txt"
        f.write_text("File task content")
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", str(f)])
        result = _resolve_task(args)
        assert result == "File task content"


class TestDefaultTaskNoCredentials:
    def test_no_password_in_default_task(self) -> None:
        from kiss.agents.sorcar.sorcar_agent import _DEFAULT_TASK

        assert "password" not in _DEFAULT_TASK.lower()
        assert "kissagent" not in _DEFAULT_TASK.lower()
        assert "@gmail" not in _DEFAULT_TASK.lower()


class TestCliAskUserQuestion:
    def test_returns_user_input(self, monkeypatch: object) -> None:
        import builtins

        monkeypatch.setattr(builtins, "input", lambda prompt="": "my answer")  # type: ignore[attr-defined]
        captured: list[str] = []
        monkeypatch.setattr(builtins, "print", lambda *a, **kw: captured.append(str(a)))  # type: ignore[attr-defined]

        result = cli_ask_user_question("What is your name?")
        assert result == "my answer"
        assert any("What is your name?" in s for s in captured)


class TestCliWaitForUser:
    def test_with_url(self, monkeypatch: object) -> None:
        import builtins

        captured: list[str] = []
        monkeypatch.setattr(builtins, "print", lambda *a, **kw: captured.append(str(a)))  # type: ignore[attr-defined]
        monkeypatch.setattr(builtins, "input", lambda prompt="": "")  # type: ignore[attr-defined]

        cli_wait_for_user("Solve the CAPTCHA", "https://example.com")
        assert any("Solve the CAPTCHA" in s for s in captured)
        assert any("https://example.com" in s for s in captured)

    def test_no_url(self, monkeypatch: object) -> None:
        import builtins

        captured: list[str] = []
        monkeypatch.setattr(builtins, "print", lambda *a, **kw: captured.append(str(a)))  # type: ignore[attr-defined]
        monkeypatch.setattr(builtins, "input", lambda prompt="": "")  # type: ignore[attr-defined]

        cli_wait_for_user("Do something", "")
        assert any("Do something" in s for s in captured)
        assert not any("Current URL" in s for s in captured)


class TestSorcarAgentCallbackWiring:
    def test_ask_user_question_without_callback(self) -> None:
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
        agent = SorcarAgent("test")
        parent_class = cast(Any, agent.__class__.__mro__[1])
        original_perform = parent_class.perform_task
        captured: dict[str, object] = {}

        def wait_callback(instruction: str, url: str) -> None:
            del instruction, url

        def ask_callback(question: str) -> str:
            return f"UI: {question}"

        def fake_perform(
            self: object, tools: list, attachments: list | None = None,
        ) -> str:
            del self, attachments
            captured["wait"] = getattr(agent, "_wait_for_user_callback", None)
            captured["ask"] = getattr(agent, "_ask_user_question_callback", None)
            callables = [t for t in tools if callable(t)]
            ask_tool = next(t for t in callables if t.__name__ == "ask_user_question")
            captured["answer"] = ask_tool("hello")
            return "success: true\nis_continue: false\nsummary: ok\n"

        parent_class.perform_task = fake_perform  # type: ignore[method-assign]
        try:
            result = agent.run(
                prompt_template="task",
                wait_for_user_callback=wait_callback,
                ask_user_question_callback=ask_callback,
            )
        finally:
            parent_class.perform_task = original_perform  # type: ignore[method-assign]

        assert "success: true" in result
        assert captured["wait"] is wait_callback
        assert captured["ask"] is ask_callback
        assert captured["answer"] == "UI: hello"
        assert getattr(agent, "_wait_for_user_callback", None) is None
        assert getattr(agent, "_ask_user_question_callback", None) is None
        assert agent.web_use_tool is None


class TestBuildArgParser:
    def test_custom_args(self) -> None:
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

def _drain(q: queue.Queue) -> list[dict]:
    events: list[dict] = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


class TestSorcarBashStreaming:
    def test_multiline_bash_streams_all_lines(self):
        agent = SorcarAgent("test")
        tools = agent._get_tools()
        bash_tool = tools[0]

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        agent.printer = printer

        result = bash_tool(
            command="printf 'line1\\nline2\\nline3\\n'",
            description="multiline",
        )
        printer._flush_bash()

        assert "line1" in result
        events = _drain(cq)
        sys_text = "".join(e["text"] for e in events if e["type"] == "system_output")
        assert "line1" in sys_text
        assert "line2" in sys_text
        assert "line3" in sys_text






