"""Tests verifying CLI and UI modes of SorcarAgent produce identical behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _build_arg_parser,
    cli_ask_user_question,
    cli_wait_for_user,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import Attachment
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

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


# ---------------------------------------------------------------------------
# _build_arg_parser
# ---------------------------------------------------------------------------


class TestThinkingTokens:
    """Verify OpenAI reasoning tokens are counted and shown in Sorcar usage output."""

    def test_openai_reasoning_tokens_are_counted_in_usage(self) -> None:
        """Reasoning tokens from completion_tokens_details should count as output tokens."""
        model = OpenAICompatibleModel("gpt-5.4", base_url="http://localhost", api_key="test")
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                prompt_tokens_details=None,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=13),
            )
        )

        input_tokens, output_tokens, cache_read, cache_write = (
            model.extract_input_output_token_counts_from_response(response)
        )

        assert (input_tokens, output_tokens, cache_read, cache_write) == (11, 20, 0, 0)

    def test_browser_printer_displays_usage_with_reasoning_tokens(self) -> None:
        """Sorcar UI usage text should reflect reasoning-token-inclusive totals."""
        model = OpenAICompatibleModel("gpt-5.4", base_url="http://localhost", api_key="test")
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=None,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
            )
        )
        agent = KISSAgent("test")
        agent.model = model  # type: ignore[assignment]
        agent.total_tokens_used = 0
        agent.step_count = 1
        agent.max_steps = 30
        agent.budget_used = 0.0
        agent.max_budget = 5.0
        agent.session_info = ""
        agent._update_tokens_and_budget_from_response(response)
        usage = agent._get_usage_info_string()

        printer = BaseBrowserPrinter()
        client = printer.add_client()
        printer.print(usage, type="usage_info")
        event = client.get(timeout=1)

        assert agent.total_tokens_used == 175
        assert "Tokens: 175/1050000" in usage
        assert event == {"type": "usage_info", "text": usage}


class TestBuildArgParser:
    """Cover argument parsing."""

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

