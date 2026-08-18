# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Prompt-construction contract tests for the Sorcar agent.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.third_party_agents.test_sorcar_agent``; the non-core tests remain there.
"""

from __future__ import annotations

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
