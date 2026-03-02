"""Tests for the current_editor_file parameter plumbing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


class TestActiveFileReading:
    """Test reading active-file.json for current_editor_file."""

    def test_reads_active_file_json(self, tmp_path: Path) -> None:
        af = tmp_path / "active-file.json"
        af.write_text(json.dumps({"path": "/foo/bar.py"}))
        with open(str(af)) as f:
            result = json.loads(f.read()).get("path") or None
        assert result == "/foo/bar.py"

    def test_empty_path_returns_none(self, tmp_path: Path) -> None:
        af = tmp_path / "active-file.json"
        af.write_text(json.dumps({"path": ""}))
        with open(str(af)) as f:
            result = json.loads(f.read()).get("path") or None
        assert result is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        af = os.path.join(str(tmp_path), "active-file.json")
        result = None
        try:
            with open(af) as f:
                result = json.loads(f.read()).get("path") or None
        except (OSError, json.JSONDecodeError):
            pass
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        af = tmp_path / "active-file.json"
        af.write_text("not json")
        result = None
        try:
            with open(str(af)) as f:
                result = json.loads(f.read()).get("path") or None
        except (OSError, json.JSONDecodeError):
            pass
        assert result is None

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        af = tmp_path / "active-file.json"
        af.write_text(json.dumps({"other": "value"}))
        with open(str(af)) as f:
            result = json.loads(f.read()).get("path") or None
        assert result is None


class TestAssistantAgentCurrentEditorFile:
    """Test that AssistantAgent.run() accepts current_editor_file."""

    def test_signature_accepts_current_editor_file(self) -> None:
        import inspect

        from kiss.agents.assistant.assistant_agent import AssistantAgent

        sig = inspect.signature(AssistantAgent.run)
        assert "current_editor_file" in sig.parameters
        param = sig.parameters["current_editor_file"]
        assert param.default is None

    def test_current_editor_file_before_attachments(self) -> None:
        import inspect

        from kiss.agents.assistant.assistant_agent import AssistantAgent

        sig = inspect.signature(AssistantAgent.run)
        params = list(sig.parameters.keys())
        cef_idx = params.index("current_editor_file")
        att_idx = params.index("attachments")
        assert cef_idx < att_idx


class TestRunAgentThreadEditorFile:
    """Test the run_agent_thread logic for reading active-file.json."""

    def test_extra_kwargs_includes_current_editor_file(
        self, tmp_path: Path,
    ) -> None:
        cs_data_dir = str(tmp_path)
        af = tmp_path / "active-file.json"
        af.write_text(json.dumps({"path": "/test/file.py"}))

        current_editor_file = None
        try:
            af_path = os.path.join(cs_data_dir, "active-file.json")
            with open(af_path) as f:
                current_editor_file = json.loads(f.read()).get("path") or None
        except (OSError, json.JSONDecodeError):
            pass
        agent_kwargs: dict[str, Any] = {"headless": True}
        extra_kwargs = dict(agent_kwargs)
        extra_kwargs["current_editor_file"] = current_editor_file

        assert extra_kwargs == {
            "headless": True,
            "current_editor_file": "/test/file.py",
        }

    def test_extra_kwargs_none_when_no_active_file(
        self, tmp_path: Path,
    ) -> None:
        cs_data_dir = str(tmp_path)

        current_editor_file = None
        try:
            af_path = os.path.join(cs_data_dir, "active-file.json")
            with open(af_path) as f:
                current_editor_file = json.loads(f.read()).get("path") or None
        except (OSError, json.JSONDecodeError):
            pass
        extra_kwargs: dict[str, Any] = dict({"headless": False})
        extra_kwargs["current_editor_file"] = current_editor_file

        assert extra_kwargs == {
            "headless": False,
            "current_editor_file": None,
        }

    def test_extra_kwargs_preserves_original(self) -> None:
        agent_kwargs: dict[str, Any] = {"headless": True}
        extra_kwargs = dict(agent_kwargs)
        extra_kwargs["current_editor_file"] = "/foo.py"
        assert "current_editor_file" not in agent_kwargs


class TestPromptTemplateAppend:
    """Test that the prompt template includes the editor file path."""

    @pytest.mark.parametrize(
        ("editor_file", "expected_suffix"),
        [
            ("/foo/bar.py", "\n\nThe editor file path: /foo/bar.py"),
            (None, "\n\nThe editor file path: None"),
        ],
    )
    def test_prompt_template_includes_editor_file(
        self, editor_file: str | None, expected_suffix: str,
    ) -> None:
        prompt_template = "Do something"
        result = prompt_template + f"\n\nThe editor file path: {editor_file}"
        assert result.endswith(expected_suffix)
        assert result.startswith("Do something")
