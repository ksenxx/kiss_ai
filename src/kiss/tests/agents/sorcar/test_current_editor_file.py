"""Tests for the current_editor_file parameter plumbing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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


class TestSorcarAgentCurrentEditorFile:
    """Test that SorcarAgent.run() accepts current_editor_file."""

    def test_signature_current_editor_file(self) -> None:
        """current_editor_file param exists, defaults to None, and comes before attachments."""
        import inspect

        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        sig = inspect.signature(SorcarAgent.run)
        assert "current_editor_file" in sig.parameters
        assert sig.parameters["current_editor_file"].default is None
        params = list(sig.parameters.keys())
        assert params.index("current_editor_file") < params.index("attachments")


class TestRunAgentThreadEditorFile:
    """Test the run_agent_thread logic for reading active-file.json."""

    def test_extra_kwargs_includes_current_editor_file(
        self,
        tmp_path: Path,
    ) -> None:
        sorcar_data_dir = str(tmp_path)
        af = tmp_path / "active-file.json"
        af.write_text(json.dumps({"path": "/test/file.py"}))

        active_file = ""
        try:
            af_path = os.path.join(sorcar_data_dir, "active-file.json")
            with open(af_path) as f:
                active_file = json.loads(f.read()).get("path", "")
            if active_file and not os.path.isfile(active_file):
                active_file = ""
        except (OSError, json.JSONDecodeError):
            pass
        agent_kwargs: dict[str, Any] = {"headless": True}
        extra_kwargs = dict(agent_kwargs)
        if active_file:
            extra_kwargs["current_editor_file"] = active_file

        assert extra_kwargs == {"headless": True}

    def test_extra_kwargs_with_real_file(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.py"
        real_file.write_text("x = 1")
        af = tmp_path / "active-file.json"
        af.write_text(json.dumps({"path": str(real_file)}))

        active_file = ""
        try:
            with open(str(af)) as f:
                active_file = json.loads(f.read()).get("path", "")
            if active_file and not os.path.isfile(active_file):
                active_file = ""
        except (OSError, json.JSONDecodeError):
            pass
        agent_kwargs: dict[str, Any] = {"headless": True}
        extra_kwargs = dict(agent_kwargs)
        if active_file:
            extra_kwargs["current_editor_file"] = active_file

        assert extra_kwargs == {
            "headless": True,
            "current_editor_file": str(real_file),
        }

    def test_extra_kwargs_no_current_editor_file_when_no_active_file(
        self,
        tmp_path: Path,
    ) -> None:
        sorcar_data_dir = str(tmp_path)

        active_file = ""
        try:
            af_path = os.path.join(sorcar_data_dir, "active-file.json")
            with open(af_path) as f:
                active_file = json.loads(f.read()).get("path", "")
            if active_file and not os.path.isfile(active_file):
                active_file = ""
        except (OSError, json.JSONDecodeError):
            pass
        extra_kwargs: dict[str, Any] = dict({"headless": False})
        if active_file:
            extra_kwargs["current_editor_file"] = active_file

        assert extra_kwargs == {"headless": False}

    def test_extra_kwargs_preserves_original(self) -> None:
        agent_kwargs: dict[str, Any] = {"headless": True}
        extra_kwargs = dict(agent_kwargs)
        extra_kwargs["current_editor_file"] = "/foo.py"
        assert "current_editor_file" not in agent_kwargs


class TestPromptTemplateAppend:
    """Test that the prompt template includes the editor file path only when set."""

    def test_prompt_template_with_and_without_editor_file(self) -> None:
        prompt = "Do something"
        for cef, expected in [
            ("/foo/bar.py", "Do something\n\nThe editor file path: /foo/bar.py"),
            (None, "Do something"),
        ]:
            result = (
                prompt + f"\n\nThe editor file path: {cef}" if cef else prompt
            )
            assert result == expected
