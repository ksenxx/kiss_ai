"""Tests for the current_editor_file parameter plumbing."""

from __future__ import annotations


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
