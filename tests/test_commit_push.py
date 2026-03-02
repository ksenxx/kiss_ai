"""Tests for commit author attribution and push functionality."""

from kiss.agents.assistant.chatbot_ui import CHATBOT_JS, _build_html


def test_commit_author_in_assistant_source():
    """The git commit command in assistant.py must set author to KISS Sorcar."""
    import inspect
    from kiss.agents.assistant import assistant

    source = inspect.getsource(assistant)
    assert "--author=KISS Sorcar <kiss-sorcar@users.noreply.github.com>" in source


def test_push_button_in_html():
    """The merge toolbar must include a Push button."""
    html = _build_html("Test", "", "/tmp")
    assert 'id="push-btn"' in html
    assert "mergePush()" in html


def test_push_js_function_exists():
    """The JS must define mergePush function that calls /push endpoint."""
    assert "function mergePush()" in CHATBOT_JS
    assert "fetch('/push'" in CHATBOT_JS


def test_commit_button_still_exists():
    """The commit button must still be present alongside push."""
    html = _build_html("Test", "", "/tmp")
    assert 'id="commit-btn"' in html
    assert "mergeCommit()" in html


def test_push_button_shows_pushing_state():
    """Push button should show 'Pushing...' text while in progress."""
    assert "Pushing..." in CHATBOT_JS


def test_push_route_in_assistant_source():
    """The /push route must be registered in the Starlette app."""
    import inspect
    from kiss.agents.assistant import assistant

    source = inspect.getsource(assistant)
    assert 'Route("/push"' in source
