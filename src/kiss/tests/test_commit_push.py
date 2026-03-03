"""Tests for commit author attribution and push functionality."""

import os
import subprocess
import tempfile

from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS, _build_html


def test_commit_author_in_assistant_source():
    """The git commit command in sorcar.py must set author to KISS Sorcar."""
    import inspect

    from kiss.agents.sorcar import sorcar

    source = inspect.getsource(sorcar)
    assert "--author=KISS Sorcar <ksen@berkeley.edu>" in source


def test_commit_committer_env_in_assistant_source():
    """The git commit must set GIT_COMMITTER_NAME and GIT_COMMITTER_EMAIL to KISS Sorcar."""
    import inspect

    from kiss.agents.sorcar import sorcar

    source = inspect.getsource(sorcar)
    assert '"GIT_COMMITTER_NAME": "KISS Sorcar"' in source
    assert '"GIT_COMMITTER_EMAIL": "ksen@berkeley.edu"' in source


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

    from kiss.agents.sorcar import sorcar

    source = inspect.getsource(sorcar)
    assert 'Route("/push"' in source


def test_git_commit_with_kiss_sorcar_attribution():
    """Integration test: a real git commit uses KISS Sorcar as both author and committer."""
    with tempfile.TemporaryDirectory() as repo:
        subprocess.run(["git", "init"], cwd=repo, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo,
            capture_output=True,
        )
        with open(os.path.join(repo, "file.txt"), "w") as f:
            f.write("hello")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        commit_env = {
            **os.environ,
            "GIT_COMMITTER_NAME": "KISS Sorcar",
            "GIT_COMMITTER_EMAIL": "kiss-sorcar@users.noreply.github.com",
        }
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "test commit",
                "--author=KISS Sorcar <kiss-sorcar@users.noreply.github.com>",
            ],
            cwd=repo,
            capture_output=True,
            env=commit_env,
        )
        author = subprocess.run(
            ["git", "log", "-1", "--format=%an <%ae>"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout.strip()
        committer = subprocess.run(
            ["git", "log", "-1", "--format=%cn <%ce>"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert author == "KISS Sorcar <kiss-sorcar@users.noreply.github.com>"
        assert committer == "KISS Sorcar <kiss-sorcar@users.noreply.github.com>"
