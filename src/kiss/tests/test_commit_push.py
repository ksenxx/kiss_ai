"""Tests for commit author attribution and push functionality."""

import os
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS, _build_html
from kiss.agents.sorcar.sorcar import _resolve_requested_file_path


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


def test_auth_route_in_assistant_source():
    """The /auth route must be registered for auth panel status/actions."""
    import inspect

    from kiss.agents.sorcar import sorcar

    source = inspect.getsource(sorcar)
    assert 'Route("/auth"' in source


def test_resolve_requested_file_path_relative():
    """Relative paths should resolve under work_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        resolved = _resolve_requested_file_path("a/b.txt", tmpdir)
        assert resolved == str((Path(tmpdir) / "a" / "b.txt").resolve())


def test_resolve_requested_file_path_windows_absolute_like():
    """Drive-letter paths should stay absolute instead of being joined to work_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        win_abs = r"C:\Users\Test\file.txt"
        resolved = _resolve_requested_file_path(win_abs, tmpdir)
        if os.name == "nt":
            assert resolved.lower() == os.path.abspath(win_abs).lower()
        else:
            # Non-Windows treats this as relative text; preserve current platform semantics.
            assert resolved == os.path.abspath(os.path.join(tmpdir, win_abs))


def test_resolve_requested_file_path_git_bash_style_drive_path():
    """Git-Bash style drive paths (/c/...) should normalize to native Windows absolute paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        git_bash_abs = "/c/Users/Test/file.txt"
        resolved = _resolve_requested_file_path(git_bash_abs, tmpdir)
        if os.name == "nt":
            assert resolved.lower() == os.path.abspath(r"C:\Users\Test\file.txt").lower()
        else:
            assert resolved == os.path.abspath(git_bash_abs)


def test_resolve_requested_file_path_wsl_mnt_drive_path():
    """WSL-style drive paths (/mnt/c/...) should also normalize on Windows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wsl_abs = "/mnt/c/Users/Test/file.txt"
        resolved = _resolve_requested_file_path(wsl_abs, tmpdir)
        if os.name == "nt":
            assert resolved.lower() == os.path.abspath(r"C:\Users\Test\file.txt").lower()
        else:
            assert resolved == os.path.abspath(wsl_abs)


def test_resolve_requested_file_path_drive_like_without_leading_slash_is_relative():
    """`c/...` stays relative; only `/c/...` or `C:\\...` is treated as drive-absolute."""
    with tempfile.TemporaryDirectory() as tmpdir:
        drive_like_relative = "c/temp/test.txt"
        resolved = _resolve_requested_file_path(drive_like_relative, tmpdir)
        assert resolved == os.path.abspath(os.path.join(tmpdir, drive_like_relative))


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
