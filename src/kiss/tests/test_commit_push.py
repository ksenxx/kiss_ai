"""Tests for commit author attribution."""

import os
import subprocess
import tempfile


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
