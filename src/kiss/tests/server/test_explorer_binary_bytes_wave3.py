# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the Source Control file views call a file binary by its first 8 KiB of bytes.

A file is binary when a NUL is among its first 8192 bytes: the test the
working-tree side of a diff always made.  The helper both sides of a
diff (and "Open File" at a commit) share tested the first 8192
*characters* of the decoded text instead, so a UTF-8 file whose NUL
lies past its first 8 KiB, but within its first 8192 characters
(multi-byte characters before it), was refused as binary.

Real git repositories in a temporary directory; no mocks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiss.server.explorer import git_file_at, git_file_diff

_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
    "PATH": "/usr/bin:/bin:/usr/local/bin",
}

# 5000 two-byte characters put the NUL at byte 10000, character 5000.
NUL_PAST_8_KIB = "é" * 5000 + "\0 then text\n"
# 4000 two-byte characters put the NUL at byte 8000.
NUL_IN_8_KIB = "é" * 4000 + "\0 then text\n"


def _commit(repo: Path, name: str, data: bytes) -> str:
    """Commit *data* as *name* in *repo* and return the new HEAD sha."""
    (repo / name).write_bytes(data)
    for args in (("add", "-A"), ("commit", "-q", "-m", name)):
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=_ENV, capture_output=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, env=_ENV, capture_output=True, text=True,
    )
    return head.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An empty repository on branch ``main``."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True, env=_ENV)
    return tmp_path


def test_a_nul_past_the_first_8_kib_is_text_on_both_sides(repo: Path) -> None:
    sha = _commit(repo, "wide.txt", NUL_PAST_8_KIB.encode())
    assert git_file_at(str(repo), sha, "wide.txt")["text"] == NUL_PAST_8_KIB
    edited = NUL_PAST_8_KIB + "one more line\n"
    (repo / "wide.txt").write_bytes(edited.encode())
    diff = git_file_diff(str(repo), "", "wide.txt", "wide.txt")
    assert "error" not in diff
    assert (diff["original"], diff["modified"]) == (NUL_PAST_8_KIB, edited)


@pytest.mark.parametrize("prefix_bytes", [8191, 8192])
def test_the_8_kib_boundary(repo: Path, prefix_bytes: int) -> None:
    content = b"a" * prefix_bytes + b"\0"
    sha = _commit(repo, "edge.bin", content)
    shown = git_file_at(str(repo), sha, "edge.bin")
    (repo / "edge.bin").write_bytes(content + b"!")
    diff = git_file_diff(str(repo), "", "edge.bin", "edge.bin")
    if prefix_bytes < 8192:
        assert shown == {"error": "Cannot display binary file: edge.bin"}
        assert diff == {"error": "Cannot display binary file: edge.bin"}
    else:
        assert shown["text"] == content.decode()
        assert diff["modified"] == content.decode() + "!"


def test_a_nul_in_the_first_8_kib_is_binary_on_both_sides(repo: Path) -> None:
    sha = _commit(repo, "early.txt", NUL_IN_8_KIB.encode())
    refused = {"error": "Cannot display binary file: early.txt"}
    assert git_file_at(str(repo), sha, "early.txt") == refused
    (repo / "early.txt").write_bytes(b"plain now\n")
    assert git_file_diff(str(repo), "", "early.txt", "early.txt") == refused  # the HEAD side
    _commit(repo, "early.txt", b"plain now\n")
    (repo / "early.txt").write_bytes(NUL_IN_8_KIB.encode())
    assert git_file_diff(str(repo), "", "early.txt", "early.txt") == refused  # the working tree
