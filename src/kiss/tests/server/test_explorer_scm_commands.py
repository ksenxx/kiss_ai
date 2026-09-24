# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``listDir`` / ``gitStatus`` / ``gitLog`` commands.

The remote webapp's activity bar (Tasks / Explorer / Source Control)
drives these three commands: the Explorer view lists folders with
``listDir`` (reply ``dirListing``), the Source Control view reads the
working tree with ``gitStatus`` and the commit graph with ``gitLog``.
Every test here talks to a REAL :class:`RemoteAccessServer` over real
``wss://`` and to a REAL git repository built in a temp dir — no mocks.
The UDS drop gate (a VS Code window never shows these views) is driven
over a real Unix-domain-socket connection.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.explorer import (
    DIR_LISTING_MAX_ENTRIES,
    GIT_LOG_DEFAULT_LIMIT,
    GIT_LOG_MAX_LIMIT,
    parse_git_log,
    parse_porcelain_status,
)
from kiss.server.web_server import (
    _PYPI_FETCH_TIMEOUT,
    RemoteAccessServer,
    _generate_self_signed_cert,
)
from kiss.tests.conftest import is_root, posix_only, requires_unix_sockets


def _find_free_port() -> int:
    """Find an available TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _git(repo: Path, *args: str) -> str:
    """Run ``git *args`` in *repo* with a scrubbed, deterministic environment.

    Author/committer identity and dates are pinned, the user's global
    and system git config are ignored (no signing, hooks or templates
    leak in), and repo-scoped ``GIT_*`` variables of the calling
    process are dropped.  Returns git's stdout.
    """
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("GIT_") or k == "GIT_EXEC_PATH"
    }
    env.update(
        {
            "GIT_AUTHOR_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "author@example.com",
            "GIT_COMMITTER_NAME": "Test Author",
            "GIT_COMMITTER_EMAIL": "author@example.com",
            "GIT_AUTHOR_DATE": "2026-01-02T03:04:05+00:00",
            "GIT_COMMITTER_DATE": "2026-01-02T03:04:05+00:00",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
         *args],
        cwd=repo, env=env, check=True, capture_output=True, text=True,
        timeout=60,
    )
    return result.stdout


def build_repo(repo: Path) -> dict[str, str]:
    """Create a small repository with two commits, a merge and a dirty tree.

    History (newest first)::

        merge   Merge branch 'feature'          (2 parents)
        feat    feature: add feature.txt        (on branch feature)
        second  second: rename a.txt -> b.txt, add dir/nested.py
        first   first: add a.txt, README.md

    Working tree after the commits: ``README.md`` modified (unstaged),
    ``dir/nested.py`` modified and staged, ``untracked.txt`` untracked,
    ``b.txt`` deleted (unstaged).

    Returns:
        A mapping of commit labels to full shas.
    """
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("a\n")
    (repo / "README.md").write_text("# readme\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first: add a.txt, README.md")
    first = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "mv", "a.txt", "b.txt")
    (repo / "dir").mkdir()
    (repo / "dir" / "nested.py").write_text("x = 1  # nested-sentinel-4f2a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second: rename a.txt -> b.txt")
    second = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "feature.txt").write_text("feature-file-sentinel-7c1e\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature: add feature.txt")
    feat = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "main")
    (repo / "main-only.txt").write_text("m\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main: add main-only.txt")
    main_only = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge branch 'feature'", "feature")
    merge = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "tag", "v1")
    # Dirty working tree.
    (repo / "README.md").write_text("# readme changed\n")
    (repo / "dir" / "nested.py").write_text("x = 2  # nested-sentinel-4f2a\n")
    _git(repo, "add", "dir/nested.py")
    (repo / "untracked.txt").write_text("u\n")
    (repo / "b.txt").unlink()
    return {
        "first": first,
        "second": second,
        "feat": feat,
        "main_only": main_only,
        "merge": merge,
    }


class ExplorerHarness:
    """A real RemoteAccessServer whose work dir is a real git repository."""

    def __init__(self) -> None:
        # Canonical (symlink-free): the daemon reports paths it has
        # resolved (fileContent/fileSaved, git worktree roots), so a
        # symlinked temp dir (macOS /var -> /private/var) would make
        # the paths tests compare differ from the ones they created.
        self.tmpdir = os.path.realpath(tempfile.mkdtemp(prefix="kiss-explorer-scm-"))
        tmp = Path(self.tmpdir)
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = tmp / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = tmp / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.work_dir = tmp / "repo"
        self.work_dir.mkdir()
        self.shas = build_repo(self.work_dir)
        # A plain (non-git) folder beside the repo, and hidden entries
        # the Explorer must skip.
        self.plain_dir = tmp / "plain"
        self.plain_dir.mkdir()
        (self.plain_dir / "only.txt").write_text("only\n")
        # Excluded from git status (info/exclude) so the Source Control
        # tests see exactly the four changes build_repo left behind; the
        # Explorer must skip it on its own.
        (self.work_dir / ".DS_Store").write_bytes(b"\x00")
        (self.work_dir / ".git" / "info" / "exclude").write_text(".DS_Store\n")

        certfile = tmp / "cert.pem"
        keyfile = tmp / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _find_free_port()
        self.base_url = f"https://127.0.0.1:{self.port}"
        self.ws_url = f"wss://127.0.0.1:{self.port}/ws"
        self.uds_path = tmp / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp / "remote-url.json",
            uds_path=self.uds_path,
            work_dir=str(self.work_dir),
        )
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self._thread.start()
        asyncio.run_coroutine_threadsafe(
            self.server.start_async(), self.loop,
        ).result(60)
        # Let the daemon's startup PyPI check cache the latest release
        # before any client connects, so Playwright tests can tell from
        # ``server._latest_version`` whether an update toast is due on
        # every page they open (it then arrives in the connection's
        # welcome sequence, right behind the config reply).  Bounded by
        # the fetch timeout; with PyPI unreachable no toast ever comes.
        deadline = time.monotonic() + _PYPI_FETCH_TIMEOUT + 1
        while (
            self.server._latest_version is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run *coro* on the server loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(60)

    def stop(self) -> None:
        """Stop the server, its loop, and restore redirected globals."""
        try:
            asyncio.run_coroutine_threadsafe(
                self.server.stop_async(), self.loop,
            ).result(60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=30)
            self.loop.close()
            if th._db_conn is not None:
                th._db_conn.close()
            th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)


_LONG_REPO_COMMITS = GIT_LOG_MAX_LIMIT + 5


def _long_repo(harness: ExplorerHarness) -> Path:
    """A repository with more commits than the gitLog hard cap (built
    once per module, on first use, through ``git fast-import`` so the
    500+ commits take well under a second)."""
    repo = Path(harness.tmpdir) / "long-repo"
    if repo.exists():
        return repo
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    stream = []
    for i in range(_LONG_REPO_COMMITS):
        msg = f"commit {i}\n".encode()
        data = f"{i}\n".encode()
        stream.append(
            b"commit refs/heads/main\n"
            b"committer Test Author <author@example.com> "
            + str(1_800_000_000 + i).encode() + b" +0000\n"
            b"data " + str(len(msg)).encode() + b"\n" + msg
            # Consecutive commits on one branch chain automatically.
            + b"M 100644 inline counter.txt\n"
            b"data " + str(len(data)).encode() + b"\n" + data + b"\n"
        )
    subprocess.run(
        ["git", "fast-import", "--quiet"], cwd=repo, input=b"".join(stream),
        check=True, capture_output=True, timeout=120,
    )
    _git(repo, "checkout", "-q", "main")
    return repo


@pytest.fixture(scope="module")
def harness():
    """One shared real server + git repo for every test in this module."""
    h = ExplorerHarness()
    yield h
    h.stop()


async def _ws_request(
    harness: ExplorerHarness, payload: dict, reply_type: str,
) -> dict:
    """Authenticate over wss://, send *payload*, return the first
    *reply_type* event."""
    async with connect(harness.ws_url, ssl=_no_verify_ssl()) as ws:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == "auth_ok":
                break
        await ws.send(json.dumps(payload))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == reply_type:
                reply: dict = msg
                return reply


def _request(harness: ExplorerHarness, payload: dict, reply_type: str) -> dict:
    reply: dict = harness.run(_ws_request(harness, payload, reply_type))
    return reply


class TestListDir:
    """``listDir`` -> ``dirListing`` over a real wss:// connection."""

    def test_empty_path_lists_the_work_dir(self, harness) -> None:
        reply = _request(
            harness, {"type": "listDir", "tabId": "t-1", "token": "1:root"},
            "dirListing",
        )
        assert reply["tabId"] == "t-1"
        assert reply["token"] == "1:root"
        assert reply["root"] == str(harness.work_dir)
        assert Path(reply["path"]) == harness.work_dir.resolve()
        assert reply["truncated"] is False
        names = [e["name"] for e in reply["entries"]]
        # Folders first, then files, each case-insensitively sorted
        # (README.md sorts among the lower-case names); the .git folder
        # and .DS_Store are excluded, and b.txt is deleted in the tree.
        assert names == [
            "dir", "feature.txt", "main-only.txt", "README.md",
            "untracked.txt",
        ]
        by_name = {e["name"]: e for e in reply["entries"]}
        assert by_name["dir"]["isDir"] is True
        assert by_name["dir"]["path"] == str(harness.work_dir.resolve() / "dir")
        assert by_name["README.md"]["isDir"] is False

    def test_relative_path_resolves_against_work_dir(self, harness) -> None:
        reply = _request(
            harness, {"type": "listDir", "path": "dir", "token": "2:dir"},
            "dirListing",
        )
        assert "error" not in reply
        assert Path(reply["path"]) == (harness.work_dir / "dir").resolve()
        assert [e["name"] for e in reply["entries"]] == ["nested.py"]

    def test_absolute_path_outside_work_dir(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "listDir", "path": str(harness.plain_dir), "token": "3:p"},
            "dirListing",
        )
        assert [e["name"] for e in reply["entries"]] == ["only.txt"]

    def test_explicit_work_dir_field_wins(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "listDir", "workDir": str(harness.plain_dir)},
            "dirListing",
        )
        assert reply["root"] == str(harness.plain_dir)
        assert [e["name"] for e in reply["entries"]] == ["only.txt"]

    def test_missing_directory_replies_error(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "listDir", "path": "no/such/dir", "token": "4:x"},
            "dirListing",
        )
        assert reply["error"].startswith("Directory not found: no/such/dir")
        assert reply["token"] == "4:x"
        assert "entries" not in reply

    def test_file_path_is_not_a_directory(self, harness) -> None:
        reply = _request(
            harness, {"type": "listDir", "path": "README.md"}, "dirListing",
        )
        assert reply["error"] == "Directory not found: README.md"

    def test_non_string_fields_are_blanked(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "listDir", "path": 123, "tabId": 5, "token": ["x"]},
            "dirListing",
        )
        assert reply["tabId"] == ""
        assert reply["token"] == ""
        assert Path(reply["path"]) == harness.work_dir.resolve()

    @posix_only("chmod permission bits")
    def test_unreadable_directory_replies_error(self, harness) -> None:
        if is_root():
            pytest.skip("root ignores directory permissions")
        locked = harness.work_dir / "locked"
        locked.mkdir()
        locked.chmod(0o000)
        try:
            reply = _request(
                harness, {"type": "listDir", "path": "locked"}, "dirListing",
            )
        finally:
            locked.chmod(0o755)
            locked.rmdir()
        assert reply["error"].startswith("Failed to list locked:")

    def test_symlinked_folder_reports_its_real_target(self, harness) -> None:
        """A symlinked folder carries ``real`` (its resolved target) so the
        Explorer can spot a link back into its own ancestry; plain
        folders and files carry none."""
        link = harness.work_dir / "dir" / "up"
        link.symlink_to(harness.work_dir, target_is_directory=True)
        try:
            reply = _request(
                harness, {"type": "listDir", "path": "dir"}, "dirListing",
            )
        finally:
            link.unlink()
        by_name = {e["name"]: e for e in reply["entries"]}
        assert by_name["up"]["isDir"] is True
        assert Path(by_name["up"]["real"]) == harness.work_dir.resolve()
        assert "real" not in by_name["nested.py"]
        root = _request(harness, {"type": "listDir"}, "dirListing")
        assert "real" not in {e["name"]: e for e in root["entries"]}["dir"]

    def test_listing_is_truncated_past_the_cap(self, harness) -> None:
        big = harness.work_dir / "big"
        big.mkdir()
        try:
            for i in range(DIR_LISTING_MAX_ENTRIES + 5):
                (big / f"f{i:05d}").write_text("")
            reply = _request(
                harness, {"type": "listDir", "path": "big"}, "dirListing",
            )
        finally:
            shutil.rmtree(big)
        assert reply["truncated"] is True
        assert len(reply["entries"]) == DIR_LISTING_MAX_ENTRIES


class TestGitStatus:
    """``gitStatus`` over a real wss:// connection against a real repo."""

    def test_changes_grouped_like_vscode(self, harness) -> None:
        reply = _request(
            harness, {"type": "gitStatus", "tabId": "t-2", "token": "7"},
            "gitStatus",
        )
        assert reply["tabId"] == "t-2"
        assert reply["token"] == "7"
        assert reply["workDir"] == str(harness.work_dir)
        assert Path(reply["repo"]) == harness.work_dir.resolve()
        assert reply["branch"] == "main"
        rows = {(c["path"], c["group"]): c for c in reply["changes"]}
        assert rows[("README.md", "changes")]["status"] == "M"
        assert rows[("dir/nested.py", "staged")]["status"] == "M"
        assert rows[("untracked.txt", "changes")]["status"] == "U"
        assert rows[("b.txt", "changes")]["status"] == "D"
        assert len(rows) == 4
        for c in reply["changes"]:
            assert c["absPath"] == str(Path(reply["repo"]) / c["path"])

    def test_not_a_repository_replies_error(self, harness) -> None:
        reply = _request(
            harness,
            {"type": "gitStatus", "workDir": str(harness.plain_dir)},
            "gitStatus",
        )
        assert reply["error"] == f"Not a git repository: {harness.plain_dir}"
        assert "changes" not in reply

    def test_missing_work_dir_replies_error(self, harness) -> None:
        missing = str(Path(harness.tmpdir) / "nope")
        reply = _request(
            harness, {"type": "gitStatus", "workDir": missing}, "gitStatus",
        )
        assert reply["error"] == f"Directory not found: {missing}"

    def test_staged_and_unstaged_same_file_appears_in_both_groups(
        self, harness,
    ) -> None:
        # dir/nested.py is staged as modified; modify it again unstaged.
        target = harness.work_dir / "dir" / "nested.py"
        original = target.read_text()
        target.write_text("x = 3\n")
        try:
            reply = _request(harness, {"type": "gitStatus"}, "gitStatus")
        finally:
            target.write_text(original)
        groups = sorted(
            c["group"] for c in reply["changes"] if c["path"] == "dir/nested.py"
        )
        assert groups == ["changes", "staged"]


class TestGitLog:
    """``gitLog`` over a real wss:// connection against a real repo."""

    def test_commits_newest_first_with_files_refs_and_parents(
        self, harness,
    ) -> None:
        reply = _request(
            harness, {"type": "gitLog", "tabId": "t-3", "token": "9"}, "gitLog",
        )
        assert reply["tabId"] == "t-3"
        assert reply["token"] == "9"
        assert reply["head"] == harness.shas["merge"]
        commits = reply["commits"]
        assert [c["sha"] for c in commits][:1] == [harness.shas["merge"]]
        assert {c["sha"] for c in commits} == set(harness.shas.values())
        by_sha = {c["sha"]: c for c in commits}
        merge = by_sha[harness.shas["merge"]]
        assert merge["parents"] == [
            harness.shas["main_only"], harness.shas["feat"],
        ]
        assert merge["subject"] == "Merge branch 'feature'"
        assert merge["shortSha"] == harness.shas["merge"][:7]
        assert merge["author"] == "Test Author"
        assert merge["date"].startswith("20")
        assert "HEAD -> main" in merge["refs"]
        assert "tag: v1" in merge["refs"]
        # A merge lists its changes against its FIRST parent (main),
        # i.e. what the feature branch brought in.
        assert merge["files"] == [{"status": "A", "path": "feature.txt"}]
        # git prints %aI as "+00:00" or, in newer releases, "Z".
        assert datetime.fromisoformat(merge["date"]) == datetime.fromisoformat(
            "2026-01-02T03:04:05+00:00",
        )
        second = by_sha[harness.shas["second"]]
        files = {f["path"]: f for f in second["files"]}
        assert files["b.txt"]["status"] == "R"
        assert files["b.txt"]["origPath"] == "a.txt"
        assert files["dir/nested.py"]["status"] == "A"
        first = by_sha[harness.shas["first"]]
        assert first["parents"] == []
        assert sorted(f["path"] for f in first["files"]) == [
            "README.md", "a.txt",
        ]
        assert first["refs"] == []
        feat = by_sha[harness.shas["feat"]]
        assert feat["refs"] == ["feature"]
        # --date-order: no parent before all of its children.
        seen: set[str] = set()
        for c in commits:
            for p in c["parents"]:
                assert p not in seen
            seen.add(c["sha"])

    def test_limit_caps_the_commit_count(self, harness) -> None:
        reply = _request(harness, {"type": "gitLog", "limit": 2}, "gitLog")
        assert len(reply["commits"]) == 2
        assert reply["commits"][0]["sha"] == harness.shas["merge"]

    @pytest.mark.parametrize("bad", [0, -3, True, "5", None, 2.5])
    def test_invalid_limit_falls_back_to_default(self, harness, bad) -> None:
        """Junk limits fall back to the default, which really is 50: a
        repository with more commits than that is cut at 50."""
        long_repo = _long_repo(harness)
        reply = _request(
            harness, {"type": "gitLog", "workDir": str(long_repo), "limit": bad},
            "gitLog",
        )
        assert len(reply["commits"]) == GIT_LOG_DEFAULT_LIMIT

    def test_huge_limit_is_clamped(self, harness) -> None:
        """A limit above the cap returns at most GIT_LOG_MAX_LIMIT commits,
        and a limit between default and cap is honoured as given."""
        long_repo = _long_repo(harness)
        reply = _request(
            harness,
            {"type": "gitLog", "workDir": str(long_repo),
             "limit": GIT_LOG_MAX_LIMIT * 10},
            "gitLog",
        )
        assert len(reply["commits"]) == GIT_LOG_MAX_LIMIT
        reply = _request(
            harness,
            {"type": "gitLog", "workDir": str(long_repo),
             "limit": GIT_LOG_DEFAULT_LIMIT + 3},
            "gitLog",
        )
        assert len(reply["commits"]) == GIT_LOG_DEFAULT_LIMIT + 3

    def test_legacy_merge_diff_path_matches_first_parent_output(
        self, harness,
    ) -> None:
        """The pre-2.31 ``-m`` fallback (forced here) yields the same
        rows as ``--diff-merges=first-parent``: one row per commit, the
        merge listing its first-parent files."""
        from kiss.server.explorer import git_log

        modern = git_log(str(harness.work_dir), 20)
        legacy = git_log(str(harness.work_dir), 20, legacy_merge_diffs=True)
        assert legacy == modern
        shas = [c["sha"] for c in legacy["commits"]]
        assert len(shas) == len(set(shas)) == 5
        merge = next(c for c in legacy["commits"] if c["sha"] == harness.shas["merge"])
        assert merge["files"] == [{"status": "A", "path": "feature.txt"}]

    @posix_only("a tab is not a valid NTFS file-name character")
    def test_unusual_names_come_through_verbatim(self, harness) -> None:
        """Tabs and quotes in file names, a comma in a branch name and a
        control character in a subject all survive the wire."""
        repo = Path(harness.tmpdir) / "odd-repo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        (repo / "tab\tname.txt").write_text("t\n")
        (repo / 'q"uote.txt').write_text("q\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "odd names")
        _git(repo, "checkout", "-q", "-b", "topic,comma")
        (repo / "plain.txt").write_text("p\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "subject with \x1f and \x1e inside")
        (repo / "tab\tname.txt").write_text("changed\n")
        reply = _request(
            harness, {"type": "gitLog", "workDir": str(repo)}, "gitLog",
        )
        top, root = reply["commits"]
        assert top["subject"] == "subject with \x1f and \x1e inside"
        assert top["refs"] == ["HEAD -> topic,comma"]
        assert top["files"] == [{"status": "A", "path": "plain.txt"}]
        assert sorted(f["path"] for f in root["files"]) == [
            'q"uote.txt', "tab\tname.txt",
        ]
        assert root["refs"] == ["main"]
        status = _request(
            harness, {"type": "gitStatus", "workDir": str(repo)}, "gitStatus",
        )
        assert status["branch"] == "topic,comma"
        assert [c["path"] for c in status["changes"]] == ["tab\tname.txt"]
        listing = _request(
            harness, {"type": "listDir", "workDir": str(repo)}, "dirListing",
        )
        assert "tab\tname.txt" in [e["name"] for e in listing["entries"]]

    @posix_only("Windows strips trailing spaces from directory names")
    def test_repo_named_with_trailing_space_is_not_confused(self, harness) -> None:
        """``repo `` (trailing space) beside ``repo`` must report ITS own
        history, not its sibling's."""
        spaced = Path(harness.tmpdir) / "repo "
        spaced.mkdir()
        _git(spaced, "init", "-q", "-b", "spaced")
        (spaced / "s.txt").write_text("s\n")
        _git(spaced, "add", "-A")
        _git(spaced, "commit", "-q", "-m", "only in spaced")
        reply = _request(
            harness, {"type": "gitLog", "workDir": str(spaced)}, "gitLog",
        )
        assert Path(reply["repo"]) == spaced.resolve()
        assert [c["subject"] for c in reply["commits"]] == ["only in spaced"]
        status = _request(
            harness, {"type": "gitStatus", "workDir": str(spaced)}, "gitStatus",
        )
        assert status["branch"] == "spaced"
        assert Path(status["repo"]) == spaced.resolve()

    def test_nul_byte_in_path_replies_error(self, harness) -> None:
        """A path with an embedded NUL cannot be resolved; the view must
        still get a reply rather than a silently dropped command."""
        reply = _request(
            harness, {"type": "listDir", "path": "bad\x00dir", "token": "n"},
            "dirListing",
        )
        assert reply["token"] == "n"
        assert reply["error"].startswith("Directory not found: bad")
        nul_wd = str(harness.work_dir) + "\x00x"
        status = _request(
            harness, {"type": "gitStatus", "workDir": nul_wd}, "gitStatus",
        )
        assert status["error"].startswith("Directory not found:")
        log = _request(harness, {"type": "gitLog", "workDir": nul_wd}, "gitLog")
        assert log["error"].startswith("Directory not found:")

    def test_not_a_repository_replies_error(self, harness) -> None:
        reply = _request(
            harness, {"type": "gitLog", "workDir": str(harness.plain_dir)},
            "gitLog",
        )
        assert reply["error"] == f"Not a git repository: {harness.plain_dir}"

    def test_missing_work_dir_replies_error(self, harness) -> None:
        missing = str(Path(harness.tmpdir) / "nope")
        reply = _request(
            harness, {"type": "gitLog", "workDir": missing}, "gitLog",
        )
        assert reply["error"] == f"Directory not found: {missing}"

    def test_empty_repository_has_no_commits(self, harness) -> None:
        empty = Path(harness.tmpdir) / "empty-repo"
        empty.mkdir()
        _git(empty, "init", "-q", "-b", "main")
        reply = _request(
            harness, {"type": "gitLog", "workDir": str(empty)}, "gitLog",
        )
        assert reply["commits"] == []
        assert reply["head"] == ""
        assert Path(reply["repo"]) == empty.resolve()
        status = _request(
            harness, {"type": "gitStatus", "workDir": str(empty)}, "gitStatus",
        )
        assert status["changes"] == []
        assert status["branch"] == "main"


@requires_unix_sockets
class TestUdsDropGate:
    """A VS Code window (UDS) never gets a reply to these commands."""

    def test_uds_delivered_commands_are_dropped(self, harness) -> None:
        async def _probe() -> list[dict]:
            reader, writer = await asyncio.open_unix_connection(
                str(harness.uds_path)
            )
            try:
                for cmd in (
                    {"type": "listDir", "tabId": "u-1"},
                    {"type": "gitStatus", "tabId": "u-1"},
                    {"type": "gitLog", "tabId": "u-1"},
                    # Positive control, sent LAST: getInputHistory IS
                    # answered over UDS (inputHistory).  Its reply
                    # arriving proves the three commands before it were
                    # processed — and dropped — rather than still queued.
                    {"type": "getInputHistory", "tabId": "u-1"},
                ):
                    writer.write((json.dumps(cmd) + "\n").encode())
                await writer.drain()
                got: list[dict] = []
                deadline = asyncio.get_event_loop().time() + 15
                while asyncio.get_event_loop().time() < deadline:
                    line = await asyncio.wait_for(reader.readline(), 15)
                    if not line:
                        break
                    ev = json.loads(line)
                    got.append(ev)
                    if ev.get("type") == "inputHistory":
                        break
                return got
            finally:
                writer.close()
                await writer.wait_closed()

        events = harness.run(_probe())
        types = [e.get("type") for e in events]
        assert "inputHistory" in types, events
        assert not set(types) & {"dirListing", "gitStatus", "gitLog"}, events


class TestParsers:
    """The pure parsers, fed exact git output shapes."""

    def test_porcelain_rename_copy_and_conflicts(self) -> None:
        text = (
            "R  new.txt\0old.txt\0"
            " C copy.txt\0src.txt\0"
            "UU conflict.txt\0"
            "?? fresh.txt\0"
            "MM both.txt\0"
            "A  added.txt\0"
            "abc\0"  # too short: skipped
        )
        rows = parse_porcelain_status(text, "/repo")
        as_tuples = [(r["path"], r["group"], r["status"], r.get("origPath")) for r in rows]
        assert as_tuples == [
            ("new.txt", "staged", "R", "old.txt"),
            ("copy.txt", "changes", "C", "src.txt"),
            ("conflict.txt", "merge", "!", None),
            ("fresh.txt", "changes", "U", None),
            ("both.txt", "staged", "M", None),
            ("both.txt", "changes", "M", None),
            ("added.txt", "staged", "A", None),
        ]
        assert rows[0]["absPath"] == str(Path("/repo") / "new.txt")

    def test_porcelain_rename_missing_orig_token_at_end(self) -> None:
        rows = parse_porcelain_status("R  new.txt", "/repo")
        assert rows == [
            {"path": "new.txt", "absPath": str(Path("/repo") / "new.txt"), "status": "R",
             "group": "staged"},
        ]

    def test_git_log_records(self) -> None:
        """The exact ``git log -z --name-status`` token stream: seven header
        tokens per commit (the last the full ``%B`` message), then
        ``\\nSTATUS`` / path pairs (rename triples); a file-less commit
        runs straight into the next header, and a truncated trailing
        entry is dropped rather than misread."""
        sha_a, sha_b = "a" * 40, "b" * 40
        text = "\0".join(
            [
                sha_a, sha_b + " " + "c" * 40, "Ann", "2026-01-02T03:04:05+00:00",
                "HEAD -> main, tag: v1, topic,comma", "Merge \x1f it",
                "Merge \x1f it\n\nBody line one.\nBody line two.\n",
                # merge without files: next header follows directly
                sha_b, "", "Bob", "2026-01-01T00:00:00+00:00", "", "Root commit",
                "Root commit\n",
                "\nA", "new.py", "R100", "old.py", "new2.py", "M", "dir/x\ty.py",
                "M",  # truncated: status without its path
            ]
        ) + "\0"
        commits = parse_git_log(text)
        assert len(commits) == 2
        assert commits[0]["parents"] == [sha_b, "c" * 40]
        assert commits[0]["refs"] == ["HEAD -> main", "tag: v1", "topic,comma"]
        assert commits[0]["subject"] == "Merge \x1f it"
        assert commits[0]["message"] == "Merge \x1f it\n\nBody line one.\nBody line two."
        assert commits[0]["files"] == []
        assert commits[1]["message"] == "Root commit"
        assert commits[1]["parents"] == []
        assert commits[1]["refs"] == []
        assert commits[1]["files"] == [
            {"status": "A", "path": "new.py"},
            {"status": "R", "origPath": "old.py", "path": "new2.py"},
            {"status": "M", "path": "dir/x\ty.py"},
        ]
        assert commits[1]["shortSha"] == "b" * 7
        assert parse_git_log("") == []
        assert parse_git_log("not a sha\0x\0y\0z\0w\0v\0u\0") == []
