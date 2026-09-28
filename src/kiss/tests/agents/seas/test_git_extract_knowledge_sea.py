# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``/git_extract_knowledge`` SEA.

Every test builds a real git repository with ``git`` and indexes it into
the test session's temporary ``KISS_HOME`` (see ``conftest.py``); the
block store, the memory pages, the cron job file and the CLI are all
the real ones.  The agent-level test runs a real
:class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server configured from the SEA's getters, so the
index report and the page write really flow through tool results.
"""

from __future__ import annotations

import errno
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas.git_extract_knowledge import git_extract_knowledge_sea as sea
from kiss.agents.seas.git_extract_knowledge import git_knowledge_index as index
from kiss.agents.seas.git_extract_knowledge.git_knowledge_store import (
    KINDS,
    Block,
    KnowledgeStore,
    query_tokens,
)
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs
from kiss.core.config import kiss_home
from kiss.core.memoryfield.pages import MemoryDir
from kiss.server.agent_file import apply_agent_overrides
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(sea.__file__).resolve()
_TESTS = Path(__file__).parent

BANK_PY = '''"""Ledger of bank accounts."""
import os
from decimal import Decimal


def open_account(owner):
    """Create an account for *owner*."""
    return {"owner": owner, "balance": Decimal(0)}


class Ledger:
    """Holds every account."""

    def deposit(self, account, amount):
        account["balance"] += amount
'''
WITHDRAW = "\n\ndef withdraw(account, amount):\n    account['balance'] -= amount\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _init(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "alice@example.com")
    _git(repo, "config", "user.name", "Alice")
    return repo


def _make_repo(root: Path, name: str = "ledger") -> Path:
    """Create a repository with three commits, a tag, a branch and mixed file kinds."""
    repo = _init(root, name)
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "README.md").write_text("# Bank Ledger\n\nA ledger tracks accounts.\n")
    (repo / "src" / "pkg" / "bank.py").write_text(BANK_PY)
    (repo / "docs" / "guide.md").write_text("# Guide\n\n## Deposits\nUse the ledger.\n")
    (repo / "docs" / "readme.bin").write_bytes(b"\x00\xff")  # a binary README is not quoted
    (repo / "logo.bin").write_bytes(b"\x00\x01\x02binary")
    (repo / "app.min.js").write_text("var a=1;" * 50)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial ledger\n\nStarts the accounts domain.")
    (repo / "src" / "pkg" / "bank.py").write_text(BANK_PY + WITHDRAW)
    _git(repo, "commit", "-q", "-am", "add withdraw")
    _git(repo, "tag", "-a", "v0.1", "-m", "first release")
    _git(repo, "branch", "feature/audit")
    (repo / "docs" / "notes.txt").write_text("interest rates\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "notes on interest")
    return repo


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fresh ``ledger`` repository whose memory (shared by slug) starts empty."""
    shutil.rmtree(kiss_home() / "memories" / "ledger", ignore_errors=True)
    return _make_repo(tmp_path)


def _memory_dir(repo: Path) -> Path:
    return index.memory_location(repo)[1]


def _store(repo: Path) -> KnowledgeStore:
    return KnowledgeStore(index.store_path(repo))


def _keys(hits: list[Any]) -> list[str]:
    return [hit.block.key for hit in hits]


def test_sea_getters_follow_the_contract(tmp_path: Path) -> None:
    """The getters pin the run: full tools + knowledge tools, no worktree, no web, no memory."""
    assert sea.tool_profile() == "full"
    assert sea.is_parallel() is True
    assert sea.use_worktree() is False
    assert sea.auto_commit() is False
    assert sea.classify_tasks() is False
    assert sea.use_web_tools() is False
    assert sea.use_memory() is False
    names = [tool.__name__ for tool in sea.tools()]
    assert names == [
        "index_repo", "knowledge_status", "knowledge_search", "knowledge_read",
        "list_knowledge_pages", "read_knowledge_page", "search_knowledge_pages",
        "write_knowledge_page", "delete_knowledge_page", "schedule_daily_update",
    ]
    prompt = sea.system_prompt()
    assert "{python}" not in prompt and "{module}" not in prompt
    assert f"{shlex.quote(sys.executable)} -m {sea.MODULE} write-page" in prompt
    assert "ABSOLUTE path" in prompt
    for page in ("overview", "domain-glossary", "architecture", "history", "faq"):
        assert f"`{page}`" in prompt
    # The real loader accepts the file: ``tools()`` returning callables makes
    # the SEA its own tools file.
    cmd: dict[str, Any] = {"agentPath": str(_SEA_PATH), "workDir": str(tmp_path)}
    apply_agent_overrides(cmd)
    assert cmd["toolsFile"] == str(_SEA_PATH)
    assert cmd["toolProfile"] == "full"
    assert cmd["useWorktree"] is False
    assert cmd["systemPrompt"] == prompt


def test_slash_command_resolves_to_the_bundled_sea() -> None:
    """``/git_extract_knowledge <repo>`` dispatches this SEA through ``run_agent``."""
    assert sea_commands.get_command("git_extract_knowledge") == _SEA_PATH
    rewritten = sea_commands.rewrite_prompt_if_command("/git_extract_knowledge /tmp/repo")
    assert rewritten is not None
    assert rewritten[1] == _SEA_PATH
    assert rewritten[0].endswith("TASK TEXT FOR run_agent:\n/tmp/repo")


def test_full_index_builds_every_block_kind_and_the_lookup_page(repo: Path) -> None:
    """The first run indexes files, chunks, symbols, commits, changes, refs, authors, dirs."""
    report = sea.index_repo(str(repo / "src"))  # any directory inside the work tree
    assert not report.startswith("Error"), report
    assert "mode: full" in report
    assert f"repository: {repo}" in report
    store = _store(repo)
    counts = store.counts()
    assert set(counts) == set(KINDS), counts
    assert counts["file"] == 7 and counts["commit"] == 3 and counts["tag"] == 1
    assert counts["branch"] == 2 and counts["author"] == 1 and counts["repo"] == 1

    bank = store.get("file:src/pkg/bank.py")
    assert bank is not None and bank.sha == _git(repo, "rev-parse", "HEAD:src/pkg/bank.py")
    assert "language: Python" in bank.text
    assert "symbols: def open_account, class Ledger, def deposit, def withdraw" in bank.text
    assert "from decimal import Decimal" in bank.text
    last = bank.text.split("last commit: ")[1]
    assert last.startswith(_git(repo, "rev-parse", "--short=12", "HEAD~1"))
    symbol = store.get("symbol:src/pkg/bank.py:11:Ledger")
    assert symbol is not None and symbol.title == "class Ledger — src/pkg/bank.py:11"
    assert "def deposit(self, account, amount):" in symbol.text
    chunk = store.get("chunk:src/pkg/bank.py:1")
    assert chunk is not None
    assert chunk.title == "src/pkg/bank.py lines 1-19 · open_account, Ledger, deposit, withdraw"
    heading = store.get("symbol:docs/guide.md:3:Deposits")
    assert heading is not None and heading.title.startswith("heading Deposits")
    logo, minified = store.get("file:logo.bin"), store.get("file:app.min.js")
    assert logo is not None and "content: not indexed (binary)" in logo.text
    assert minified is not None and "content: not indexed (generated)" in minified.text
    assert store.get("chunk:logo.bin:1") is None and store.get("chunk:app.min.js:1") is None

    head = _git(repo, "rev-parse", "HEAD")
    first = _git(repo, "rev-list", "--max-parents=0", "HEAD")
    commit = store.get(f"commit:{first}")
    assert commit is not None
    assert "message:\ninitial ledger\n\nStarts the accounts domain." in commit.text
    assert "parents: (root commit)" in commit.text and "src/pkg/bank.py" in commit.text
    change = store.get(f"change:{_git(repo, 'rev-parse', 'HEAD~1')}:src/pkg/bank.py")
    assert change is not None and "+def withdraw(account, amount):" in change.text
    assert change.title.endswith("src/pkg/bank.py (+4 -0)")
    tag = store.get("tag:v0.1")
    assert tag is not None and "message: first release" in tag.text
    assert tag.sha == _git(repo, "rev-parse", "HEAD~1")
    main = store.get("branch:main")
    assert store.get("branch:feature/audit") is not None and main is not None
    assert main.sha == head
    author = store.get("author:alice@example.com")
    assert author is not None and author.title == "Alice (3 commits)"
    root, docs = store.get("dir:."), store.get("dir:docs")
    assert root is not None and "files (recursive): 7" in root.text
    assert "README:\n# Bank Ledger" in root.text
    assert "entries:\nREADME.md\napp.min.js\ndocs/\nlogo.bin\nsrc/" in root.text
    assert docs is not None and "README:" not in docs.text
    assert store.get("dir:src/pkg") is not None
    repo_block = store.get("repo")
    assert repo_block is not None and f"HEAD: {head}" in repo_block.text
    assert "commits: 3 on all refs, 3 on HEAD" in repo_block.text
    assert "tags: 1 (latest: v0.1)" in repo_block.text
    assert store.get_meta("head") == head and store.get_meta("mode") == "full"

    memory_dir = _memory_dir(repo)
    assert memory_dir == kiss_home() / "memories" / "ledger"
    lookup = MemoryDir(memory_dir).read(sea.LOOKUP_PAGE)
    assert f"{shlex.quote(sys.executable)} -m {sea.MODULE} search {repo}" in lookup.body
    assert str(store.path) in lookup.body and head[:12] in lookup.body
    assert "04:00 America/Los_Angeles" in lookup.body
    # The lookup page itself is mirrored as a note block.
    assert store.get(f"note:{sea.LOOKUP_PAGE}") is not None
    assert "curated pages mirrored as note blocks: 1" in report


def test_incremental_index_tracks_changes_and_dropped_commits(repo: Path) -> None:
    """A second run indexes only changed files and new commits; a rewrite drops old ones."""
    sea.index_repo(str(repo))
    store = _store(repo)
    before = store.counts()
    # No change: nothing indexed.
    report = index.index_repo(repo, store)
    assert report.mode == "incremental" and report.files_indexed == 0
    assert report.commits_indexed == 0 and report.commits_removed == 0
    assert store.counts() == before

    audit = "def audit(ledger):\n    return ledger\n" + "".join(
        f"# audit rule {i}\n" for i in range(600)
    )
    (repo / "src" / "pkg" / "audit.py").write_text(audit)
    (repo / "src" / "pkg" / "bank.py").write_text(BANK_PY)  # withdraw removed again
    _git(repo, "rm", "-q", "docs/notes.txt")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "audit module; drop notes")
    head = _git(repo, "rev-parse", "HEAD")
    report = index.index_repo(repo, store)
    assert report.mode == "incremental"
    assert report.changed_paths == ["src/pkg/audit.py", "src/pkg/bank.py"]
    assert report.files_removed == 1 and report.commits_indexed == 1
    assert report.new_commits[0].endswith(" audit module; drop notes")
    assert store.get("file:docs/notes.txt") is None
    assert store.get("symbol:src/pkg/audit.py:1:audit") is not None
    assert store.get("symbol:src/pkg/bank.py:18:withdraw") is None  # stale symbol gone
    assert store.get(f"commit:{head}") is not None
    counts = store.counts()
    assert counts["commit"] == 4 and counts["file"] == 7
    docs = store.get("dir:docs")  # docs/ still holds guide.md, but notes.txt is gone
    assert docs is not None and "notes.txt" not in docs.text and "guide.md" in docs.text
    # A patch longer than MAX_PATCH_CHARS continues in numbered blocks, nothing is cut.
    first_part = store.get(f"change:{head}:src/pkg/audit.py")
    second_part = store.get(f"change:{head}#2:src/pkg/audit.py")
    assert first_part is not None and second_part is not None
    assert first_part.title.endswith("part 1/2") and "+# audit rule 599" in second_part.text
    assert _keys(store.search("audit rule 599", kinds=["change"]))[0] == second_part.key
    status = sea.knowledge_status(str(repo))
    assert f"last indexed HEAD: {head}" in status
    assert "(incremental run)" in status and "curated pages: knowledge-lookup" in status

    # Notes written by the agent survive every re-index; an amended-away commit is dropped.
    sea.write_knowledge_page(str(repo), "overview", "# Overview\n\nA ledger.", "Overview", "x")
    _git(repo, "commit", "-q", "--amend", "-m", "audit module (amended)")
    report = index.index_repo(repo, store)
    assert report.mode == "incremental"
    assert report.commits_indexed == 1 and report.commits_removed == 1
    assert store.get(f"commit:{head}") is None
    assert store.get(f"change:{head}:src/pkg/audit.py") is None
    assert store.get(f"commit:{_git(repo, 'rev-parse', 'HEAD')}") is not None
    assert store.get("note:overview") is not None
    assert store.counts()["commit"] == 4

    # A file edited and reverted between two runs has the blob the store knows,
    # but its last commit is now the revert: the ``file`` block is refreshed.
    guide = repo / "docs" / "guide.md"
    original = guide.read_text()
    guide.write_text(original + "\ntemporary line\n")
    _git(repo, "commit", "-q", "-am", "guide: temporary line")
    guide.write_text(original)
    _git(repo, "commit", "-q", "-am", "guide: revert the temporary line")
    revert = _git(repo, "rev-parse", "HEAD")
    report = index.index_repo(repo, store)
    assert report.commits_indexed == 2 and report.changed_paths == ["docs/guide.md"]
    block = store.get("file:docs/guide.md")
    assert block is not None and f"last commit: {revert[:12]}" in block.text
    assert store.counts()["file"] == 7  # re-indexed, not duplicated


def test_every_ref_merge_and_odd_name_is_indexed(tmp_path: Path) -> None:
    """Branch-only commits, merge resolutions, spaces, non-ASCII, tabs and control bytes."""
    shutil.rmtree(kiss_home() / "memories" / "odd", ignore_errors=True)
    repo = _init(tmp_path, "odd")
    (repo / "rules.txt").write_text("base\n")
    (repo / "space name.txt").write_text("SPACEFACT\n")
    (repo / "café.txt").write_text("UNICODEFACT\n")
    (repo / "tab\tname.txt").write_text("TABFACT\n")
    (repo / "src").mkdir()
    (repo / "src" / "😀.py").write_text("ASTRALFACT = 1\n")
    (repo / "src" / "ascii.py").write_text("ASCIIFACT = 1\n")
    (repo / "long.txt").write_text("x" * 7000 + " LONGTAILFACT\n")
    (repo / "big.csv").write_bytes(b"a,b\n" * 300_000 + b"OVERSIZETAILFACT\n")  # 1.2 MB
    (repo / "policies.lock").write_text("pin = 1\n" * 40 + "LOCKTAILFACT\n")
    # Git calls a file binary from its first 8 KB only: this one gets a real patch
    # that contains a NUL byte, which must not be mistaken for a record boundary.
    (repo / "late_nul.txt").write_bytes(b"a" * 9000 + b"\n\x00\nNULFACT\n")
    _git(repo, "add", "-A")
    message = tmp_path / "message.txt"
    message.write_bytes(b"subject with \x1f byte\n\nbody with \x1e and \x1d bytes CTRLFACT\n")
    _git(repo, "commit", "-q", "-F", str(message))
    root = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "side")
    (repo / "rules.txt").write_text("side SIDEFACT\n")
    _git(repo, "commit", "-qam", "side change")
    side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "side-tag")
    _git(repo, "checkout", "-q", "main")
    (repo / "rules.txt").write_text("main\n")
    _git(repo, "commit", "-qam", "main change")
    subprocess.run(["git", "-C", str(repo), "merge", "side"], capture_output=True, check=False)
    (repo / "rules.txt").write_text("MERGERESOLUTIONFACT\n")
    _git(repo, "add", "rules.txt")
    _git(repo, "-c", "core.editor=true", "commit", "-q", "--no-edit")
    merge = _git(repo, "rev-parse", "HEAD")

    sea.index_repo(str(repo))
    store = _store(repo)
    assert store.counts()["commit"] == 4  # the side branch's commit is indexed too
    side_commit = store.get(f"commit:{side}")
    assert side_commit is not None and "side change" in side_commit.text
    assert _keys(store.search("SIDEFACT", kinds=["change"])) == [f"change:{side}:rules.txt"]
    resolution = store.get(f"change:{merge}:rules.txt")
    assert resolution is not None and "+MERGERESOLUTIONFACT" in resolution.text
    ctrl = store.get(f"commit:{root}")
    assert ctrl is not None and "subject with \x1f byte" in ctrl.text and "CTRLFACT" in ctrl.text
    assert ctrl.title.endswith("subject with \x1f byte")
    for path, fact in (
        ("space name.txt", "SPACEFACT"), ("café.txt", "UNICODEFACT"), ("tab\tname.txt", "TABFACT"),
    ):
        change = store.get(f"change:{root}:{path}")
        assert change is not None and f"+{fact}" in change.text, path
        assert store.get(f"file:{path}") is not None
    nul_change = store.get(f"change:{root}#2:late_nul.txt")  # the 9,000 a's fill part 1
    assert nul_change is not None and "+\\0\n+NULFACT" in nul_change.text
    assert _keys(store.search("NULFACT", kinds=["change"])) == [nul_change.key]
    assert store.get(f"change:{root}:policies.lock") is not None  # the next file survived too
    assert _keys(store.search("LONGTAILFACT", kinds=["chunk"])) == ["chunk:long.txt:1:p2"]
    assert store.get("chunk:long.txt:1:p1") is not None and store.get("chunk:long.txt:1") is None
    assert store.search("OVERSIZETAILFACT", kinds=["chunk"])
    assert store.search("LOCKTAILFACT", kinds=["chunk"])
    assert store.get("file:big.csv").text.count("lines: 300001") == 1  # type: ignore[union-attr]
    # path_prefix is a real prefix test, also for names outside the BMP.
    under_src = store.search("ASTRALFACT ASCIIFACT", k=10, kinds=["chunk"], path_prefix="src/")
    assert sorted(hit.block.path for hit in under_src) == ["src/ascii.py", "src/😀.py"]
    assert index.unquote_path('"caf\\303\\251 \\t\\"x\\"\\\\"') == 'café \t"x"\\'
    assert index.unquote_path("plain") == "plain"

    # Checking out another branch changes files without any new commit: their
    # last touch is looked up per file.  A staged, never committed file has none.
    _git(repo, "checkout", "-q", "side")
    (repo / "tab\tname.txt").chmod(0o755)  # mode-only change of a C-quoted name
    _git(repo, "add", "tab\tname.txt")
    _git(repo, "commit", "-q", "-m", "tab file executable")
    mode_only = _git(repo, "rev-parse", "HEAD")
    (repo / "staged.txt").write_text("STAGEDFACT\n")
    _git(repo, "add", "staged.txt")
    report = index.index_repo(repo, store)
    # The mode-only change left the blob alone, but the file's last commit moved.
    assert report.changed_paths == ["rules.txt", "staged.txt", "tab\tname.txt"]
    assert report.commits_indexed == 1
    rules = store.get("file:rules.txt")
    assert rules is not None and f"last commit: {side[:12]}" in rules.text
    tab = store.get("file:tab\tname.txt")
    assert tab is not None and f"last commit: {mode_only[:12]}" in tab.text
    staged = store.get("file:staged.txt")
    assert staged is not None and "last commit" not in staged.text
    change = store.get(f"change:{mode_only}:tab\tname.txt")
    assert change is not None and "new mode 100755" in change.text
    # Above MAX_LAST_TOUCH_LOOKUPS unattributed files, the lookups past the budget wait
    # for the next run (none of these staged files has a commit to find anyway).
    for i in range(index.MAX_LAST_TOUCH_LOOKUPS + 1):
        (repo / f"bulk{i}.txt").write_text(f"bulk {i}\n")
    _git(repo, "add", "-A")
    report = index.index_repo(repo, store)
    assert report.files_indexed == index.MAX_LAST_TOUCH_LOOKUPS + 1
    bulk = store.get("file:bulk0.txt")
    assert bulk is not None and "last commit" not in bulk.text


def test_unshallowed_history_is_indexed_on_the_next_run(tmp_path: Path) -> None:
    """Commits that become reachable later (fetch --unshallow) are indexed then."""
    shutil.rmtree(kiss_home() / "memories" / "shallow", ignore_errors=True)
    origin = _make_repo(tmp_path, "origin")
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(shallow))
    store = _store(shallow)
    assert index.index_repo(shallow, store).commits_indexed == 1
    head = _git(shallow, "rev-parse", "HEAD")
    boundary = store.get(f"commit:{head}")
    assert boundary is not None and "parents: (root commit)" in boundary.text
    assert store.get(f"change:{head}:README.md") is not None  # diff against nothing
    _git(shallow, "fetch", "-q", "--unshallow")
    report = index.index_repo(shallow, store)
    # The two revealed ancestors plus the former boundary commit, whose parents
    # and diff changed, are indexed; its invented changes are gone.
    assert report.commits_indexed == 3 and report.commits_removed == 1
    assert store.counts()["commit"] == 3
    repaired = store.get(f"commit:{head}")
    parent = _git(shallow, "rev-parse", "HEAD~1")
    assert repaired is not None and f"parents: {parent}" in repaired.text
    assert store.get(f"change:{head}:README.md") is None
    assert store.get(f"change:{head}:docs/notes.txt") is not None
    assert index.index_repo(shallow, store).commits_indexed == 0


def test_revealed_history_does_not_age_a_file_s_last_commit(tmp_path: Path) -> None:
    """An unshallowed ancestor touching a file is older than the commit already named.

    Regression: the touched-but-unchanged rule re-indexed the file with the
    newest of the NEW commits, an ancestor here, and the recreation commit
    gave way to the deletion commit.  The per-file lookup decides instead.
    """
    shutil.rmtree(kiss_home() / "memories" / "recreated", ignore_errors=True)
    origin = _init(tmp_path, "origin-recreated")
    (origin / "f.txt").write_text("first\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "add f")
    _git(origin, "rm", "-q", "f.txt")
    (origin / "g.txt").write_text("g\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "delete f, add g")
    (origin / "f.txt").write_text("first\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "recreate f")
    recreate = _git(origin, "rev-parse", "HEAD")
    shallow = tmp_path / "recreated"
    _git(tmp_path, "clone", "-q", "--depth", "2", origin.as_uri(), str(shallow))
    store = _store(shallow)
    index.index_repo(shallow, store)
    block = store.get("file:f.txt")
    assert block is not None and f"last commit: {recreate[:12]}" in block.text
    _git(shallow, "fetch", "-q", "--unshallow")
    report = index.index_repo(shallow, store)
    assert report.commits_indexed == 2  # the revealed root and the re-indexed boundary
    block = store.get("file:f.txt")
    assert block is not None and f"last commit: {recreate[:12]}" in block.text


def test_unshallowing_keeps_the_head_commit_of_a_file_the_boundary_changed(
    tmp_path: Path,
) -> None:
    """Re-indexing the former boundary commit must not attribute a file to it.

    Three commits change ``f.txt``; a depth-2 clone is indexed (f.txt -> HEAD),
    then unshallowed.  The boundary commit is re-indexed and touches f.txt,
    so f.txt is re-indexed too — with the HEAD lookup, not with that older
    commit as its "last commit".
    """
    shutil.rmtree(kiss_home() / "memories" / "thrice", ignore_errors=True)
    origin = _init(tmp_path, "origin-thrice")
    for i in range(3):
        (origin / "f.txt").write_text(f"version {i}\n")
        _git(origin, "add", "-A")
        _git(origin, "commit", "-q", "-m", f"f version {i}")
    head = _git(origin, "rev-parse", "HEAD")
    shallow = tmp_path / "thrice"
    _git(tmp_path, "clone", "-q", "--depth", "2", origin.as_uri(), str(shallow))
    store = _store(shallow)
    index.index_repo(shallow, store)
    _git(shallow, "fetch", "-q", "--unshallow")
    report = index.index_repo(shallow, store)
    assert report.commits_indexed == 2 and report.changed_paths == ["f.txt"]
    block = store.get("file:f.txt")
    assert block is not None and f"last commit: {head[:12]}" in block.text


def test_lookups_beyond_the_budget_are_drained_by_the_next_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Touched files past MAX_LAST_TOUCH_LOOKUPS wait in ``pending_touch`` for the next runs."""
    shutil.rmtree(kiss_home() / "memories" / "budget", ignore_errors=True)
    repo = _init(tmp_path, "budget")
    for i in range(3):
        (repo / f"f{i}.txt").write_text(f"file {i}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "three files")
    store = _store(repo)
    index.index_repo(repo, store)
    for i in range(3):
        (repo / f"f{i}.txt").chmod(0o755)  # mode-only: blobs unchanged, all three touched
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "executable")
    executable = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(index, "MAX_LAST_TOUCH_LOOKUPS", 2)
    report = index.index_repo(repo, store)
    assert report.commits_indexed == 1 and report.changed_paths == ["f0.txt", "f1.txt"]
    assert index.pending_touches(store) == ["f2.txt"]
    stale = store.get("file:f2.txt")
    assert stale is not None and f"last commit: {executable[:12]}" not in stale.text
    report = index.index_repo(repo, store)  # nothing new in the repository
    assert report.commits_indexed == 0 and report.changed_paths == ["f2.txt"]
    assert index.pending_touches(store) == []
    drained = store.get("file:f2.txt")
    assert drained is not None and f"last commit: {executable[:12]}" in drained.text
    assert index.index_repo(repo, store).changed_paths == []


def test_index_errors_for_bad_specs_and_empty_repositories(tmp_path: Path) -> None:
    """Non-repositories, relative paths and commit-less repositories are reported, not raised."""
    assert sea.index_repo(str(tmp_path / "missing")).startswith("Error: ")
    assert "relative path" in sea.index_repo("relative/dir")
    assert "relative path" in sea.index_repo(".")
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "not inside a git work tree" in sea.index_repo(str(plain))
    empty = tmp_path / "empty"
    _git(tmp_path, "init", "-q", str(empty))
    assert "has no commits yet" in sea.index_repo(str(empty))
    assert sea.knowledge_status(str(plain)).startswith("Error: ")
    assert sea.knowledge_search(str(plain), "x").startswith("Error: ")
    assert sea.knowledge_read(str(plain), "repo").startswith("Error: ")
    assert sea.list_knowledge_pages(str(plain)).startswith("Error: ")
    assert sea.read_knowledge_page(str(plain), "overview").startswith("Error: ")
    assert sea.search_knowledge_pages(str(plain), "q").startswith("Error: ")
    assert sea.write_knowledge_page(str(plain), "overview", "x").startswith("Error: ")
    assert sea.delete_knowledge_page(str(plain), "overview").startswith("Error: ")
    assert sea.schedule_daily_update(str(plain)).startswith("Error: ")


def test_url_specs_are_cloned_under_kiss_home_and_kept_in_sync(tmp_path: Path) -> None:
    """A clone URL is cloned once, then fetched and reset to the remote's default branch."""
    shutil.rmtree(kiss_home() / "memories" / "ledger-remote", ignore_errors=True)
    origin = _make_repo(tmp_path, "origin-ledger")
    bare = tmp_path / "ledger-remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(origin), str(bare))
    url = bare.as_uri()
    assert index.is_url(url) and index.is_url("git@github.com:o/r.git")
    assert not index.is_url("/x/y")
    checkout = index.resolve_repo(url)
    assert checkout == (kiss_home() / "knowledge" / "checkouts" / "ledger-remote").resolve()
    assert _git(checkout, "rev-parse", "HEAD") == _git(origin, "rev-parse", "HEAD")
    assert sea.knowledge_status(url).startswith("No memory yet")
    assert "mode: full" in sea.index_repo(url)
    assert index.memory_location(checkout)[0] == "ledger-remote"
    assert sea.canonical_spec(url) == url
    assert sea.main(["status", url]) == 0  # the CLI passes URLs through untouched

    (origin / "docs" / "more.md").write_text("# More\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "more docs")
    _git(origin, "push", "-q", str(bare), "main")
    assert index.resolve_repo(url) == checkout
    assert _git(checkout, "rev-parse", "HEAD") == _git(origin, "rev-parse", "HEAD")
    report = index.index_repo(checkout, _store(checkout))
    assert report.mode == "incremental" and report.changed_paths == ["docs/more.md"]

    # Another remote with the same repository name gets its own checkout.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = _make_repo(other_dir, "other-origin")
    (other / "ONLY_IN_OTHER").write_text("x\n")
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "other")
    other_bare = other_dir / "ledger-remote.git"
    _git(other_dir, "clone", "-q", "--bare", str(other), str(other_bare))
    second = index.resolve_repo(other_bare.as_uri())
    assert second != checkout and second.name.startswith("ledger-remote-")
    assert (second / "ONLY_IN_OTHER").exists() and not (checkout / "ONLY_IN_OTHER").exists()
    assert index.resolve_repo(other_bare.as_uri()) == second  # stable on the next call


def test_search_tokenizes_queries_filters_kinds_and_paths(repo: Path) -> None:
    """Any text is a valid query; all-words matches come first, then any-word matches."""
    sea.index_repo(str(repo))
    store = _store(repo)
    tokens = query_tokens('Ledger.deposit("x") OR foo_bar*')
    assert tokens == ["ledger", "deposit", "x", "foo", "bar*"]  # "or" is a stop word
    assert query_tokens("how does the ledger work") == ["ledger", "work"]
    assert query_tokens("What is it") == ["what", "is", "it"]  # nothing else to search for
    assert store.search("", k=5) == [] and store.search("withdraw", k=0) == []
    hits = store.search("withdraw", k=3, kinds=["symbol"])
    # The definition itself ranks first; the symbols whose context lines
    # reach into ``withdraw`` follow.
    assert hits[0].block.key == "symbol:src/pkg/bank.py:18:withdraw" and len(hits) == 3
    assert "[withdraw]" in hits[0].snippet
    # Every word must match first: only "add withdraw" mentions withdraw and
    # touches bank.py; then any-word matches (the initial commit added
    # bank.py) fill the remaining slots.
    hits = store.search("withdraw bank", k=10, kinds=["commit"])
    assert [h.block.title.split(" ", 1)[1] for h in hits] == ["add withdraw", "initial ledger"]
    under_docs = store.search("ledger", k=20, path_prefix="docs/")
    assert under_docs and all(h.block.path.startswith("docs/") for h in under_docs)
    assert {h.block.kind for h in under_docs} == {"file", "chunk", "symbol", "change"}
    prefixed = store.search("dep*", k=5, kinds=["symbol"])
    assert prefixed[0].block.key.rsplit(":", 1)[1].lower().startswith("dep")
    assert all(re.search(r"\bdep", h.block.text.lower()) for h in prefixed)
    # FTS5 syntax in a query is neutralized: these are plain tokens, not operators.
    assert store.search('"unbalanced (parens -zzq^', k=5) == []
    assert [h.block.kind for h in store.search("NOT indexed", k=5, kinds=["file"])] == ["file"] * 3

    # History ranks below the current state: a change block and a chunk block
    # with the same text tie on BM25, and the chunk wins, even over a change
    # whose BM25 score is better (a title match) but not 1/0.6 = 1.7x better.
    store.upsert([
        Block(
            "change", "change:deadbeef:src/x.py", "x.py in deadbeef", "quorum quorum",
            "src/x.py", "deadbeef",
        ),
        Block("chunk", "chunk:src/x.py:1", "src/x.py lines 1-2", "quorum quorum", "src/x.py", ""),
        Block(
            "change", "change:cafebabe:src/y.py", "quorum quorum in y.py",
            "quorum quorum quorum", "src/y.py", "cafebabe",
        ),
    ], "2026-01-01T00:00:00Z")
    ranked = [h.block.key for h in store.search("quorum", k=3)]
    assert ranked == ["chunk:src/x.py:1", "change:cafebabe:src/y.py", "change:deadbeef:src/x.py"]
    only_changes = [h.block.key for h in store.search("quorum", k=3, kinds=["change"])]
    assert only_changes == ["change:cafebabe:src/y.py", "change:deadbeef:src/x.py"]
    store.delete(key="change:deadbeef:src/x.py")
    store.delete(key="chunk:src/x.py:1")
    store.delete(key="change:cafebabe:src/y.py")

    text = sea.knowledge_search(str(repo), "withdraw", k=2, kinds="symbol,change")
    assert text.startswith("1. symbol:src/pkg/bank.py:18:withdraw\n")
    assert "change:" in sea.knowledge_search(str(repo), "withdraw", k=6, kinds="symbol,change")
    error = sea.knowledge_search(str(repo), "withdraw", kinds="bogus")
    assert error.startswith("Error: unknown block kinds")
    assert sea.knowledge_search(str(repo), "zzzqqq") == "No blocks match 'zzzqqq'."
    assert sea.knowledge_read(str(repo), "tag:v0.1").startswith("tag:v0.1\ntag v0.1 → ")
    assert sea.knowledge_read(str(repo), "tag:v9") == "No block with key 'tag:v9'."


def test_pages_are_written_to_the_domain_memory_and_mirrored(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Curated pages land where every Sorcar run in the repo finds them, plus note blocks."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # offline hashed embeddings
    sea.index_repo(str(repo))
    result = sea.write_knowledge_page(
        str(repo), "domain-glossary",
        "# Glossary\n\n- **Ledger**: holds accounts (src/pkg/bank.py:11).",
        title="Domain glossary", summary="Terms of the ledger domain.",
    )
    assert result.startswith("Wrote ledger/domain-glossary.md")
    page = MemoryDir(_memory_dir(repo)).read("domain-glossary")
    assert page.title == "Domain glossary" and "holds accounts" in page.body
    store = _store(repo)
    note = store.get("note:domain-glossary")
    assert note is not None and "holds accounts" in note.text and "Domain glossary" in note.text
    listing = sea.list_knowledge_pages(str(repo))
    assert "domain-glossary" in listing and sea.LOOKUP_PAGE in listing
    assert "holds accounts" in sea.read_knowledge_page(str(repo), "domain-glossary")
    found = sea.search_knowledge_pages(str(repo), "what is a ledger account", k=2)
    assert "domain-glossary" in found
    # The block store finds the page's words too.
    assert store.search("glossary ledger", kinds=["note"])[0].block.key == "note:domain-glossary"

    assert sea.write_knowledge_page(str(repo), "Bad Name", "x").startswith("Error")
    assert sea.write_knowledge_page(str(repo), "empty", "   ").startswith("Error")
    assert sea.read_knowledge_page(str(repo), "nope").startswith("Error")
    # A page edited by another agent's memory_write is re-mirrored at the next index run.
    MemoryDir(_memory_dir(repo)).write("domain-glossary", "# Glossary\n\nRevised by someone else.")
    sea.index_repo(str(repo))
    revised = store.get("note:domain-glossary")
    assert revised is not None and "Revised by someone else" in revised.text
    assert sea.delete_knowledge_page(str(repo), "domain-glossary").startswith("Deleted")
    assert store.get("note:domain-glossary") is None
    assert sea.delete_knowledge_page(str(repo), "domain-glossary").startswith("Error")


def test_daily_schedule_is_four_in_the_morning_pacific() -> None:
    """The cron scheduler, fed the expression, fires at 04:00 Pacific, summer or winter.

    The scheduler evaluates cron expressions in America/Los_Angeles whatever
    the machine's clock (``cron_agent.SCHEDULE_TZ``), so the expression names
    the Pacific hour directly; converting it to the machine's local time
    (an earlier version) made a UTC daemon run the job at 11:00 Pacific.
    """
    from kiss.agents.sorcar.cron_agent import SCHEDULE_TZ, compute_next_run

    assert sea.daily_update_schedule() == f"0 {sea.DAILY_UPDATE_HOUR_PACIFIC} * * *"
    for stamp in ("2026-07-15T12:00:00+00:00", "2026-01-15T12:00:00+00:00"):
        now = datetime.fromisoformat(stamp).timestamp()
        next_run = compute_next_run(sea.daily_update_schedule(), now)
        assert next_run is not None
        fired = datetime.fromtimestamp(next_run, SCHEDULE_TZ)
        assert (fired.hour, fired.minute) == (sea.DAILY_UPDATE_HOUR_PACIFIC, 0), stamp
        assert 0 < next_run - now <= 24 * 3600


def test_schedule_daily_update_registers_one_cron_job(repo: Path) -> None:
    """The daily job is a run_agent directive to this SEA; a second call does not duplicate it."""
    name = sea.JOB_NAME_PREFIX + "ledger"
    assert not [job for job in load_jobs() if job["name"] == name]
    other = yaml.safe_load(
        cron_job("create", name="unrelated", command="true", schedule="every 1d")
    )
    # Scheduling from a sub-directory stores the checkout's root, never the raw spec.
    created = yaml.safe_load(sea.schedule_daily_update(str(repo / "docs"), max_budget=2.5))
    job_id = created["created"]["id"]
    try:
        jobs = [job for job in load_jobs() if job["name"] == name]
        assert len(jobs) == 1
        job = jobs[0]
        assert job["schedule"] == sea.daily_update_schedule() == "0 4 * * *"
        assert job["model_name"] == sea.DAILY_UPDATE_MODEL == "claude-fable-5-1"
        assert job["max_budget"] == 2.5 + sea.DAILY_UPDATE_RELAY_BUDGET_USD
        assert job["timeout"] == sea.DAILY_UPDATE_TIMEOUT_SECONDS
        assert job["enabled"] is True and job["one_shot"] is False and job["deliver"] == "local"
        assert f"agent      = {str(_SEA_PATH)!r}" in job["prompt"]
        assert f"task       = {f'update {repo}'!r}" in job["prompt"]
        assert f"timeout    = {str(sea.DAILY_UPDATE_TIMEOUT_SECONDS)!r}" in job["prompt"]
        assert "max_budget = '2.5'" in job["prompt"]
        assert f"model_name = {sea.DAILY_UPDATE_MODEL!r}" in job["prompt"]
        again = sea.schedule_daily_update(str(repo), max_budget=2.5)
        assert again.startswith(f"Already scheduled: job {job_id} ({name})")
        assert "Pacific time" in again
        assert len([job for job in load_jobs() if job["name"] == name]) == 1
        # An outdated job for the same repository (another budget here; an
        # older schedule or prompt likewise) is replaced, not duplicated.
        replaced = sea.schedule_daily_update(str(repo))
        assert replaced.startswith(f"Removed outdated job(s) {job_id}.\ncreated:\n")
        jobs = [job for job in load_jobs() if job["name"] == name]
        assert len(jobs) == 1 and jobs[0]["id"] != job_id
        job_id = jobs[0]["id"]
        # A job in the format of an older version of this SEA (host-local
        # schedule, no model or budget in the directive) is recognised and
        # removed too, while the up-to-date job stays.
        old = yaml.safe_load(cron_job(
            "create", name=name, schedule="0 11 * * *", max_budget="5", timeout="300",
            prompt=(
                "Call the run_agent tool IMMEDIATELY, as your very first action, with "
                f"these arguments and no others:\n  agent   = {str(_SEA_PATH)!r}\n"
                f"  task    = {f'update {repo}'!r}\n  timeout = '14400'\nDo not explore."
            ),
        ))
        assert [job for job in load_jobs() if job["id"] == old["created"]["id"]]
        again = sea.schedule_daily_update(str(repo))
        assert again.startswith(
            f"Removed outdated job(s) {old['created']['id']}.\nAlready scheduled: job {job_id}"
        )
        assert [job["id"] for job in load_jobs() if job["name"] == name] == [job_id]
        assert jobs[0]["max_budget"] == (
            sea.DAILY_UPDATE_BUDGET_USD + sea.DAILY_UPDATE_RELAY_BUDGET_USD
        )
        assert f"max_budget = {str(sea.DAILY_UPDATE_BUDGET_USD)!r}" in jobs[0]["prompt"]
        # The unrelated job is untouched.
        assert [job for job in load_jobs() if job["name"] == "unrelated"]
        # Same prompt, schedule, model and budget but a short timeout: outdated too.
        current = [job for job in load_jobs() if job["id"] == job_id][0]
        cron_job("remove", job_id=job_id)
        short = yaml.safe_load(cron_job(
            "create", name=name, schedule=current["schedule"], prompt=current["prompt"],
            model_name=current["model_name"], max_budget=str(current["max_budget"]),
            timeout="300",
        ))
        assert "created" in short, short
        renewed = sea.schedule_daily_update(str(repo))
        assert renewed.startswith(f"Removed outdated job(s) {short['created']['id']}.\ncreated:")
        jobs = [job for job in load_jobs() if job["name"] == name]
        assert len(jobs) == 1 and jobs[0]["timeout"] == sea.DAILY_UPDATE_TIMEOUT_SECONDS
        job_id = jobs[0]["id"]
    finally:
        cron_job("remove", job_id=job_id)
        cron_job("remove", job_id=other["created"]["id"])


def test_cli_drives_the_same_tools(repo: Path, tmp_path: Path) -> None:
    """Fan-out sub-agents and any shell user reach the tools through ``python -m``."""
    env = {**os.environ, "KISS_HOME": str(kiss_home())}

    def cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", sea.MODULE, *args], capture_output=True, text=True, env=env,
            cwd=cwd,
        )

    out = cli("index", "ledger", cwd=tmp_path)  # relative to the shell's directory
    assert out.returncode == 0 and "mode: full" in out.stdout, out.stderr
    out = cli("search", str(repo), "withdraw", "--kinds", "symbol", "--k", "1")
    assert out.returncode == 0 and out.stdout.startswith("1. symbol:src/pkg/bank.py:18:withdraw")
    out = cli("search", str(repo), "withdraw", "--path", "docs/")
    assert out.returncode == 0 and out.stdout.strip() == "No blocks match 'withdraw'."
    out = cli("read", str(repo), "author:alice@example.com")
    assert "commits: 3" in out.stdout
    body = tmp_path / "page.md"
    body.write_text("# Module src/pkg\n\n`bank.py` holds the Ledger (src/pkg/bank.py:11).\n")
    out = cli(
        "write-page", str(repo), "module-src-pkg", "--title", "Module src/pkg",
        "--summary", "The ledger package.", "--file", str(body),
    )
    assert out.returncode == 0 and out.stdout.startswith("Wrote ledger/module-src-pkg.md")
    assert "module-src-pkg" in cli("pages", str(repo)).stdout
    assert "holds the Ledger" in cli("read-page", str(repo), "module-src-pkg").stdout
    assert "last indexed HEAD" in cli("status", str(repo)).stdout
    assert cli("delete-page", str(repo), "module-src-pkg").returncode == 0
    out = cli("read", str(tmp_path), "repo")
    assert out.returncode == 1 and out.stdout.startswith("Error: ")


def test_lookup_page_commands_run_for_paths_with_spaces(tmp_path: Path) -> None:
    """The quoted shell commands on the lookup page work as written."""
    shutil.rmtree(kiss_home() / "memories" / "space-repo", ignore_errors=True)
    repo = _make_repo(tmp_path, "space repo")
    sea.index_repo(str(repo))
    page = MemoryDir(_memory_dir(repo)).read(sea.LOOKUP_PAGE).body
    command = next(line.strip() for line in page.splitlines() if " status " in line)
    out = subprocess.run(
        shlex.split(command), capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "KISS_HOME": str(kiss_home())},
    )
    assert out.returncode == 0 and f"repository: {repo}" in out.stdout, out.stderr


def test_cli_in_process_covers_every_subcommand(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` maps each sub-command to its tool and exits 1 on an error message."""
    assert sea.main(["index", str(repo)]) == 0
    assert sea.main(["status", str(repo)]) == 0
    assert sea.main(["search", str(repo), "ledger", "--k", "2", "--kinds", "file"]) == 0
    assert sea.main(["read", str(repo), "repo"]) == 0
    body = tmp_path / "faq.md"
    body.write_text("# FAQ\n\nQ: what is a ledger? A: see src/pkg/bank.py:11.\n")
    assert sea.main(["write-page", str(repo), "faq", "--file", str(body)]) == 0
    assert sea.main(["pages", str(repo)]) == 0
    assert sea.main(["read-page", str(repo), "faq"]) == 0
    assert sea.main(["delete-page", str(repo), "faq"]) == 0
    assert sea.main(["read-page", str(repo), "faq"]) == 1
    assert sea.main(["schedule", str(repo)]) == 0
    out = capsys.readouterr().out
    scheduled = [job for job in load_jobs() if job["name"] == sea.JOB_NAME_PREFIX + "ledger"]
    assert len(scheduled) == 1
    cron_job("remove", job_id=scheduled[0]["id"])
    assert "mode: full" in out and "1. file:" in out and "Wrote ledger/faq.md" in out
    with pytest.raises(SystemExit):
        sea.main([])


def test_segments_are_linear_and_cap_giant_diffs() -> None:
    """Segments split on NUL across chunk boundaries; an oversized segment keeps its head."""
    small = b"\0A\0B" + b"x" * (3 << 20) + b"\0C"
    assert list(index._segments(io.BytesIO(small))) == [b"", b"A", b"B" + b"x" * (3 << 20), b"C"]
    assert list(index._segments(io.BytesIO(b""))) == []
    huge = b"\0HEAD\n" + b"p" * (index.MAX_RECORD_BYTES + (3 << 20)) + b"\0Z"
    records = list(index._segments(io.BytesIO(huge)))
    assert records[0] == b"" and records[-1] == b"Z"
    kept = records[1]
    assert kept.startswith(b"HEAD\n") and len(kept) < len(huge)
    assert index.MAX_RECORD_BYTES <= len(kept) <= index.MAX_RECORD_BYTES + (2 << 20)


def test_git_failures_and_odd_inputs_are_reported(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every git failure surfaces as ``KnowledgeError``; odd stream inputs are handled."""
    with pytest.raises(index.KnowledgeError, match="exit 128"):
        index.run_git(repo, "rev-parse", "--verify", "no-such-ref")
    with pytest.raises(index.KnowledgeError, match="is not a git work tree"):
        index.memory_location(tmp_path)
    with pytest.raises(index.KnowledgeError, match="git log failed"):
        list(index.iter_commits(repo, ["0" * 40]))
    assert list(index.iter_commits(repo, [])) == []
    with pytest.raises(index.KnowledgeError, match="unexpected git log record"):
        index._parse_header("garbage")
    assert list(index.read_blobs(repo, ["0" * 40])) == []  # a missing blob is skipped
    # An empty commit has neither numstat nor patch: one commit block, no change blocks.
    header = "\x1f".join([
        "a" * 40, "", "Bob", "bob@x", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
        "1767225600", "", "empty\n",
    ])
    empty = index._parse_header(header)
    index._parse_diff(empty, "\n")
    assert empty.numstat == [] and empty.patches == {} and empty.subject == "empty"
    blocks = index.commit_blocks(empty)
    assert [block.kind for block in blocks] == ["commit"] and "files changed" not in blocks[0].text
    # Directories with more entries than shown are cut with a count.
    many = index.dir_blocks({f"d/f{i}.txt": "x" for i in range(index.MAX_DIR_ENTRIES + 1)}, {})
    text = next(block.text for block in many if block.key == "dir:d")
    assert "… 1 more entries" in text
    # Without a git executable every command fails loudly.
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(index.KnowledgeError, match="failed in"):
        index.run_git(repo, "status")


def test_submodule_gitlinks_are_not_files(tmp_path: Path) -> None:
    """A submodule entry (mode 160000) is skipped by the file scan."""
    outer = _make_repo(tmp_path, "outer")
    inner = _make_repo(tmp_path, "inner")
    _git(
        outer, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(inner),
        "vendor/inner",
    )
    _git(outer, "commit", "-q", "-m", "add submodule")
    files = index.ls_files(outer)
    assert ".gitmodules" in files and not any(path.startswith("vendor/inner") for path in files)


def test_symbols_are_found_across_language_families() -> None:
    """The definition-line patterns cover the common languages."""
    cases = {
        "Python": (
            "async def fetch(x):\nclass Node:\n", [(1, "def", "fetch"), (2, "class", "Node")],
        ),
        "TypeScript": (
            "export default class App {}\nexport const run = async (x: number) => x;\n"
            "function helper() {}\n",
            [(1, "class", "App"), (2, "function", "run"), (3, "function", "helper")],
        ),
        "Go": (
            "func (s *Server) Start() error {\ntype Config struct {\n",
            [(1, "func", "Start"), (2, "struct", "Config")],
        ),
        "Rust": (
            "pub async fn serve() {}\nimpl Server {\n", [(1, "fn", "serve"), (2, "impl", "Server")],
        ),
        "Java": (
            "public final class Main {\n    public static void main(String[] a) {\n",
            [(1, "class", "Main"), (2, "method", "main")],
        ),
        "Kotlin": (
            "data class User(val id: Int)\n    suspend fun load() {}\n",
            [(1, "class", "User"), (2, "fun", "load")],
        ),
        "C": (  # keywords are not names
            "static int parse_args(int argc, char **argv)\n{\nelse if (x) {\nstruct node {\n",
            [(1, "function", "parse_args"), (4, "struct", "node")],
        ),
        "Ruby": (
            "module Bank\n  def deposit!(x)\n", [(1, "module", "Bank"), (2, "def", "deposit!")],
        ),
        "PHP": (
            "final class Account {\n    public function balance() {}\n",
            [(1, "class", "Account"), (2, "function", "balance")],
        ),
        "Shell": (
            "function build() {\nrun_tests () {\n",
            [(1, "function", "build"), (2, "function", "run_tests")],
        ),
        "Lua": (
            "local function tick()\nfunction M.start()\n",
            [(1, "function", "tick"), (2, "function", "M.start")],
        ),
        "Markdown": (
            "# Title\ntext\n## Part ##\n", [(1, "heading", "Title"), (3, "heading", "Part")],
        ),
        "JSON": ('{"def": 1}\n', []),
    }
    for language, (source, expected) in cases.items():
        assert index.find_symbols(source.splitlines(), language) == expected, language
    assert index.language_of("a/Dockerfile") == "Dockerfile"
    assert index.language_of("x.PY") == "Python"
    assert index.language_of("weird.zzz") == ""


def test_chunks_hold_the_whole_file_and_oversized_files_keep_metadata() -> None:
    """Chunks are bounded by lines and characters; over-long lines become pieces."""
    lines = ["short"] * 100 + ["y" * 13_000] + ["tail"]
    spans = index.chunk_spans(lines)
    assert spans == [
        (0, 80, 0), (80, 100, 0), (100, 101, 1), (100, 101, 2), (100, 101, 3), (101, 102, 0),
    ]
    dense = ["z" * 500] * 30  # 30 lines but 15,000 characters: split by characters
    assert index.chunk_spans(dense) == [(0, 11, 0), (11, 22, 0), (22, 30, 0)]
    assert index.chunk_spans([]) == []
    blocks = index.file_blocks("big.txt", "sha", "\n".join(lines).encode(), ("abc", "2026-01-01"))
    keys = [block.key for block in blocks]
    assert keys == [  # the file block comes last: it marks the file as fully indexed
        "chunk:big.txt:1", "chunk:big.txt:81", "chunk:big.txt:101:p1", "chunk:big.txt:101:p2",
        "chunk:big.txt:101:p3", "chunk:big.txt:102", "file:big.txt",
    ]
    assert "".join(b.text.split("\n", 1)[1] for b in blocks[2:5]) == "y" * 13_000
    assert "last commit: abc (2026-01-01)" in blocks[-1].text
    oversized = index.file_blocks(
        "huge.csv", "sha", b"a,b\n" * (index.MAX_CONTENT_BYTES // 4 + 1), None,
    )
    assert len(oversized) == 1 and "not indexed (too large: " in oversized[0].text


def test_agent_indexes_writes_a_page_and_finishes(repo: Path, tmp_path: Path) -> None:
    """With the SEA's configuration the model gets the knowledge tools and real tool results."""
    page = "# Overview\n\nA ledger of bank accounts (src/pkg/bank.py:11)."
    script = [
        tool_call_body("index_repo", {"repo": str(repo)}, prompt_tokens=500),
        tool_call_body(
            "write_knowledge_page",
            {"repo": str(repo), "name": "overview", "content": page, "title": "Overview",
             "summary": "What the ledger is."},
            prompt_tokens=600,
        ),
        tool_call_body(
            "knowledge_search", {"repo": str(repo), "query": "ledger", "kinds": "note"},
            prompt_tokens=700,
        ),
        finish_body("<h3>Memory built</h3>", prompt_tokens=800),
    ]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("git-knowledge-sea-test")
        result = agent.run(
            prompt_template=str(repo),
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=6,
            max_budget=5.0,
            model_config={"base_url": url, "api_key": "local"},
            tools=sea.tools(),
            tool_profile=sea.tool_profile(),
            base_system_prompt=sea.system_prompt(),
            web_tools=sea.use_web_tools(),
            use_memory=sea.use_memory(),
            is_parallel=sea.is_parallel(),
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True and parsed["summary"] == "<h3>Memory built</h3>"
    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 4
    names = {t["function"]["name"] for t in agentic[0]["tools"]}
    expected = {"index_repo", "write_knowledge_page", "knowledge_search", "Bash", "Read", "finish"}
    assert expected <= names
    assert "memory_search" not in names and "go_to_url" not in names
    system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
    assert str(system["content"]).startswith(sea.SYSTEM_PROMPT[:200])
    results = [str(m["content"]) for m in agentic[3]["messages"] if m["role"] == "tool"]
    assert "mode: full" in results[0] and f"repository: {repo}" in results[0]
    assert results[1].startswith("Wrote ledger/overview.md")
    assert results[2].startswith("1. note:overview")
    assert MemoryDir(_memory_dir(repo)).read("overview").body.strip() == page


def test_report_lists_are_capped() -> None:
    """Long change lists are cut with a count of the rest."""
    report = index.IndexReport(
        repo="r", slug="s", memory_dir="m", store_path="p", mode="full", head="h" * 40,
        previous_head="", seconds=1.0, counts={"file": 3}, files_total=3, files_indexed=3,
        files_removed=0, changed_paths=[f"f{i}" for i in range(5)], commits_indexed=0,
        commits_removed=0, new_commits=[], languages={"Python": 3}, top_dirs=[(".", 3)],
        authors=["A (1 commits)"], tags=[],
    )
    text = index.format_report(report, list_limit=2)
    assert "changed files:\n  f0\n  f1\n  … 3 more" in text and "new commits" not in text
    assert "tags: (none)" in text


def test_store_delete_requires_a_filter(tmp_path: Path) -> None:
    """An unfiltered delete is refused; filtered deletes report their row counts."""
    store = KnowledgeStore(tmp_path / "k.sqlite3")
    stamp = index.now_iso()
    blocks = [Block("note", "note:a", "a", "alpha"), Block("note", "note:b", "b", "beta")]
    assert store.upsert(blocks, stamp) == 2
    with pytest.raises(ValueError):
        store.delete()
    assert store.delete(key="note:a") == 1 and store.delete(kind="note") == 1
    assert store.counts() == {} and store.keys("note") == []
    assert list(store.iter_blocks("note")) == []
    store.set_meta(head="x")
    assert store.get_meta("head") == "x" and store.get_meta("nope", "d") == "d"
    assert store.file_shas() == {} and store.delete_shas(["x"]) == 0


def test_store_batches_large_writes_and_dedupes_query_tokens(tmp_path: Path) -> None:
    """Writes above the batch size land in several transactions; repeated words collapse."""
    store = KnowledgeStore(tmp_path / "big.sqlite3")
    n = 2 * 2000 + 1  # WRITE_BATCH is 2000
    blocks = [Block("note", f"note:n{i}", f"n{i}", f"word{i} common") for i in range(n)]
    assert store.upsert(blocks, index.now_iso()) == n
    assert store.counts() == {"note": n} and len(list(store.iter_blocks("note"))) == n
    assert query_tokens("common common Common") == ["common"]
    assert len(store.search("common", k=100)) == 100


def test_sha256_repo_odd_config_and_hostile_names(tmp_path: Path) -> None:
    """SHA-256 object names, diff.noprefix/log.showRoot config, mode-only renames of quoted
    names, a file literally named ``f:2``, non-UTF-8 names, control bytes in a tag message
    at a ref tip, a NEL byte inside a line, and clock skew between parent and child."""
    shutil.rmtree(kiss_home() / "memories" / "s256", ignore_errors=True)
    repo = tmp_path / "s256"
    repo.mkdir()
    _git(tmp_path, "init", "-q", "-b", "main", "--object-format=sha256", str(repo))
    _git(repo, "config", "user.email", "alice@example.com")
    _git(repo, "config", "user.name", "Alice")
    _git(repo, "config", "diff.noprefix", "true")  # would drop a/ b/ from patch headers
    _git(repo, "config", "log.showRoot", "false")  # would hide the root commit's diff
    (repo / "f:2").write_text("COLONFACT\n")
    (repo / "f").write_text("".join(f"f line {i}\n" for i in range(900)))  # a 2-part patch
    (repo / "café.sh").write_text("echo QUOTEDFACT\n")
    (repo / "nel.txt").write_text("one\u0085still one\ntwo\n")
    (repo / "skew.txt").write_text("v1\n")
    # APFS (macOS) refuses file names that are not valid UTF-8 with EILSEQ; the
    # non-UTF-8 name is exercised only where the filesystem can hold one.
    try:
        os.link(repo / "f:2", repo / b"raw\xff.txt".decode("utf-8", "surrogateescape"))
        has_raw_name = True
    except OSError as exc:
        if exc.errno != errno.EILSEQ:
            raise
        has_raw_name = False
    _git(repo, "add", "-A")
    future = "2030-01-01T00:00:00+00:00"  # both dates: the committer date drives traversal
    skewed = {**os.environ, "GIT_AUTHOR_DATE": future, "GIT_COMMITTER_DATE": future}
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "root in the future"],
        check=True, env=skewed,
    )
    root = _git(repo, "rev-parse", "HEAD")
    assert len(root) == 64
    tag_message = tmp_path / "tag.txt"
    tag_message.write_bytes(b"tag subject \x1e\x1f\x1d\n\nTAGCTRLFACT body\n")
    _git(repo, "tag", "-a", "v1", "-F", str(tag_message))
    (repo / "café.sh").chmod(0o755)  # mode-only change: no ---/+++ lines in the patch
    (repo / "skew.txt").write_text("v2 SKEWFACT\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "child in the past")
    child = _git(repo, "rev-parse", "HEAD")

    sea.index_repo(str(repo))
    store = _store(repo)
    assert store.counts()["commit"] == 2
    # The root commit's diff is indexed although log.showRoot is off; the
    # a/ and b/ prefixes are forced although diff.noprefix is on.
    root_change = store.get(f"change:{root}:f:2")
    assert root_change is not None and "+COLONFACT" in root_change.text
    assert store.get(f"change:{root}:nel.txt") is not None
    # A file named f:2 cannot collide with part 2 of file f's patch: both exist.
    part_two = store.get(f"change:{root}#2:f")
    assert part_two is not None and "+f line 899" in part_two.text
    assert root_change.title.endswith("f:2 (+1 -0)") and "f line" not in root_change.text
    mode_only = store.get(f"change:{child}:café.sh")
    assert mode_only is not None and "old mode 100644" in mode_only.text
    # A NEL byte is not a line break.
    nel = store.get("file:nel.txt")
    assert nel is not None and "lines: 2" in nel.text
    raw = store.get("file:raw\\xff.txt")
    if has_raw_name:
        assert raw is not None and "content: not indexed" not in raw.text
    else:
        assert raw is None
    tag = store.get("tag:v1")
    assert tag is not None and tag.sha == root and "TAGCTRLFACT" in tag.text
    assert store.get("branch:main") is not None
    # The child commit is the file's last touch even though its date is older.
    skew = store.get("file:skew.txt")
    assert skew is not None and f"last commit: {child[:12]}" in skew.text
    assert _keys(store.search("SKEWFACT", kinds=["change"])) == [f"change:{child}:skew.txt"]


def test_store_is_replaced_when_another_repository_takes_the_slug(tmp_path: Path) -> None:
    """Two unrelated checkouts named alike share a slug; the newcomer replaces the blocks
    but keeps the agent's notes, and the cron job is deduplicated by its exact prompt."""
    shutil.rmtree(kiss_home() / "memories" / "twin", ignore_errors=True)
    first = _make_repo(tmp_path / "a", "twin")
    sea.index_repo(str(first))
    sea.write_knowledge_page(str(first), "overview", "# Overview\n\nKeep me.", "Overview", "x")
    store = _store(first)
    assert store.counts()["commit"] == 3 and store.get("note:overview") is not None
    second = _init(tmp_path / "b", "twin")
    (second / "other.txt").write_text("OTHERFACT\n")
    _git(second, "add", "-A")
    _git(second, "commit", "-q", "-m", "unrelated history")
    report = index.index_repo(second, store)
    assert report.mode == "full" and report.commits_indexed == 1 and report.commits_removed == 0
    counts = store.counts()
    assert counts["commit"] == 1 and counts["file"] == 1
    assert counts["note"] == 2  # overview and the knowledge-lookup page survive
    assert store.get("note:overview") is not None
    assert store.get("file:README.md") is None and store.get("file:other.txt") is not None
    # Re-running on the same repository is incremental again.
    assert index.index_repo(second, store).mode == "incremental"


def test_blocks_are_written_children_first() -> None:
    """The file and commit blocks come last so an interrupted run cannot leave a file or
    commit that looks indexed while its chunks or changes are missing."""
    blocks = index.file_blocks("a.py", "blob", BANK_PY.encode(), ("s" * 40, "2026-01-01"))
    assert [b.kind for b in blocks[:-1]] and blocks[-1].kind == "file"
    assert all(b.kind != "file" for b in blocks[:-1])
    commit = index.Commit(
        "c" * 40, "", "A", "a@x", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", 0,
        "", "subject",
    )
    commit.numstat.append(("1", "0", "a.py"))
    commit.patches["a.py"] = "diff --git a/a.py b/a.py\n+++ b/a.py\n+x\n"
    blocks = index.commit_blocks(commit)
    assert [b.kind for b in blocks] == ["change", "commit"]


def test_first_build_attributes_files_to_head_history_only(tmp_path: Path) -> None:
    """A newer commit on an unmerged branch must not become the last commit of a file
    whose checked-out content comes from HEAD; names that look like pathspecs stay literal."""
    shutil.rmtree(kiss_home() / "memories" / "diverge", ignore_errors=True)
    repo = _init(tmp_path, "diverge")
    (repo / "rules.txt").write_text("MAINFACT\n")
    (repo / "x.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main rules")
    main_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "side")
    (repo / "rules.txt").write_text("SIDEFACT\n")
    _git(repo, "commit", "-qam", "side rules")
    _git(repo, "checkout", "-q", "main")
    store = _store(repo)
    assert index.index_repo(repo, store).commits_indexed == 2
    rules = store.get("file:rules.txt")
    assert rules is not None and "MAINFACT" in rules.text
    assert f"last commit: {main_commit[:12]}" in rules.text
    # Staged names with glob or magic characters are looked up literally.
    (repo / "[x].txt").write_text("bracket\n")
    (repo / ":(bogus)name.txt").write_text("magic\n")
    _git(repo, "--literal-pathspecs", "add", "[x].txt", ":(bogus)name.txt")
    report = index.index_repo(repo, store)
    assert report.changed_paths == [":(bogus)name.txt", "[x].txt"]
    for path in report.changed_paths:
        block = store.get(f"file:{path}")
        assert block is not None and "last commit" not in block.text, path


def test_interrupted_run_is_repaired_and_shallowing_keeps_the_store(tmp_path: Path) -> None:
    """A file or commit whose marker block is missing (the previous run died before writing
    it) is indexed again; a shallow clone of an indexed history is the same repository."""
    shutil.rmtree(kiss_home() / "memories" / "again", ignore_errors=True)
    origin = _make_repo(tmp_path, "again")
    store = _store(origin)
    index.index_repo(origin, store)
    head = _git(origin, "rev-parse", "HEAD")
    symbols = store.counts()["symbol"]
    # Simulate the interruption: the children are there, the markers are not.
    assert store.delete(key="file:src/pkg/bank.py") == 1
    assert store.delete(key=f"commit:{head}") == 1
    assert store.get("symbol:src/pkg/bank.py:11:Ledger") is not None
    report = index.index_repo(origin, store)
    # bank.py for its missing marker; notes.txt because the re-indexed commit touched it.
    assert report.mode == "incremental"
    assert report.changed_paths == ["docs/notes.txt", "src/pkg/bank.py"]
    assert report.commits_indexed == 1 and report.new_commits[0].endswith(" notes on interest")
    assert store.get("file:src/pkg/bank.py") is not None and store.get(f"commit:{head}") is not None
    assert store.counts()["symbol"] == symbols  # re-indexed symbols replaced, not doubled

    shallow = tmp_path / "elsewhere" / "again"
    _git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(shallow))
    report = index.index_repo(shallow, store)  # same slug, same store
    assert report.mode == "incremental" and report.files_removed == 0
    assert report.commits_removed == 2 and report.commits_indexed == 0  # HEAD is kept as is
    assert report.files_indexed == 6  # the files the dropped commits touched are re-attributed
    assert store.counts()["file"] == 7 and store.counts()["commit"] == 1


def test_giant_files_and_diffs_are_flagged_not_dropped(tmp_path: Path) -> None:
    """Content above the 64 MB limits is left out, and the blocks say so explicitly."""
    shutil.rmtree(kiss_home() / "memories" / "giant", ignore_errors=True)
    repo = _init(tmp_path, "giant")
    line = b"0123456789abcdef" * 4 + b"\n"  # 65 bytes
    (repo / "dump.txt").write_bytes(line * (index.MAX_RECORD_BYTES // len(line) + 1000))
    (repo / "after.txt").write_text("AFTERFACT\n")  # sorts after dump.txt in the diff
    (repo / "before.txt").write_text("BEFOREFACT\n")
    (repo / "zz_nul.txt").write_bytes(b"a" * 9000 + b"\n\x00\nZFACT\n")  # NUL past the cap
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "dump")
    sha = _git(repo, "rev-parse", "HEAD")
    store = _store(repo)
    index.index_repo(repo, store)
    dump = store.get("file:dump.txt")
    assert dump is not None and "content: not indexed (too large:" in dump.text
    assert f"limit {index.MAX_CONTENT_BYTES}" in dump.text
    commit = store.get(f"commit:{sha}")
    assert commit is not None and f"patch: truncated at {index.MAX_RECORD_BYTES}" in commit.text
    assert "dump.txt" in commit.text and "before.txt" in commit.text  # numstat is complete
    before = store.get(f"change:{sha}:before.txt")
    assert before is not None and "+BEFOREFACT" in before.text
    for path in ("dump.txt", "zz_nul.txt"):
        dropped = store.get(f"change:{sha}:{path}")
        assert dropped is not None and "ZFACT" not in dropped.text  # recorded, patch cut
    assert store.get(f"change:{sha}#2:dump.txt") is not None  # the kept 64 MB are searchable
