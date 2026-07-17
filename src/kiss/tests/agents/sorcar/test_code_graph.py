# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``code_graph`` tool (graphify #1 + #3 + #5).

Covers:

* tree-sitter graph builds on a real multi-language mini project
  (nodes, edges, EXTRACTED/INFERRED confidence tagging),
* SHA256-cache incremental rebuilds (only changed files re-extracted,
  deleted files pruned),
* ``query`` / ``path`` / ``explain`` output formats,
* query-before-grep interception (``grep_hint``) and its integration
  into ``UsefulTools.Bash``,
* git post-commit hook install / append-preserving / uninstall and a
  real commit firing an incremental update,
* the ``code_graph`` agent tool built by ``make_code_graph_tool``,
* graceful degradation (unsupported files, empty dirs, missing graph).

The primary feature paths parse real files with real tree-sitter grammars
and run real Git commands in temporary repositories.  Narrow fault-injection
cases additionally exercise defensive operating-system and malformed-AST paths.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kiss.agents.sorcar.code_graph as cg
from kiss.agents.sorcar.code_graph import (
    CodeGraph,
    build_graph,
    graph_dir,
    grep_hint,
    install_post_commit_hook,
    load_graph,
    make_code_graph_tool,
    uninstall_post_commit_hook,
)
from kiss.agents.sorcar.useful_tools import UsefulTools

PY_MAIN = '''\
import os
from helpers import util_fn

class Application:
    """Top level app."""

    def run(self):
        util_fn()
        self.stop()

    def stop(self):
        pass


def entry_point():
    app = Application()
    app.run()
'''

PY_HELPERS = '''\
def util_fn():
    return shared_calc()


def shared_calc():
    return 42
'''

JS_APP = """\
import { render } from './view.js';

class Widget {
  draw() {
    render();
  }
}

function startWidget() {
  const w = new Widget();
  w.draw();
}
"""


def _make_project(root: Path) -> None:
    """Write the fixture mini-project (python + javascript + noise)."""
    (root / "main.py").write_text(PY_MAIN)
    (root / "helpers.py").write_text(PY_HELPERS)
    (root / "app.js").write_text(JS_APP)
    (root / "notes.txt").write_text("not code\n")
    (root / "data.xyzunknown").write_text("???\n")


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
        check=True,
    )
    return result.stdout


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    _make_project(tmp_path)
    return tmp_path


@pytest.fixture()
def built(project: Path) -> CodeGraph:
    return build_graph(str(project))


# ---------------------------------------------------------------------------
# Build: nodes, edges, confidence
# ---------------------------------------------------------------------------


class TestBuild:
    def test_nodes_extracted(self, built: CodeGraph) -> None:
        labels = {n["label"] for n in built.nodes.values()}
        assert "Application" in labels
        assert "run" in labels
        assert "util_fn" in labels
        assert "shared_calc" in labels
        assert "Widget" in labels
        assert "startWidget" in labels
        # file nodes exist for every parsed source file
        assert "main.py" in labels
        assert "helpers.py" in labels
        assert "app.js" in labels

    def test_node_fields(self, built: CodeGraph) -> None:
        app = built.find_node("Application")
        assert app is not None
        assert app["kind"] == "class"
        assert app["file"] == "main.py"
        assert app["line"] == 4

    def test_contains_and_defines_edges(self, built: CodeGraph) -> None:
        rels = {(e["source"], e["relation"], e["target"]) for e in built.edges}
        app = built.find_node("Application")
        run = built.find_node("run")
        main = built.find_node("main.py")
        assert app is not None and run is not None and main is not None
        assert (app["id"], "contains", run["id"]) in rels
        assert (main["id"], "defines", app["id"]) in rels

    def test_import_edges_extracted_confidence(self, built: CodeGraph) -> None:
        main = built.find_node("main.py")
        assert main is not None
        imports = [
            e
            for e in built.edges
            if e["source"] == main["id"] and e["relation"] == "imports"
        ]
        targets = {built.nodes[e["target"]]["label"] for e in imports}
        assert "os" in targets
        assert "helpers" in targets
        assert all(e["confidence"] == "EXTRACTED" for e in imports)

    def test_call_edge_same_file_extracted(self, built: CodeGraph) -> None:
        run = built.find_node("run")
        stop = built.find_node("stop")
        assert run is not None and stop is not None
        edge = built.find_edge(run["id"], stop["id"], "calls")
        assert edge is not None
        assert edge["confidence"] == "EXTRACTED"

    def test_call_edge_cross_file_inferred(self, built: CodeGraph) -> None:
        # util_fn is defined in helpers.py; the call from
        # Application.run in main.py is resolved in the second
        # (call-graph) pass and therefore tagged INFERRED.
        run = built.find_node("run")
        util = built.find_node("util_fn")
        assert run is not None and util is not None
        edge = built.find_edge(run["id"], util["id"], "calls")
        assert edge is not None
        assert edge["confidence"] == "INFERRED"

    def test_js_call_edge(self, built: CodeGraph) -> None:
        draw = built.find_node("draw")
        start = built.find_node("startWidget")
        assert draw is not None and start is not None
        edge = built.find_edge(start["id"], draw["id"], "calls")
        assert edge is not None

    def test_non_code_files_skipped(self, built: CodeGraph) -> None:
        files = {n["file"] for n in built.nodes.values()}
        assert "notes.txt" not in files
        assert "data.xyzunknown" not in files

    def test_graph_persisted(self, project: Path, built: CodeGraph) -> None:
        gpath = graph_dir(str(project)) / "graph.json"
        cpath = graph_dir(str(project)) / "cache.json"
        assert gpath.is_file()
        assert cpath.is_file()
        data = json.loads(gpath.read_text())
        assert data["nodes"]
        assert data["edges"]
        loaded = load_graph(str(project))
        assert loaded is not None
        assert set(loaded.nodes) == set(built.nodes)

    def test_load_graph_missing(self, tmp_path: Path) -> None:
        assert load_graph(str(tmp_path)) is None

    def test_load_graph_corrupt(self, project: Path, built: CodeGraph) -> None:
        (graph_dir(str(project)) / "graph.json").write_text("{not json")
        assert load_graph(str(project)) is None

    def test_skips_hidden_and_vendor_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def top(): pass\n")
        vend = tmp_path / "node_modules" / "lib"
        vend.mkdir(parents=True)
        (vend / "v.py").write_text("def vendored(): pass\n")
        hid = tmp_path / ".secret"
        hid.mkdir()
        (hid / "h.py").write_text("def hidden(): pass\n")
        graph = build_graph(str(tmp_path))
        labels = {n["label"] for n in graph.nodes.values()}
        assert "top" in labels
        assert "vendored" not in labels
        assert "hidden" not in labels

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "ok.py").write_text("def fine(): pass\n")
        bad = tmp_path / "bad.py"
        bad.write_bytes(b"\xff\xfe invalid \xff")
        graph = build_graph(str(tmp_path))
        labels = {n["label"] for n in graph.nodes.values()}
        assert "fine" in labels

    def test_empty_dir(self, tmp_path: Path) -> None:
        graph = build_graph(str(tmp_path))
        assert graph.nodes == {}
        assert graph.edges == []


# ---------------------------------------------------------------------------
# Incremental (feature #5): SHA256 cache
# ---------------------------------------------------------------------------


class TestIncremental:
    def test_unchanged_files_not_reextracted(self, project: Path) -> None:
        build_graph(str(project))
        cache1 = json.loads((graph_dir(str(project)) / "cache.json").read_text())
        graph2 = build_graph(str(project), incremental=True)
        cache2 = json.loads((graph_dir(str(project)) / "cache.json").read_text())
        assert cache1 == cache2
        assert graph2.stats["reextracted"] == 0

    def test_changed_file_reextracted(self, project: Path) -> None:
        build_graph(str(project))
        (project / "helpers.py").write_text(
            PY_HELPERS + "\n\ndef brand_new_fn():\n    return util_fn()\n"
        )
        graph = build_graph(str(project), incremental=True)
        assert graph.stats["reextracted"] == 1
        assert graph.find_node("brand_new_fn") is not None
        # untouched files keep their nodes
        assert graph.find_node("Application") is not None

    def test_deleted_file_pruned(self, project: Path) -> None:
        build_graph(str(project))
        (project / "app.js").unlink()
        graph = build_graph(str(project), incremental=True)
        assert graph.find_node("Widget") is None
        assert graph.find_node("app.js") is None
        assert all(built_file != "app.js" for built_file in
                   {n["file"] for n in graph.nodes.values()})

    def test_new_file_added(self, project: Path) -> None:
        build_graph(str(project))
        (project / "extra.py").write_text("def extra_fn():\n    pass\n")
        graph = build_graph(str(project), incremental=True)
        assert graph.stats["reextracted"] == 1
        assert graph.find_node("extra_fn") is not None

    def test_full_rebuild_ignores_cache(self, project: Path) -> None:
        build_graph(str(project))
        graph = build_graph(str(project), incremental=False)
        assert graph.stats["reextracted"] == graph.stats["files"]

    def test_update_specific_files(self, project: Path) -> None:
        build_graph(str(project))
        (project / "main.py").write_text(PY_MAIN + "\ndef added():\n    pass\n")
        (project / "helpers.py").write_text(PY_HELPERS + "\ndef also():\n    pass\n")
        graph = build_graph(str(project), only_files=["main.py"])
        assert graph.find_node("added") is not None
        # helpers.py was not in only_files, so its change is not seen
        assert graph.find_node("also") is None


# ---------------------------------------------------------------------------
# Query / path / explain (feature #1)
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_finds_relevant_subgraph(self, built: CodeGraph) -> None:
        out = built.query("what does Application run do")
        assert "NODE" in out
        assert "Application" in out
        assert "main.py" in out
        assert "EDGE" in out

    def test_query_edge_format(self, built: CodeGraph) -> None:
        out = built.query("run stop")
        assert "--calls [EXTRACTED]-->" in out

    def test_query_exact_label_outranks_longer_substring(self) -> None:
        nodes = {
            "exact": {
                "id": "exact", "label": "UsefulTools", "kind": "class",
                "file": "useful_tools.py", "line": 10,
            },
            "test-a": {
                "id": "test-a", "label": "TestUsefulTools", "kind": "class",
                "file": "test_a.py", "line": 10,
            },
            "test-b": {
                "id": "test-b", "label": "TestUsefulToolsBranches", "kind": "class",
                "file": "test_b.py", "line": 10,
            },
            "test-c": {
                "id": "test-c", "label": "AnotherUsefulToolsTest", "kind": "class",
                "file": "test_c.py", "line": 10,
            },
        }
        graph = CodeGraph(nodes, [])

        out = graph.query("UsefulTools")

        assert out.splitlines()[0].startswith("NODE UsefulTools ")

    def test_query_no_match(self, built: CodeGraph) -> None:
        out = built.query("zzz_nonexistent_qqq")
        assert "No matching nodes" in out

    def test_query_token_budget(self, built: CodeGraph) -> None:
        out = built.query("Application", max_chars=200)
        assert len(out) <= 260  # budget + truncation notice

    def test_query_budget_prioritizes_seed_and_nearest_neighbors(self) -> None:
        nodes = {
            "target": {
                "id": "target", "label": "target_fn", "kind": "function",
                "file": "target.py", "line": 10,
            },
            "caller": {
                "id": "caller", "label": "Caller", "kind": "function",
                "file": "caller.py", "line": 5,
            },
            "hub": {
                "id": "hub", "label": "target.py", "kind": "file",
                "file": "target.py", "line": 1,
            },
        }
        edges = [
            {"source": "caller", "target": "target", "relation": "calls",
             "confidence": "EXTRACTED"},
            {"source": "hub", "target": "target", "relation": "defines",
             "confidence": "EXTRACTED"},
        ]
        # A file hub can connect the seed to hundreds of alphabetically earlier
        # nodes.  Under a tight hint budget, those distant nodes must never
        # displace the exact seed or its direct caller.
        for index in range(30):
            node_id = f"other-{index}"
            nodes[node_id] = {
                "id": node_id, "label": f"aaa_{index:02d}", "kind": "function",
                "file": "target.py", "line": 20 + index,
            }
            edges.append(
                {"source": "hub", "target": node_id, "relation": "defines",
                 "confidence": "EXTRACTED"}
            )
        graph = CodeGraph(nodes, edges)

        out = graph.query("target_fn", max_chars=260)

        assert out.splitlines()[0].startswith("NODE target_fn ")
        assert "NODE Caller " in out
        assert "EDGE Caller --calls [EXTRACTED]--> target_fn" in out
        first_distant = out.find("NODE aaa_")
        assert first_distant == -1 or out.index("NODE Caller ") < first_distant

    def test_path_between_nodes(self, built: CodeGraph) -> None:
        out = built.path("entry_point", "shared_calc")
        assert "entry_point" in out
        assert "shared_calc" in out
        assert "-->" in out
        # every hop shows relation + confidence
        assert "calls" in out
        assert "EXTRACTED" in out or "INFERRED" in out

    def test_path_no_route(self, built: CodeGraph) -> None:
        out = built.path("Widget", "shared_calc")
        assert "No path" in out

    def test_path_unknown_endpoint(self, built: CodeGraph) -> None:
        out = built.path("no_such_thing_xyz", "run")
        assert "No node matching" in out

    def test_explain_node(self, built: CodeGraph) -> None:
        out = built.explain("Application")
        assert "Application" in out
        assert "class" in out
        assert "main.py:4" in out
        assert "degree" in out
        assert "contains" in out

    def test_explain_unknown(self, built: CodeGraph) -> None:
        out = built.explain("no_such_thing_xyz")
        assert "No node matching" in out

    def test_explain_fuzzy_match(self, built: CodeGraph) -> None:
        out = built.explain("application")
        assert "Application" in out


# ---------------------------------------------------------------------------
# grep interception (feature #3)
# ---------------------------------------------------------------------------


class TestGrepHint:
    def test_hint_for_grep(self, project: Path, built: CodeGraph) -> None:
        hint = grep_hint("grep -rn 'Application' .", str(project))
        assert hint is not None
        assert "[code_graph]" in hint
        assert "Application" in hint

    def test_hint_for_rg(self, project: Path, built: CodeGraph) -> None:
        hint = grep_hint('rg "util_fn" src/', str(project))
        assert hint is not None
        assert "util_fn" in hint

    def test_hint_extracts_pattern_after_flags(
        self, project: Path, built: CodeGraph
    ) -> None:
        hint = grep_hint("grep -r -n --include=*.py shared_calc .", str(project))
        assert hint is not None
        assert "shared_calc" in hint

    def test_no_hint_without_graph(self, tmp_path: Path) -> None:
        assert grep_hint("grep -rn 'Application' .", str(tmp_path)) is None

    def test_no_hint_for_non_grep(self, project: Path, built: CodeGraph) -> None:
        assert grep_hint("ls -la", str(project)) is None
        assert grep_hint("echo grep", str(project)) is None

    def test_no_hint_when_pattern_unmatched(
        self, project: Path, built: CodeGraph
    ) -> None:
        assert grep_hint("grep -rn 'zzz_nope_qqq' .", str(project)) is None

    def test_no_hint_for_empty_or_flag_only(
        self, project: Path, built: CodeGraph
    ) -> None:
        assert grep_hint("grep -rn", str(project)) is None
        assert grep_hint("grep", str(project)) is None

    def test_no_hint_none_workdir(self) -> None:
        assert grep_hint("grep -rn 'Application' .", None) is None

    def test_no_hint_when_identifier_is_too_short_for_query(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "short.py").write_text("def x():\n    pass\n")
        build_graph(str(tmp_path))
        assert grep_hint("grep x short.py", str(tmp_path)) is None

    def test_bash_tool_intercepts_with_answer(self, project: Path, built: CodeGraph) -> None:
        tools = UsefulTools(work_dir=str(project))
        out = tools.Bash("grep -rn 'Application' .", "search")
        assert "[code_graph]" in out
        assert "NODE Application" in out
        # The expensive grep is denied: the interception response is the
        # answer, matching graphify's one-model-call hook pattern.
        assert "class Application" not in out

    def test_repeated_identifier_grep_falls_through_for_verification(
        self, project: Path, built: CodeGraph
    ) -> None:
        tools = UsefulTools(work_dir=str(project))
        command = "grep -n 'Application' main.py"

        first = tools.Bash(command, "initial graph lookup")
        second = tools.Bash(command, "verify with grep")

        assert "[code_graph]" in first
        assert "[code_graph]" not in second
        assert "class Application" in second

    def test_bash_tool_no_hint_for_other_commands(
        self, project: Path, built: CodeGraph
    ) -> None:
        tools = UsefulTools(work_dir=str(project))
        out = tools.Bash("echo hello_plain", "echo")
        assert "[code_graph]" not in out
        assert "hello_plain" in out
        unmatched = tools.Bash("grep -n 'Top level app' main.py", "literal search")
        assert "[code_graph]" not in unmatched
        assert "Top level app" in unmatched

    def test_bash_streaming_intercepts_before_spawn(
        self, project: Path, built: CodeGraph
    ) -> None:
        chunks: list[str] = []
        tools = UsefulTools(work_dir=str(project), stream_callback=chunks.append)
        out = tools.Bash("grep -rn 'Application' .", "search")
        assert "[code_graph]" in out
        assert "NODE Application" in out
        assert "class Application" not in out
        assert chunks == []


# ---------------------------------------------------------------------------
# Git post-commit hook (feature #5)
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _make_project(tmp_path)
    _git(tmp_path, "init")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    return tmp_path


def _hook_path(repo_dir: Path) -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    p = Path(out)
    return p if p.is_absolute() else repo_dir / p


class TestGitHook:
    def test_install_creates_executable_hook(self, repo: Path) -> None:
        msg = install_post_commit_hook(str(repo))
        assert "installed" in msg.lower()
        hook = _hook_path(repo) / "post-commit"
        assert hook.is_file()
        assert os.access(hook, os.X_OK)
        assert "kiss code_graph" in hook.read_text()

    def test_install_idempotent(self, repo: Path) -> None:
        install_post_commit_hook(str(repo))
        msg = install_post_commit_hook(str(repo))
        assert "already" in msg.lower()
        text = (_hook_path(repo) / "post-commit").read_text()
        assert text.count(">>> kiss code_graph hook >>>") == 1

    def test_install_appends_to_existing_hook(self, repo: Path) -> None:
        hook = _hook_path(repo) / "post-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\necho preexisting_hook_ran\n")
        hook.chmod(0o755)
        install_post_commit_hook(str(repo))
        text = hook.read_text()
        assert "preexisting_hook_ran" in text
        assert "kiss code_graph" in text

    def test_uninstall_removes_only_our_section(self, repo: Path) -> None:
        hook = _hook_path(repo) / "post-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\necho preexisting_hook_ran\n")
        hook.chmod(0o755)
        install_post_commit_hook(str(repo))
        msg = uninstall_post_commit_hook(str(repo))
        assert "removed" in msg.lower()
        text = hook.read_text()
        assert "preexisting_hook_ran" in text
        assert "kiss code_graph" not in text

    def test_uninstall_deletes_hook_when_only_ours(self, repo: Path) -> None:
        install_post_commit_hook(str(repo))
        uninstall_post_commit_hook(str(repo))
        hook = _hook_path(repo) / "post-commit"
        assert not hook.exists() or "kiss code_graph" not in hook.read_text()

    def test_uninstall_when_not_installed(self, repo: Path) -> None:
        msg = uninstall_post_commit_hook(str(repo))
        assert "not installed" in msg.lower()

    def test_install_outside_git_repo(self, tmp_path: Path) -> None:
        msg = install_post_commit_hook(str(tmp_path))
        assert "not a git repository" in msg.lower()

    def test_commit_triggers_background_update(self, repo: Path) -> None:
        build_graph(str(repo))
        install_post_commit_hook(str(repo))
        (repo / "hooked.py").write_text("def hook_added_fn():\n    pass\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "add hooked.py")
        deadline = time.monotonic() + 60
        found = False
        while time.monotonic() < deadline:
            graph = load_graph(str(repo))
            if graph is not None and graph.find_node("hook_added_fn") is not None:
                found = True
                break
            time.sleep(0.5)
        assert found, "post-commit hook did not update the graph in time"


# ---------------------------------------------------------------------------
# CLI (used by the git hook)
# ---------------------------------------------------------------------------


class TestCli:
    def test_cli_build_and_query(self, project: Path) -> None:
        env = {**os.environ}
        out = subprocess.run(
            [
                "python",
                "-m",
                "kiss.agents.sorcar.code_graph",
                "build",
                str(project),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert (graph_dir(str(project)) / "graph.json").is_file()
        q = subprocess.run(
            [
                "python",
                "-m",
                "kiss.agents.sorcar.code_graph",
                "query",
                str(project),
                "Application",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert q.returncode == 0
        assert "Application" in q.stdout

    def test_cli_update_with_files(self, project: Path) -> None:
        build_graph(str(project))
        (project / "cliadd.py").write_text("def cli_added():\n    pass\n")
        out = subprocess.run(
            [
                "python",
                "-m",
                "kiss.agents.sorcar.code_graph",
                "update",
                str(project),
                "cliadd.py",
            ],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, out.stderr
        graph = load_graph(str(project))
        assert graph is not None
        assert graph.find_node("cli_added") is not None

    def test_cli_bad_verb(self, project: Path) -> None:
        out = subprocess.run(
            ["python", "-m", "kiss.agents.sorcar.code_graph", "bogus", str(project)],
            capture_output=True,
            text=True,
        )
        assert out.returncode != 0

    def test_cli_update_deduplicates_live_hook_processes(
        self, project: Path
    ) -> None:
        build_graph(str(project))
        (project / "waiting.py").write_text("def waiting():\n    pass\n")
        lock = graph_dir(str(project)) / ".update.lock"
        lock.write_text(f"{os.getpid()}\n")
        out = subprocess.run(
            [
                "python", "-m", "kiss.agents.sorcar.code_graph",
                "update", str(project),
            ],
            capture_output=True,
            text=True,
        )
        try:
            assert out.returncode == 0, out.stderr
            assert "already running" in out.stdout.lower()
            graph = load_graph(str(project))
            assert graph is not None
            assert graph.find_node("waiting") is None
        finally:
            lock.unlink(missing_ok=True)

    def test_cli_update_recovers_dead_process_lock(self, project: Path) -> None:
        build_graph(str(project))
        (project / "recovered.py").write_text("def recovered():\n    pass\n")
        lock = graph_dir(str(project)) / ".update.lock"
        lock.write_text("99999999\n")
        out = subprocess.run(
            [
                "python", "-m", "kiss.agents.sorcar.code_graph",
                "update", str(project),
            ],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, out.stderr
        graph = load_graph(str(project))
        assert graph is not None
        assert graph.find_node("recovered") is not None
        assert not lock.exists()


# ---------------------------------------------------------------------------
# Agent tool (feature #1 exposure)
# ---------------------------------------------------------------------------


class TestAgentTool:
    def test_tool_created(self, project: Path) -> None:
        tool = make_code_graph_tool(str(project))
        assert tool is not None
        assert tool.__name__ == "code_graph"
        assert tool.__doc__ is not None
        assert "query" in tool.__doc__

    def test_tool_build_then_query(self, project: Path) -> None:
        tool = make_code_graph_tool(str(project))
        assert tool is not None
        out = tool(action="build")
        assert "nodes" in out
        out = tool(action="query", argument="Application run")
        assert "Application" in out

    def test_tool_query_without_graph_auto_builds(self, project: Path) -> None:
        tool = make_code_graph_tool(str(project))
        assert tool is not None
        out = tool(action="query", argument="Application")
        assert "Application" in out

    def test_tool_path_and_explain(self, project: Path, built: CodeGraph) -> None:
        tool = make_code_graph_tool(str(project))
        assert tool is not None
        assert "shared_calc" in tool(action="path", argument="entry_point shared_calc")
        assert "degree" in tool(action="explain", argument="Application")

    def test_tool_path_needs_two_names(self, project: Path, built: CodeGraph) -> None:
        tool = make_code_graph_tool(str(project))
        assert tool is not None
        out = tool(action="path", argument="only_one")
        assert "two" in out.lower()

    def test_tool_missing_argument(self, project: Path, built: CodeGraph) -> None:
        tool = make_code_graph_tool(str(project))
        assert tool is not None
        out = tool(action="query", argument="")
        assert "argument" in out.lower()

    def test_tool_unknown_action(self, project: Path) -> None:
        tool = make_code_graph_tool(str(project))
        assert tool is not None
        out = tool(action="frobnicate")
        assert "unknown action" in out.lower()

    def test_tool_hook_actions(self, repo: Path) -> None:
        tool = make_code_graph_tool(str(repo))
        assert tool is not None
        assert "installed" in tool(action="install_hook").lower()
        assert "removed" in tool(action="uninstall_hook").lower()

    def test_tool_registered_in_sorcar_agent(self, project: Path) -> None:
        # Minimal-coupling check: the agent wires the tool through
        # make_code_graph_tool without further dependencies.
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        agent = SorcarAgent("t")
        agent.work_dir = str(project)
        tools = agent._get_tools()
        names = {getattr(t, "__name__", "") for t in tools}
        assert "code_graph" in names

    def test_docker_bash_also_gets_query_before_grep_hint(
        self, project: Path, built: CodeGraph
    ) -> None:
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        class FakeDockerManager:
            def Bash(self, command: str, description: str) -> str:  # noqa: N802
                return f"docker-result:{command}:{description}"

        agent = SorcarAgent("docker")
        agent.work_dir = str(project)
        agent.docker_manager = FakeDockerManager()
        agent._use_web_tools = False
        bash = agent._get_tools()[0]
        out = bash("rg Application .", "search")
        assert "[code_graph]" in out
        assert "NODE Application" in out
        assert "docker-result" not in out

        verification = bash("rg Application .", "verify")
        assert "[code_graph]" not in verification
        assert "docker-result:rg Application .:verify" == verification

# ---------------------------------------------------------------------------
# Independent review regressions (gpt-5.6-sol)
# ---------------------------------------------------------------------------


class TestReviewedCorrectness:
    def test_duplicate_method_names_keep_calls_in_their_own_class(
        self, tmp_path: Path
    ) -> None:
        """Definition/call identity must not collapse duplicate labels."""
        (tmp_path / "duplicate.py").write_text(
            "class First:\n"
            "    def run(self):\n"
            "        self.helper()\n"
            "    def helper(self):\n"
            "        pass\n\n"
            "class Second:\n"
            "    def run(self):\n"
            "        self.helper()\n"
            "    def helper(self):\n"
            "        pass\n"
        )
        graph = build_graph(str(tmp_path))
        runs = sorted(
            (n for n in graph.nodes.values() if n["label"] == "run"),
            key=lambda n: n["line"],
        )
        helpers = sorted(
            (n for n in graph.nodes.values() if n["label"] == "helper"),
            key=lambda n: n["line"],
        )
        assert len(runs) == len(helpers) == 2
        assert graph.find_edge(runs[0]["id"], helpers[0]["id"], "calls")
        assert graph.find_edge(runs[1]["id"], helpers[1]["id"], "calls")
        assert not graph.find_edge(runs[0]["id"], helpers[1]["id"], "calls")
        assert not graph.find_edge(runs[1]["id"], helpers[0]["id"], "calls")

    def test_reverse_path_preserves_edge_direction(self, built: CodeGraph) -> None:
        out = built.path("stop", "run")
        assert "stop <--calls [EXTRACTED]-- run" in out

    def test_path_from_node_to_itself_is_informative(self, built: CodeGraph) -> None:
        assert "same node" in built.path("run", "run").lower()

    def test_query_expands_three_hops(self, built: CodeGraph) -> None:
        out = built.query("entry_point")
        assert "shared_calc" in out

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("grep -e Application .", "Application"),
            ("grep --regexp=util_fn .", "util_fn"),
            ("rg -e shared_calc .", "shared_calc"),
        ],
    )
    def test_grep_pattern_flags_are_intercepted(
        self, project: Path, built: CodeGraph, command: str, expected: str
    ) -> None:
        hint = grep_hint(command, str(project))
        assert hint is not None
        assert expected in hint

    def test_post_commit_handles_filename_with_spaces(self, repo: Path) -> None:
        build_graph(str(repo))
        install_post_commit_hook(str(repo))
        (repo / "has space.py").write_text("def spaced_name():\n    pass\n")
        _git(repo, "add", "has space.py")
        _git(repo, "commit", "-m", "space path")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            graph = load_graph(str(repo))
            if graph is not None and graph.find_node("spaced_name") is not None:
                break
            time.sleep(0.25)
        else:
            pytest.fail("post-commit update lost a whitespace-containing path")

    def test_supported_language_tables_end_to_end(self, tmp_path: Path) -> None:
        """Every advertised language is parsed, not merely table-listed."""
        sources = {
            "sample.ts": "interface TFace {}\nfunction tsFun() {}\n",
            "sample.tsx": "interface XFace {}\nfunction tsxFun() { return <div/>; }\n",
            "sample.go": "package main\ntype GoType struct{}\nfunc goFun() {}\n",
            "sample.rs": "struct RustType {}\nfn rust_fun() {}\n",
            "Sample.java": "class JavaType { void javaMethod() {} }\n",
            "sample.rb": "module RubyMod\n  def ruby_method\n  end\nend\n",
            "sample.c": "struct CType { int x; };\nint c_fun(void) { return 0; }\n",
            "sample.cpp": "class CppType {};\nint cpp_fun() { return 0; }\n",
        }
        for name, source in sources.items():
            (tmp_path / name).write_text(source)
        graph = build_graph(str(tmp_path), incremental=False)
        labels = {n["label"] for n in graph.nodes.values()}
        assert {
            "TFace", "tsFun", "XFace", "tsxFun", "GoType", "goFun",
            "RustType", "rust_fun", "JavaType", "javaMethod", "RubyMod",
            "ruby_method", "CType", "c_fun", "CppType", "cpp_fun",
        } <= labels


def test_recursive_call_is_retained(tmp_path: Path) -> None:
    (tmp_path / "recursive.py").write_text(
        "def recurse():\n    recurse()\n"
    )
    graph = build_graph(str(tmp_path))
    node = graph.find_node("recurse")
    assert node is not None
    assert graph.find_edge(node["id"], node["id"], "calls") is not None


class _StubNode:
    """Tiny tree-sitter-shaped node for malformed-AST defensive paths."""

    def __init__(
        self,
        node_type: str,
        text: bytes = b"",
        *,
        children: list["_StubNode"] | None = None,
        named_children: list["_StubNode"] | None = None,
        fields: dict[str, "_StubNode"] | None = None,
        line: int = 0,
    ) -> None:
        self.type = node_type
        self.text = text
        self.children = children or []
        self.named_children = (
            self.children if named_children is None else named_children
        )
        self._fields = fields or {}
        self.start_point = (line, 0)

    def child_by_field_name(self, name: str) -> "_StubNode | None":
        return self._fields.get(name)


class TestDefensiveAndCoveragePaths:
    """Fault-injection complements the end-to-end tests for full branches."""

    def test_parser_and_malformed_ast_degrade_without_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert cg._parser_for("certainly_not_a_tree_sitter_language") is None
        assert cg._extract_file("x.py", b"", "not-a-language") is None

        invalid_name = _StubNode("class_definition")
        empty_import = _StubNode("import_statement", b"import ")
        missing_callee = _StubNode("call")
        invalid_callee = _StubNode(
            "call",
            fields={"function": _StubNode("parenthesized_expression", b"(lambda: 1)")},
        )
        root = _StubNode(
            "module",
            children=[invalid_name, empty_import, missing_callee, invalid_callee],
        )
        parser = SimpleNamespace(
            parse=lambda _source: SimpleNamespace(root_node=root)
        )
        monkeypatch.setattr(cg, "_parser_for", lambda _lang: parser)
        record = cg._extract_file("broken.py", b"broken", "python")
        assert record == {"defs": [], "imports": [], "calls": []}

    def test_name_and_import_fallbacks(self) -> None:
        identifier = _StubNode("identifier", b"fallback_name")
        wrapped = _StubNode("function_declarator", named_children=[identifier])
        function = _StubNode("function_definition", fields={"declarator": wrapped})
        assert cg._c_declarator_name(function) == "fallback_name"
        assert cg._c_declarator_name(_StubNode("function_definition")) is None
        no_identifier = _StubNode(
            "function_declarator",
            named_children=[_StubNode("parameter_list", b"()")],
        )
        assert cg._c_declarator_name(
            _StubNode("function_definition", fields={"declarator": no_identifier})
        ) is None
        assert cg._def_name(_StubNode("class_definition"), "field:name") is None
        assert cg._def_name(
            _StubNode("type_declaration", named_children=[_StubNode("comment")]),
            "type_spec",
        ) is None
        assert cg._def_name(
            _StubNode(
                "type_declaration",
                named_children=[_StubNode("type_spec")],
            ),
            "type_spec",
        ) is None
        assert cg._import_label(_StubNode("import", b"import ")) is None
        assert cg._import_label(_StubNode("import", b"import x from")) == "x"
        assert cg._import_label(_StubNode("import", b'import ""')) is None

    def test_adjacency_ignores_dangling_edges_and_is_cached(self) -> None:
        node = {"id": "a", "label": "A", "kind": "class", "file": "", "line": 0}
        graph = CodeGraph(
            {"a": node},
            [{"source": "a", "target": "missing", "relation": "calls",
              "confidence": "INFERRED"}],
        )
        assert graph._adjacency() == {"a": []}
        assert graph._adjacency() is graph._adj
        assert graph.path("A", "missing") == "No node matching 'missing' in the code graph."

    def test_ambiguous_local_call_is_not_invented(self, tmp_path: Path) -> None:
        (tmp_path / "ambiguous.py").write_text(
            "class C:\n"
            "    def run(self):\n        self.same()\n"
            "    def same(self):\n        pass\n"
            "    def same(self):\n        pass\n"
        )
        graph = build_graph(str(tmp_path))
        run = graph.find_node("run")
        assert run is not None
        assert not any(
            edge["source"] == run["id"] and edge["relation"] == "calls"
            for edge in graph.edges
        )

    def test_duplicate_calls_are_deduplicated(self, tmp_path: Path) -> None:
        (tmp_path / "twice.py").write_text(
            "def target():\n    pass\n\n"
            "def source():\n    target()\n    target()\n"
        )
        graph = build_graph(str(tmp_path))
        source = graph.find_node("source")
        target = graph.find_node("target")
        assert source is not None and target is not None
        calls = [
            edge for edge in graph.edges
            if edge["source"] == source["id"] and edge["target"] == target["id"]
        ]
        assert len(calls) == 1

    def test_bad_caller_record_is_safely_ignored(self) -> None:
        record = {
            "defs": [], "imports": [],
            "calls": [{"caller": 99, "callee": "none"}],
        }
        # The record is intentionally malformed to exercise stale/corrupt
        # cache defense, not normal extraction.
        graph = cg._assemble({"bad.py": record}, {})
        assert graph.find_node("bad.py") is not None
        assert not any(edge["relation"] == "calls" for edge in graph.edges)

    def test_atomic_write_removes_temporary_after_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        destination = tmp_path / "out.json"

        def fail_replace(_source: str, _destination: Path) -> None:
            raise OSError("injected replace failure")

        monkeypatch.setattr(cg.os, "replace", fail_replace)
        with pytest.raises(OSError, match="injected"):
            cg._atomic_write_json(destination, {"x": 1})
        assert list(tmp_path.iterdir()) == []

    def test_incompatible_cache_and_graph_are_rejected(self, project: Path) -> None:
        graph = build_graph(str(project))
        cache_path = graph_dir(str(project)) / "cache.json"
        cache = json.loads(cache_path.read_text())
        cache["version"] = -1
        cache_path.write_text(json.dumps(cache))
        rebuilt = build_graph(str(project))
        assert rebuilt.stats["reextracted"] == graph.stats["files"]

        graph_path = graph_dir(str(project)) / "graph.json"
        payload = json.loads(graph_path.read_text())
        payload["version"] = -1
        graph_path.write_text(json.dumps(payload))
        assert load_graph(str(project)) is None

    def test_initial_only_files_skips_uncached_files(self, project: Path) -> None:
        graph = build_graph(str(project), only_files=["main.py"])
        assert graph.find_node("Application") is not None
        assert graph.find_node("Widget") is None
        assert graph.find_node("util_fn") is None

    def test_read_error_and_parser_error_skip_only_affected_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = tmp_path / "good.py"
        unreadable = tmp_path / "unreadable.py"
        good.write_text("def good():\n    pass\n")
        unreadable.write_text("def unreadable():\n    pass\n")
        original_read = Path.read_bytes

        def read_bytes(path: Path) -> bytes:
            if path.name == "unreadable.py":
                raise OSError("injected unreadable file")
            return original_read(path)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
        graph = build_graph(str(tmp_path))
        assert graph.find_node("good") is not None
        assert graph.find_node("unreadable") is None

        monkeypatch.undo()
        monkeypatch.setitem(cg._EXT_TO_LANG, ".py", "invalid-language")
        graph = build_graph(str(tmp_path), incremental=False)
        assert graph.nodes == {}

    @pytest.mark.parametrize(
        "command",
        ["grep 'unterminated", "grep -e", "grep --regexp=", "grep -eCompact ."],
    )
    def test_grep_edge_syntax_does_not_crash(
        self, project: Path, built: CodeGraph, command: str
    ) -> None:
        result = grep_hint(command, str(project))
        if command == "grep -eCompact .":
            assert result is None  # valid syntax, but no matching graph node
        else:
            assert result is None

    def test_install_handles_no_newline_and_corrupt_markers(self, repo: Path) -> None:
        hook = _hook_path(repo) / "post-commit"
        hook.write_text("#!/bin/sh\necho existing")
        install_post_commit_hook(str(repo))
        assert "existing\n# >>>" in hook.read_text()
        hook.write_text("#!/bin/sh\n# >>> kiss code_graph hook >>>\n")
        # Corrupt markers must produce a diagnostic, never ValueError.
        assert "corrupt" in uninstall_post_commit_hook(str(repo)).lower()

    def test_uninstall_outside_repository(self, tmp_path: Path) -> None:
        assert "not a git repository" in uninstall_post_commit_hook(str(tmp_path))

    def test_non_shell_hook_is_not_corrupted(self, repo: Path) -> None:
        hook = _hook_path(repo) / "post-commit"
        original = "#!/usr/bin/env python3\nprint('keep me')\n"
        hook.write_text(original)
        result = install_post_commit_hook(str(repo))
        assert "shell" in result.lower()
        assert hook.read_text() == original

    def test_graph_storage_is_git_ignored(self, repo: Path) -> None:
        build_graph(str(repo))
        assert _git(repo, "status", "--porcelain", "--", ".kiss") == ""

    def test_absolute_hooks_path(self, repo: Path, tmp_path: Path) -> None:
        hooks = tmp_path / "absolute-hooks"
        _git(repo, "config", "core.hooksPath", str(hooks))
        assert "installed" in install_post_commit_hook(str(repo)).lower()
        assert (hooks / "post-commit").is_file()

    def test_missing_tree_sitter_hides_tool(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
        assert make_code_graph_tool(str(project)) is None


class TestDirectCliAndLockCoverage:
    def test_main_all_verbs(self, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert cg.main([]) == 2
        assert "usage:" in capsys.readouterr().err
        assert cg.main(["build", str(project)]) == 0
        assert "built:" in capsys.readouterr().out
        assert cg.main(["query", str(project), "Application"]) == 0
        assert "Application" in capsys.readouterr().out
        (project / "direct.py").write_text("def direct_added():\n    pass\n")
        assert cg.main(["update", str(project), "direct.py"]) == 0
        assert "updated:" in capsys.readouterr().out
        assert cg.main(["nonsense", str(project)]) == 2
        assert "unknown verb" in capsys.readouterr().err

    def test_main_update_skips_live_lock(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        graph_dir(str(project)).mkdir(parents=True, exist_ok=True)
        lock = graph_dir(str(project)) / ".update.lock"
        lock.write_text(f"{os.getpid()}\n")
        try:
            assert cg.main(["update", str(project)]) == 0
            assert "already running" in capsys.readouterr().out
        finally:
            lock.unlink(missing_ok=True)

    def test_pid_probe_outcomes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not cg._pid_is_running(0)
        assert cg._pid_is_running(os.getpid())

        def process_gone(_pid: int, _signal: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(cg.os, "kill", process_gone)
        assert not cg._pid_is_running(123)

        def permission_denied(_pid: int, _signal: int) -> None:
            raise PermissionError

        monkeypatch.setattr(cg.os, "kill", permission_denied)
        assert cg._pid_is_running(123)

        def generic_os_error(_pid: int, _signal: int) -> None:
            raise OSError

        monkeypatch.setattr(cg.os, "kill", generic_os_error)
        assert not cg._pid_is_running(123)

    def test_lock_lifecycle_and_stale_malformed_lock(self, tmp_path: Path) -> None:
        lock = cg._acquire_update_lock(str(tmp_path))
        assert lock is not None and lock.is_file()
        cg._release_update_lock(lock)
        assert not lock.exists()

        lock.write_text("not-a-pid")
        assert cg._acquire_update_lock(str(tmp_path)) is None
        old = time.time() - cg._STALE_LOCK_SECONDS - 10
        os.utime(lock, (old, old))
        acquired = cg._acquire_update_lock(str(tmp_path))
        assert acquired == lock
        cg._release_update_lock(lock)

        lock.write_text("99999999\n")
        acquired = cg._acquire_update_lock(str(tmp_path))
        assert acquired == lock
        cg._release_update_lock(lock)

    def test_release_never_removes_another_owners_lock(self, tmp_path: Path) -> None:
        lock = tmp_path / ".update.lock"
        lock.write_text("99999999")
        cg._release_update_lock(lock)
        assert lock.exists()
        lock.write_text("malformed")
        cg._release_update_lock(lock)
        assert lock.exists()
        lock.unlink()
        cg._release_update_lock(lock)  # missing is harmless


def test_last_defensive_branches(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # grep option whose value is metadata, not the search pattern.
    build_graph(str(repo))
    assert grep_hint("grep -A 2 Application .", str(repo)) is not None

    # Shell-hook recognizer rejects malformed/empty/non-shebang scripts.
    assert not cg._is_shell_hook("echo no-shebang\n")
    assert not cg._is_shell_hook("#!'unterminated\n")
    assert not cg._is_shell_hook("#!\n")
    assert cg._is_shell_hook("#!/usr/bin/env bash\n")

    hook = _hook_path(repo) / "post-commit"
    hook.write_text("")
    assert "installed" in install_post_commit_hook(str(repo)).lower()
    uninstall_post_commit_hook(str(repo))

    hook.write_text("#!/bin/sh\n# >>> kiss code_graph hook >>>\n")
    assert "corrupt" in install_post_commit_hook(str(repo)).lower()
    hook.write_text("#!/bin/sh\necho no-marker\n")
    assert "not installed" in uninstall_post_commit_hook(str(repo)).lower()

    # Existing git exclude without a newline and a second idempotent call.
    exclude_text = _git(repo, "rev-parse", "--git-path", "info/exclude").strip()
    exclude = Path(exclude_text)
    if not exclude.is_absolute():
        exclude = repo / exclude
    exclude.write_text("another-pattern")
    cg._ensure_graph_git_excluded(str(repo))
    assert exclude.read_text() == "another-pattern\n.kiss/code_graph/\n"
    cg._ensure_graph_git_excluded(str(repo))
    assert exclude.read_text().count(".kiss/code_graph/") == 1

    # Force the absolute-path result without replacing real filesystem logic.
    absolute = tmp_path / "absolute-exclude"
    real_run = cg.subprocess.run

    def absolute_git_path(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(stdout=str(absolute))

    monkeypatch.setattr(cg.subprocess, "run", absolute_git_path)
    cg._ensure_graph_git_excluded(str(repo))
    assert absolute.read_text() == ".kiss/code_graph/\n"
    monkeypatch.setattr(cg.subprocess, "run", real_run)


def test_hook_read_errors_are_diagnostics(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _hook_path(repo) / "post-commit"
    hook.write_text("#!/bin/sh\n")
    real_read_text = Path.read_text

    def unreadable(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == hook:
            raise OSError("injected")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert "cannot read" in install_post_commit_hook(str(repo)).lower()
    assert "cannot read" in uninstall_post_commit_hook(str(repo)).lower()


def test_lock_stat_and_unlink_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = graph_dir(str(tmp_path))
    storage.mkdir(parents=True)
    lock = storage / ".update.lock"
    lock.write_text("malformed")
    real_stat = Path.stat

    def missing_stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == lock:
            raise OSError("injected disappearing lock")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", missing_stat)
    assert cg._acquire_update_lock(str(tmp_path)) is None
    monkeypatch.setattr(Path, "stat", real_stat)

    lock.write_text("99999999")
    real_unlink = Path.unlink
    raised = False

    def racing_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal raised
        if path == lock and not raised:
            raised = True
            real_unlink(path, *args, **kwargs)
            raise FileNotFoundError
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", racing_unlink)
    acquired = cg._acquire_update_lock(str(tmp_path))
    assert acquired == lock
    cg._release_update_lock(lock)
