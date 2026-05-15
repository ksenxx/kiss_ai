"""Integration tests verifying that the Sorcar agent follows SYSTEM.md instructions.

Each test runs a real LLM call with a focused task and inspects the agent's
tool-call sequence to confirm compliance with a specific SYSTEM.md rule.

Violations confirmed by database analysis of 30 recent tasks:
  - USER_PREFS.md read as first tool: 33% compliance
  - SORCAR.md read: 46% compliance
  - Read before Edit: 50% violation rate
  - uv run check before finish: 44% skip rate
  - Web research 30-site rule: 5 URLs visited (task 1222), 2 URLs (task 1179)
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.printer import Printer

# ---------------------------------------------------------------------------
# Capturing printer — records all tool_call / tool_result events for inspection
# ---------------------------------------------------------------------------


class _CapturingPrinter(Printer):
    """Printer that silently records every event for later inspection."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Record an event.

        Args:
            content: The content payload.
            type: Event type string.
            **kwargs: Extra metadata (tool_name, tool_input, etc.).

        Returns:
            Empty string (no display).
        """
        self.events.append({"type": type, "content": content, **kwargs})
        return ""

    def token_callback(self, token: str) -> None:
        """No-op token handler.

        Args:
            token: Ignored.
        """

    def thinking_callback(self, is_start: bool) -> None:
        """No-op thinking handler.

        Args:
            is_start: Ignored.
        """

    def reset(self) -> None:
        """No-op reset."""


def _tool_calls(printer: _CapturingPrinter) -> list[dict[str, Any]]:
    """Extract tool_call events in order."""
    return [e for e in printer.events if e["type"] == "tool_call"]


def _run_agent(
    task: str,
    work_dir: str,
    max_steps: int = 12,
    max_budget: float = 1.0,
    web_tools: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a SorcarAgent and return (parsed_result, tool_calls).

    Args:
        task: The prompt string.
        work_dir: Working directory for the agent.
        max_steps: Step limit.
        max_budget: Budget limit in USD.
        web_tools: Whether to enable browser tools.

    Returns:
        Tuple of (result_dict, list_of_tool_call_events).
    """
    printer = _CapturingPrinter()
    agent = SorcarAgent("SystemMDTest")
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        raw = agent.run(
            prompt_template=task,
            model_name="claude-sonnet-4-5",
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=work_dir,
            printer=printer,
            web_tools=web_tools,
            verbose=False,
        )
    finally:
        os.chdir(old_cwd)
    result = yaml.safe_load(raw)
    calls = _tool_calls(printer)
    return result, calls


# ---------------------------------------------------------------------------
# Test 1: USER_PREFS.md must be read at the start of every task
# SYSTEM.md: "Read PWD/USER_PREFS.md at the start of every task."
# DB evidence: Only 33% of tasks read it as the first tool call.
# ---------------------------------------------------------------------------

def test_user_prefs_read_at_start() -> None:
    """Agent must Read USER_PREFS.md as its first tool call."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## User Preferences\n- Use concise variable names\n")

    _, calls = _run_agent(
        task="Create a file called hello.txt containing 'hello world'.",
        work_dir=work_dir,
    )

    assert len(calls) >= 1, "Agent made no tool calls"
    first = calls[0]
    first_name = first.get("content", "")
    first_input = first.get("tool_input", {})
    first_path = first_input.get("file_path", "") if isinstance(first_input, dict) else ""
    assert first_name == "Read" or "Read" in str(first), (
        f"First tool call should be Read, got: {first_name}"
    )
    assert "USER_PREFS" in first_path, (
        f"First Read should target USER_PREFS.md, got path: {first_path}"
    )


# ---------------------------------------------------------------------------
# Test 2: SORCAR.md must be read for project-specific overrides
# SYSTEM.md: "Read PWD/SORCAR.md for overriding project-specific instructions."
# DB evidence: Only 46% of tasks read it.
# ---------------------------------------------------------------------------

def test_sorcar_md_read() -> None:
    """Agent must Read SORCAR.md when it exists in the work directory."""
    work_dir = tempfile.mkdtemp()
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project Instructions\n- Always use snake_case.\n")
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")

    _, calls = _run_agent(
        task="Create a file called greeting.txt with the text 'hi'.",
        work_dir=work_dir,
    )

    read_paths = [
        (c.get("tool_input") or {}).get("file_path", "")
        for c in calls
        if "Read" in str(c.get("content", ""))
    ]
    assert any("SORCAR.md" in p for p in read_paths), (
        f"Agent never read SORCAR.md. Read paths: {read_paths}"
    )


# ---------------------------------------------------------------------------
# Test 3: Read every file before modifying it
# SYSTEM.md: "Read every file before modifying it."
# DB evidence: 50% of coding tasks Edit without a prior Read.
# ---------------------------------------------------------------------------

def test_read_before_edit() -> None:
    """Agent must Read a file before calling Edit on it."""
    work_dir = tempfile.mkdtemp()
    target = os.path.join(work_dir, "greet.py")
    with open(target, "w") as f:
        f.write("def greet():\n    return 'hello'\n")
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")

    _, calls = _run_agent(
        task=(
            f"Change the return value in {target} from 'hello' to 'goodbye'. "
            "Use Edit() to make the change."
        ),
        work_dir=work_dir,
    )

    # Build timeline of read/edit events per file
    read_files: set[str] = set()
    for c in calls:
        name = c.get("content", "")
        inp = c.get("tool_input", {}) if isinstance(c.get("tool_input"), dict) else {}
        fp = inp.get("file_path", "")
        if "Read" in str(name) and fp:
            read_files.add(fp)
        if "Edit" in str(name) and fp:
            assert fp in read_files, (
                f"Edit called on {fp} without prior Read. "
                f"Files read so far: {read_files}"
            )


# ---------------------------------------------------------------------------
# Test 4: Pre-finish lint/typecheck for coding tasks
# SYSTEM.md: "Run required checks (lint, typecheck, tests); fix any failures."
# Sorcar-specific: "Lint/typecheck/format: `uv run check --full`."
# DB evidence: 44% of coding tasks skip lint check.
# ---------------------------------------------------------------------------

def test_lint_check_before_finish() -> None:
    """Agent must run `uv run check` before calling finish on coding tasks."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")
    target = os.path.join(work_dir, "math_utils.py")
    with open(target, "w") as f:
        f.write("def add(a: int, b: int) -> int:\n    return a + b\n")

    _, calls = _run_agent(
        task=(
            f"Add a multiply(a: int, b: int) -> int function to {target}. "
            "Run lint checks before finishing."
        ),
        work_dir=work_dir,
        max_steps=15,
    )

    # Check that Bash with "uv run check" or "check" was called before finish
    bash_calls = [
        c for c in calls
        if "Bash" in str(c.get("content", ""))
    ]
    has_check = any(
        "uv run check" in str(c.get("tool_input", {}))
        or "check" in str(c.get("tool_input", {}).get("command", "")).lower()
        for c in bash_calls
    )
    assert has_check, (
        "Agent did not run lint/typecheck before finishing. "
        f"Bash commands: {[c.get('tool_input', {}).get('command', '') for c in bash_calls]}"
    )


# ---------------------------------------------------------------------------
# Test 5: visibility_constraint — full answer in finish(summary=...)
# SYSTEM.md: "Compose the full detailed answer directly inside the summary
#   string of finish()."
# ---------------------------------------------------------------------------

def test_visibility_constraint_full_answer() -> None:
    """For informational questions, the finish summary must contain the full answer."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")

    result, _ = _run_agent(
        task="What is the Fibonacci sequence? Give the first 10 numbers.",
        work_dir=work_dir,
        max_steps=5,
        max_budget=0.5,
    )

    summary = result.get("summary", "")
    # The summary should contain actual Fibonacci numbers, not just a meta-description
    # First 10: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    assert "34" in summary or "21" in summary, (
        f"Summary doesn't contain Fibonacci numbers (expected 34 or 21). "
        f"Summary: {summary[:300]}"
    )
    # Should not be a meta-description like "I found the answer"
    assert len(summary) > 50, (
        f"Summary too short — may be a meta-description instead of full answer. "
        f"Summary: {summary}"
    )


# ---------------------------------------------------------------------------
# Test 6: Web research must create information file with proper format
# SYSTEM.md: "Create PWD/tmp/information-{unique_id}.md with header:
#   `# Web Research — Websites visited: 0/30`"
# DB evidence: Task 1222 visited 5 sites, task 1179 visited 2 sites.
# ---------------------------------------------------------------------------

def test_web_research_creates_information_file() -> None:
    """Web research task must create an information-*.md file with proper header."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")
    tmp_dir = os.path.join(work_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    _, calls = _run_agent(
        task=(
            "Research: What is the population of Tokyo? "
            "Search the internet for this information."
        ),
        work_dir=work_dir,
        max_steps=20,
        max_budget=2.0,
        web_tools=True,
    )

    # Check that an information-*.md file was created in tmp/
    info_files = [
        f for f in os.listdir(tmp_dir)
        if f.startswith("information-") and f.endswith(".md")
    ]
    assert len(info_files) >= 1, (
        f"No information-*.md file created in {tmp_dir}. "
        f"Files: {os.listdir(tmp_dir) if os.path.exists(tmp_dir) else 'dir missing'}"
    )

    # Check file has proper header format
    with open(os.path.join(tmp_dir, info_files[0])) as f:
        content = f.read()
    assert "Web Research" in content, (
        f"Information file missing 'Web Research' header. Content: {content[:200]}"
    )


if __name__ == "__main__":
    import sys
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    tests = [
        ("test_user_prefs_read_at_start", test_user_prefs_read_at_start),
        ("test_sorcar_md_read", test_sorcar_md_read),
        ("test_read_before_edit", test_read_before_edit),
        ("test_lint_check_before_finish", test_lint_check_before_finish),
        ("test_visibility_constraint_full_answer", test_visibility_constraint_full_answer),
        ("test_web_research_creates_information_file", test_web_research_creates_information_file),
    ]
    for name, func in tests:
        if test_name and test_name != name:
            continue
        print(f"\n{'='*60}\nRunning {name}...")
        try:
            func()
            print("  PASSED")
        except AssertionError as e:
            print(f"  FAILED: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
