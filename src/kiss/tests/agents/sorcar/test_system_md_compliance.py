"""Integration tests verifying that the Sorcar agent follows SYSTEM.md instructions.

Each test runs a real LLM call with a focused task and inspects the agent's
tool-call sequence to confirm compliance with a specific SYSTEM.md rule.

Violations confirmed by database analysis of 91 tasks from 2026-05-14:
  - USER_PREFS.md read first via Read(): 10% compliance (54/60 violated)
  - SORCAR.md read: 8% compliance (55/60 violated)
  - Agent uses Bash(cat) instead of Read(): 11 tasks
  - SORCAR read before USER_PREFS (wrong order): 1 task
  - Edit without prior Read: 12% violation rate (7/60)
  - No uv run check for code tasks: 70% skip rate (7/10)
  - Web research under 30 sites: 4 tasks (1-5 URLs, curl used to evade)
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
            content: The content payload (tool name for tool_call events).
            type: Event type string.
            **kwargs: Extra metadata (tool_input, etc.).

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


def _get_tool_name(call: dict[str, Any]) -> str:
    """Get tool name from a tool_call event.

    Args:
        call: A tool_call event dict.

    Returns:
        The tool name string.
    """
    return str(call.get("content", ""))


def _get_tool_input(call: dict[str, Any]) -> dict[str, Any]:
    """Get tool input args from a tool_call event.

    Args:
        call: A tool_call event dict.

    Returns:
        The tool_input dict, or empty dict.
    """
    inp = call.get("tool_input", {})
    return inp if isinstance(inp, dict) else {}


# ---------------------------------------------------------------------------
# Test 1: USER_PREFS.md must be the VERY FIRST Read() tool call
# SYSTEM.md: "Your VERY FIRST tool call in every task MUST be Read(USER_PREFS.md)"
# DB evidence: 90% of tasks violated this (54/60)
# ---------------------------------------------------------------------------

def test_user_prefs_read_is_first_tool_call() -> None:
    """Agent must call Read(USER_PREFS.md) as its very first tool call."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## User Preferences\n- Use concise variable names\n")
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project\n- Test project\n")

    _, calls = _run_agent(
        task="Create a file called hello.txt containing 'hello world'.",
        work_dir=work_dir,
    )

    assert len(calls) >= 1, "Agent made no tool calls"
    first = calls[0]
    first_name = _get_tool_name(first)
    first_path = _get_tool_input(first).get("file_path", "")
    assert first_name == "Read", (
        f"First tool call must be Read, got: {first_name}"
    )
    assert "USER_PREFS" in first_path, (
        f"First Read must target USER_PREFS.md, got path: {first_path}"
    )


# ---------------------------------------------------------------------------
# Test 2: SORCAR.md must be read as the SECOND tool call (not via cat)
# SYSTEM.md: "Your SECOND tool call MUST be Read(SORCAR.md)"
# DB evidence: 92% of tasks never read SORCAR.md at all (55/60)
# ---------------------------------------------------------------------------

def test_sorcar_md_read_as_second_call() -> None:
    """Agent must call Read(SORCAR.md) as its second tool call."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project Instructions\n- Always use snake_case.\n")

    _, calls = _run_agent(
        task="Create a file called greeting.txt with the text 'hi'.",
        work_dir=work_dir,
    )

    assert len(calls) >= 2, f"Agent made fewer than 2 tool calls ({len(calls)})"
    second = calls[1]
    second_name = _get_tool_name(second)
    second_path = _get_tool_input(second).get("file_path", "")
    assert second_name == "Read", (
        f"Second tool call must be Read, got: {second_name}"
    )
    assert "SORCAR.md" in second_path, (
        f"Second Read must target SORCAR.md, got path: {second_path}"
    )


# ---------------------------------------------------------------------------
# Test 3: Agent must NOT use Bash(cat) to read USER_PREFS or SORCAR
# SYSTEM.md: "Do NOT use Bash('cat USER_PREFS.md ...') or Bash('head ...')"
# DB evidence: 11 tasks used cat/head for USER_PREFS, 6 for SORCAR
# ---------------------------------------------------------------------------

def test_no_bash_cat_for_mandatory_files() -> None:
    """Agent must not use Bash(cat/head) to read USER_PREFS.md or SORCAR.md."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n- Test pref\n")
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project\n")

    _, calls = _run_agent(
        task="What is 2+2? Just answer the question.",
        work_dir=work_dir,
        max_steps=8,
    )

    # Check no Bash calls contain cat/head of USER_PREFS or SORCAR
    for call in calls:
        if _get_tool_name(call) == "Bash":
            cmd = _get_tool_input(call).get("command", "")
            for fname in ["USER_PREFS", "SORCAR"]:
                if fname in cmd and any(w in cmd for w in ["cat ", "head ", "tail "]):
                    raise AssertionError(
                        f"Agent used Bash to read {fname}: {cmd[:120]}"
                    )


# ---------------------------------------------------------------------------
# Test 4: Read every file before modifying it (via Read tool, not cat)
# SYSTEM.md: "You MUST call Read(file_path) on every file BEFORE Edit()"
# DB evidence: 7/60 tasks edited without prior Read (12%)
# ---------------------------------------------------------------------------

def test_read_before_edit() -> None:
    """Agent must Read a file before calling Edit on it."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project\n")
    target = os.path.join(work_dir, "greet.py")
    with open(target, "w") as f:
        f.write("def greet():\n    return 'hello'\n")

    _, calls = _run_agent(
        task=(
            f"Change the return value in {target} from 'hello' to 'goodbye'. "
            "Use Edit() to make the change."
        ),
        work_dir=work_dir,
    )

    read_files: set[str] = set()
    for c in calls:
        name = _get_tool_name(c)
        inp = _get_tool_input(c)
        fp = inp.get("file_path", "")
        if name == "Read" and fp:
            read_files.add(fp)
        if name == "Edit" and fp:
            assert fp in read_files, (
                f"Edit called on {fp} without prior Read. "
                f"Files read so far: {read_files}"
            )


# ---------------------------------------------------------------------------
# Test 5: Pre-finish lint/typecheck for coding tasks
# SYSTEM.md: "If you created or modified ANY .py file: run uv run check --full"
# DB evidence: 70% of code tasks skip lint (7/10)
# ---------------------------------------------------------------------------

def test_lint_check_before_finish() -> None:
    """Agent must run `uv run check` before finishing when code was modified."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project\n")
    target = os.path.join(work_dir, "math_utils.py")
    with open(target, "w") as f:
        f.write("def add(a: int, b: int) -> int:\n    return a + b\n")

    _, calls = _run_agent(
        task=(
            f"Add a multiply(a: int, b: int) -> int function to {target}."
        ),
        work_dir=work_dir,
        max_steps=15,
    )

    # Check any Bash call with "uv run check" or "check --full"
    bash_commands = [
        _get_tool_input(c).get("command", "")
        for c in calls
        if _get_tool_name(c) == "Bash"
    ]
    has_check = any(
        "uv run check" in cmd or "check --full" in cmd
        for cmd in bash_commands
    )
    assert has_check, (
        "Agent did not run lint/typecheck before finishing. "
        f"Bash commands: {bash_commands}"
    )


# ---------------------------------------------------------------------------
# Test 6: visibility_constraint — full answer in finish(summary=...)
# SYSTEM.md: "Compose the full detailed answer directly inside the summary"
# ---------------------------------------------------------------------------

def test_visibility_constraint_full_answer() -> None:
    """For informational questions, the finish summary must contain the full answer."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project\n")

    result, _ = _run_agent(
        task="What is the Fibonacci sequence? Give the first 10 numbers.",
        work_dir=work_dir,
        max_steps=5,
        max_budget=0.5,
    )

    summary = result.get("summary", "")
    # First 10: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    assert "34" in summary or "21" in summary, (
        f"Summary doesn't contain Fibonacci numbers (expected 34 or 21). "
        f"Summary: {summary[:300]}"
    )
    assert len(summary) > 50, (
        f"Summary too short — may be a meta-description. Summary: {summary}"
    )


# ---------------------------------------------------------------------------
# Test 7: Web research must use go_to_url() not curl, and create info file
# SYSTEM.md: "You MUST use go_to_url() to visit each site. Do NOT use Bash(curl)"
# DB evidence: Task 1179 used curl to evade, task 1222 only visited 5 URLs
# ---------------------------------------------------------------------------

def test_web_research_creates_information_file() -> None:
    """Web research task must create an information-*.md file with proper header."""
    work_dir = tempfile.mkdtemp()
    prefs_path = os.path.join(work_dir, "USER_PREFS.md")
    with open(prefs_path, "w") as f:
        f.write("## Prefs\n")
    sorcar_path = os.path.join(work_dir, "SORCAR.md")
    with open(sorcar_path, "w") as f:
        f.write("## Project\n")
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

    with open(os.path.join(tmp_dir, info_files[0])) as f:
        content = f.read()
    assert "Web Research" in content, (
        f"Information file missing 'Web Research' header. Content: {content[:200]}"
    )


if __name__ == "__main__":
    import sys
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    tests = [
        ("test_user_prefs_read_is_first_tool_call", test_user_prefs_read_is_first_tool_call),
        ("test_sorcar_md_read_as_second_call", test_sorcar_md_read_as_second_call),
        ("test_no_bash_cat_for_mandatory_files", test_no_bash_cat_for_mandatory_files),
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
