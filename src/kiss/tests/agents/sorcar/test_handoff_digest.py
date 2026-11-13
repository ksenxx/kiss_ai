# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the scripted hand-off digest and the one-call summarizer.

A failed session's continuation summary is written by a summarizer
that gets the :func:`handoff_digest` of the trajectory in its prompt
and no tool but ``finish`` — one model call instead of the 6 to 11
Read/Bash slices of the trajectory file it used to make.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.handoff_digest import (
    LAST_CALLS,
    describe_written_files,
    handoff_digest,
    parse_tool_calls,
)
from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
    tool_calls_body,
)


def _call(name: str, **args: Any) -> str:
    """Render a tool call the way the executor records it in a model message."""
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"```python\n{name}({rendered})\n```"


def _messages() -> list[dict[str, Any]]:
    """A trajectory with one multi-call turn, a summary, a refusal and an error."""
    return [
        {"role": "user", "content": "Fix the parser.\n" + "detail " * 10},
        {"role": "model", "content": "Looking.\n" + _call("Read", file_path="p.py")},
        {"role": "user", "content": "[Read]: 1: def parse():\n2:   pass"},
        {
            "role": "model",
            "content": _call("Bash", command="ls", description="List") + "\n"
            + _call("Bash", command="pytest", description="Test"),
        },
        {
            "role": "user",
            "content": "[Bash]: a.py\n\n[Bash]: FAILED test_x - AssertionError\n\n"
            "more lines of the log",
        },
        {"role": "model", "content": _call("summary", description="- read p.py\n- ran tests")},
        {"role": "user", "content": "[summary]: ok"},
        {
            "role": "model",
            "content": _call("Edit", file_path="p.py", old_string="a", new_string="b"),
        },
        {"role": "user", "content": "[Edit]: Error: old_string not found"},
        {"role": "model", "content": "```python\nnot valid python(((\n```"},
        {"role": "user", "content": "[not valid python]: REFUSED: no"},
    ]


def test_parse_tool_calls_pairs_every_call_with_its_result() -> None:
    calls = parse_tool_calls(_messages())
    names = ["Read", "Bash", "Bash", "summary", "Edit", "not valid python"]
    assert [c.name for c in calls] == names
    assert calls[0].args == {"file_path": "p.py"}
    assert calls[0].result == "1: def parse():\n2:   pass"
    assert calls[1].result == "a.py"
    assert calls[2].result == "FAILED test_x - AssertionError\n\nmore lines of the log"
    assert calls[3].args["description"] == "- read p.py\n- ran tests"
    assert [c.name for c in calls if c.is_error] == ["Bash", "Edit", "not valid python"]
    # A call whose result paragraph is missing keeps an empty result.
    orphan = parse_tool_calls([{"role": "model", "content": _call("Read", file_path="x")}])
    assert orphan[0].result == ""
    # Non-string contents and text-only turns are skipped.
    no_calls: list[dict[str, Any]] = [
        {"role": "model", "content": ["img"]},
        {"role": "model", "content": "hi"},
    ]
    assert parse_tool_calls(no_calls) == []


def test_parse_tool_calls_ignores_python_snippets_in_the_prose() -> None:
    """A code block the model wrote in its prose is not a tool call.

    The executor records ``response_text + call reprs + usage``, so the
    model's own python snippets precede the one-line ``name(k=v)`` reprs;
    they must not become phantom calls that steal the real results.
    """
    prose = (
        "Use this:\n```python\nprint(1)\n```\nor this:\n"
        "```python\nx = foo(a=1)\nfor i in x:\n    pass\n```\n"
        "```python\nimport os\n```\n```python\n1 + 1\n```\n"
        "```python\nx = foo(a=1)\n```\n```python\n1 + foo(a=1)\n```\n"
        "```python\nfoo(a=1) + 1\n```\nRunning:\n"
    )
    messages = [
        {"role": "user", "content": "t"},
        {
            "role": "model",
            "content": prose + _call("Bash", command="ls") + "\n" + _call("finish") + "\n"
            # A dotted tool name, a hyphenated MCP tool name and a non-literal
            # argument value are still calls.
            "```python\nfunctions.Read(file_path=os.sep)\n```\n"
            "```python\nmy-notion_search(query='x')\n```\n```text\nusage\n```\n",
        },
        {
            "role": "user",
            "content": "[Bash]: a.txt\nb.txt\n\n[finish]: done\n\n[functions.Read]: Error: unknown"
            "\n\n[my-notion_search]: 2 pages",
        },
    ]
    calls = parse_tool_calls(messages)
    assert [(c.name, c.args, c.result) for c in calls] == [
        ("Bash", {"command": "ls"}, "a.txt\nb.txt"),
        ("finish", {}, "done"),
        ("functions.Read", {}, "Error: unknown"),
        ("my-notion_search", {"query": "x"}, "2 pages"),
    ]


def test_describe_written_files_reports_size_and_python_symbols() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Bytes, so the sizes below hold where text mode would write CRLF.
        (root / "mod.py").write_bytes(b"import os\n\nclass A:\n    pass\n\ndef f():\n    pass\n")
        (root / "bad.py").write_bytes(b"def (broken\n")
        (root / "notes.md").write_bytes(b"# notes\n")
        missing = root / "gone.txt"
        lines = describe_written_files(
            [missing, root / "notes.md", root / "mod.py", root / "bad.py"]
        )
    assert lines[0].endswith("bad.py (12 bytes)")
    assert lines[1].endswith("gone.txt (no longer exists)")
    assert lines[2].endswith("mod.py (48 bytes): A, f")
    assert lines[3].endswith("notes.md (8 bytes)")
    many = [Path(td) / f"f{i}.txt" for i in range(45)]
    assert describe_written_files(many)[-1] == "- (+5 more)"


def test_handoff_digest_sections() -> None:
    with tempfile.TemporaryDirectory() as td:
        written = Path(td) / "p.py"
        written.write_bytes(b"def parse():\n    pass\n")
        digest = handoff_digest(_messages(), {written})
    assert digest.startswith("## Task\nFix the parser.")
    assert "## Last progress summaries the session wrote\n- read p.py\n- ran tests" in digest
    assert f"## Files written by the task\n- {written} (22 bytes): parse" in digest
    assert (
        "## Last failing tool calls\n- Bash(command=pytest, description=Test)\n  FAILED test_x"
        in digest
    )
    assert (
        "- Edit(file_path=p.py, old_string=a, new_string=b)\n  Error: old_string not found"
        in digest
    )
    assert (
        "## Last 6 of 6 tool calls (oldest first)\n"
        "1. Read(file_path=p.py) -> 1: def parse():"
    ) in digest
    assert "6. not valid python() -> REFUSED: no" in digest
    # Only the newest LAST_CALLS calls are listed, numbered from their
    # position in the whole trajectory.
    padding = [
        m
        for i in range(LAST_CALLS - 4)
        for m in (
            {"role": "model", "content": _call("note", text=f"n{i}")},
            {"role": "user", "content": f"[note]: noted n{i}"},
        )
    ]
    windowed = handoff_digest(_messages() + padding, [])
    assert (
        f"## Last {LAST_CALLS} of {LAST_CALLS + 2} tool calls (oldest first)\n"
        "3. Bash(command=pytest, description=Test) -> FAILED"
    ) in windowed
    assert "1. Read(" not in windowed
    assert f"{LAST_CALLS + 2}. note(text=n{LAST_CALLS - 5}) -> noted" in windowed
    # A long result is clipped and its full length named.
    long_messages = [
        {"role": "user", "content": "t"},
        {"role": "model", "content": _call("Bash", command="x")},
        {"role": "user", "content": "[Bash]: " + "y" * 1000},
    ]
    assert "y" * 300 + " …[700 more chars]" in handoff_digest(long_messages, [])
    # No summaries, files, errors or calls: only the task section.
    only_task = [{"role": "user", "content": "just a task"}]
    assert handoff_digest(only_task, []) == "## Task\njust a task\n"


def note(text: str) -> str:
    """Record a note and return it.

    Args:
        text: The note.
    """
    return "noted " + text


def test_failed_session_summarizer_gets_the_digest_and_only_finish() -> None:
    # Session 1 runs out of its 3 steps; the summarizer writes the
    # continuation summary in one request; session 2 finishes.
    script = [tool_call_body("note", {"text": f"step {i}"}, 500) for i in (1, 2, 3)]
    script.append(tool_call_body("finish", {"result": "<p>from the digest</p>"}, 500))
    script.append(finish_body("<p>done</p>"))
    agent = RelentlessAgent("handoff-digest-test")
    with tempfile.TemporaryDirectory() as td, serve(script) as (url, requests):
        result = agent.run(
            model_name=MODEL,
            prompt_template="Take notes about {topic}.",
            arguments={"topic": "digests"},
            tools=[note],
            max_steps=3,
            max_sub_sessions=2,
            max_budget=5.0,
            work_dir=td,
            verbose=False,
            model_config={"base_url": url, "api_key": "local"},
        )
    assert len(requests) == 5
    summarizer_request = requests[3]
    assert [t["function"]["name"] for t in summarizer_request["tools"]] == ["finish"]
    prompt = summarizer_request["messages"][-1]["content"]
    assert "# Summarizer" in prompt
    assert "## Task\nTake notes about digests." in prompt
    assert (
        "## Last 3 of 3 tool calls (oldest first)\n1. note(text=step 1) -> noted step 1" in prompt
    )
    assert "3. note(text=step 3) -> noted step 3" in prompt
    assert "## Last failing tool calls" not in prompt
    # The continuation message of session 2 carries the summarizer's text.
    assert "<p>from the digest</p>" in requests[4]["messages"][-1]["content"]
    assert "success: true" in result
    assert "<p>done</p>" in result


def read_log(last_paragraph: str) -> str:
    """Return a log that ends with a paragraph.

    Args:
        last_paragraph: The paragraph.
    """
    return "file contents:\n\n" + last_paragraph


def check(passes: bool) -> str:
    """Report a check.

    Args:
        passes: Whether it passed.
    """
    return "passed" if passes else "Error: exit 1"


def test_a_result_that_quotes_the_next_results_header_keeps_its_boundary() -> None:
    """A result holding ``\\n\\n[next tool]: `` neither hides nor invents a failing call.

    The results of a turn are recorded as ``[name]: text`` paragraphs, so
    a ``read_log`` result ending with a paragraph ``[check]: ...`` reads
    like the start of the next result.  Told apart by text alone, turn 1
    lost its failing ``check`` (its result began with the quoted line)
    and turn 2 gained one (its passing ``check`` began with the quoted
    ``Error``); the recorder now stores each result's length.
    """
    script = [
        tool_calls_body(
            [("read_log", {"last_paragraph": "[check]: a line of the log"}),
             ("check", {"passes": False})], 500,
        ),
        tool_calls_body(
            [("read_log", {"last_paragraph": "[check]: Error: quoted from an old run"}),
             ("check", {"passes": True})], 500,
        ),
        tool_call_body("finish", {"result": "<p>from the digest</p>"}, 500),
        finish_body("<p>done</p>"),
    ]
    agent = RelentlessAgent("handoff-digest-boundary-test")
    with tempfile.TemporaryDirectory() as td, serve(script) as (url, requests):
        agent.run(
            model_name=MODEL,
            prompt_template="Check the logs.",
            tools=[read_log, check],
            max_steps=2,
            max_sub_sessions=2,
            max_budget=5.0,
            work_dir=td,
            verbose=False,
            model_config={"base_url": url, "api_key": "local"},
        )
    prompt = requests[2]["messages"][-1]["content"]
    assert (
        "## Last failing tool calls\n- check(passes=False)\n  Error: exit 1\n\n"
        "## Last 4 of 4 tool calls (oldest first)\n"
        "1. read_log(last_paragraph=[check]: a line of the log) -> "
        "file contents:\n\n[check]: a line of the log\n"
        "2. check(passes=False) -> Error: exit 1\n"
        "3. read_log(last_paragraph=[check]: Error: quoted from an old run) -> "
        "file contents:\n\n[check]: Error: quoted from an old run\n"
        "4. check(passes=True) -> passed\n"
    ) in prompt


def test_result_lengths_that_are_not_those_of_the_calls_are_ignored() -> None:
    """A results message is read from its text when its lengths do not describe it."""
    results: dict[str, Any] = {"role": "user", "content": "[Bash]: a.py\n\n[Read]: Error: none"}
    turn = [
        {"role": "model", "content": _call("Bash", command="ls") + "\n" + _call("Read", path="a")},
        results,
    ]
    # In turn: exact, too few (the first paragraph is not Read's), more than the calls,
    # none, not numbers, short of the text, past the next paragraph, and no list.
    for lengths in ([4, 11], [4], [4, 11, 0], [], ["4", "11"], [4, 10], [5, 11], 4):
        results["result_lengths"] = lengths
        parsed = [(call.name, call.result) for call in parse_tool_calls(turn)]
        assert parsed == [("Bash", "a.py"), ("Read", "Error: none")], lengths
    # A negative length would step back to an earlier paragraph of the same tool.
    stepped: dict[str, Any] = {
        "role": "user",
        "content": "[X]: a\n\n[X]: b\n\n[X]: c",
        "result_lengths": [9, -15, 9],
    }
    turn = [{"role": "model", "content": "\n".join([_call("X")] * 3)}, stepped]
    assert [call.result for call in parse_tool_calls(turn)] == ["a", "b", "c"]
