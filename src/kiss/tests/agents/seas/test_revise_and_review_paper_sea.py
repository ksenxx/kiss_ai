# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``/revise_and_review_paper`` SEA.

The tests exercise the real module: the getters the SEA contract reads,
the slash-command resolution, the task builders as the fan-out guard
classifies them, ``loop_status`` on review files written to disk, and a
``ChatSorcarAgent`` running with the SEA's configuration against the
scripted local chat-completions server.  No mocks: the tool results the
scripted model receives come from the real tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas.review_paper import review_paper_sea
from kiss.agents.seas.revise_and_review_paper import revise_and_review_paper_sea as sea
from kiss.agents.seas.write_paper import write_paper_sea
from kiss.agents.sorcar import fanout_guard, sea_commands
from kiss.agents.sorcar.agent_dispatch import DEFAULT_DISPATCH_TIMEOUT_SECONDS, resolve_timeout
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_settings import resolve_settings
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(sea.__file__).resolve()

_REVIEW_HEAD = """\
Summary
The paper presents Hydra-KV, a log-structured store.

Strengths
Clear design.

Weaknesses
No YCSB numbers.

Detailed review
Section 4 lacks a baseline.
"""


def _review(tmp_path: Path, name: str, tail: str) -> Path:
    """Write a review with the shared head and *tail* as its score block; return its path."""
    path = tmp_path / name
    path.write_text(_REVIEW_HEAD + tail, encoding="utf-8")
    return path


def test_sea_getters_follow_the_user_contract() -> None:
    """The SEA appends the coordinator rules, offers three tools, neither browses nor fans out."""
    prompt = sea.add_to_system_prompt()
    assert prompt == sea.SYSTEM_PROMPT
    assert sea.WRITE_PAPER_SEA in prompt and sea.REVIEW_PAPER_SEA in prompt
    assert Path(sea.WRITE_PAPER_SEA) == Path(write_paper_sea.__file__).resolve()
    assert Path(sea.REVIEW_PAPER_SEA) == Path(review_paper_sea.__file__).resolve()
    assert f'timeout="{sea.WRITER_TIMEOUT_SECONDS}"' in prompt
    assert f'timeout="{sea.REVIEWER_TIMEOUT_SECONDS}"' in prompt
    assert 'use_memory="false"' in prompt and "Never pass `chat_id`" in prompt
    assert "`writer_task(" in prompt and "`reviewer_task(" in prompt and "`loop_status(" in prompt
    assert "ask the user once with `ask_user_question`" in prompt
    names = [t.__name__ for t in sea.add_to_tools()]
    assert names == ["writer_task", "reviewer_task", "loop_status"]
    assert sea.settings() == {
        "use_web_tools": False,
        "allow_fan_out": False,
        "auto_classify": False,
        "tool_profile": "full",
        "timeout": sea.DISPATCH_TIMEOUT_SECONDS,
    }
    assert sea.DISPATCH_TIMEOUT_SECONDS == 86400
    # No preset named, so the resolved settings are these five keys under
    # the default ``session`` preset; the deprecated getters are gone.
    assert resolve_settings(vars(sea)) == {"kind": "session", **sea.settings()}
    for legacy in (
        "system_prompt", "use_web_tools", "allow_fan_out", "auto_classify", "tool_profile",
        "dispatch_timeout", "append_to_system_prompt",
    ):
        assert not hasattr(sea, legacy), legacy
    assert "strong accept" in sea.description()


def test_slash_command_resolves_to_the_bundled_sea() -> None:
    """``/revise_and_review_paper <text>`` resolves to the text and this file.

    The daemon runs the SEA directly on the text; a ``run_agent``
    dispatch of the same file waits ``settings()["timeout"]`` (a day)
    when the caller names no timeout.
    """
    assert sea_commands.get_command("revise_and_review_paper") == _SEA_PATH
    hit = sea_commands.slash_command_task(
        "/revise_and_review_paper Writing: a paper. Review: for ICLR."
    )
    assert hit is not None
    task_text, path = hit
    assert path == _SEA_PATH
    assert task_text == "Writing: a paper. Review: for ICLR."
    settings = sea_commands.sea_settings(path)
    assert settings == {"kind": "session", **sea.settings()}
    assert resolve_timeout("", settings) == 86400.0
    assert sea_commands.sea_description(_SEA_PATH) == sea.description()


def test_dispatch_timeout_comes_from_the_settings_of_long_running_seas(tmp_path: Path) -> None:
    """``resolve_timeout`` takes ``settings()["timeout"]`` when positive, else the default.

    The paper SEAs declare their own waits; a SEA without ``timeout``
    (the bundled ``/dummy``) gets :data:`DEFAULT_DISPATCH_TIMEOUT_SECONDS`.
    Real SEA files whose ``settings()`` returns a non-numeric, boolean,
    infinite or non-positive ``timeout``, or raises, fail loudly at ``sea_settings`` (a
    broken script must not run with guessed parameters); a non-positive
    value falls back to the default and a positive float is kept as is.
    """
    for command, seconds in [("write_paper", 21600.0), ("review_paper", 7200.0)]:
        path = sea_commands.get_command(command)
        assert path is not None
        assert resolve_timeout("", sea_commands.sea_settings(path)) == seconds, command
    dummy = sea_commands.get_command("dummy")
    assert dummy is not None
    dummy_settings = sea_commands.sea_settings(dummy)
    assert "timeout" not in dummy_settings
    assert resolve_timeout("", dummy_settings) == DEFAULT_DISPATCH_TIMEOUT_SECONDS == 3600.0

    def _script(n: int, body: str) -> Path:
        folder = tmp_path / f"t{n}"
        folder.mkdir()
        script = folder / f"t{n}_sea.py"
        script.write_text(f"def description():\n    return 'x'\n{body}", encoding="utf-8")
        return script

    resolved = {
        "": 3600.0,
        "def settings():\n    return {'timeout': 1800.0}\n": 1800.0,
        "def settings():\n    return {'timeout': 1800}\n": 1800.0,
        # A removed legacy getter is an ordinary module function: ignored.
        "def dispatch_timeout():\n    return 1800.0\n": 3600.0,
    }
    for n, (body, expected) in enumerate(resolved.items()):
        settings = sea_commands.sea_settings(_script(n, body))
        assert resolve_timeout("", settings) == expected, body
        # An explicit positive argument always wins over the script.
        assert resolve_timeout("42", settings) == 42.0, body
    broken = [
        "def settings():\n    return {'timeout': 'soon'}\n",
        "def settings():\n    return {'timeout': True}\n",
        "def settings():\n    return {'timeout': float('inf')}\n",
        # A non-positive wait means nothing: refused like an infinite one.
        "def settings():\n    return {'timeout': 0}\n",
        "def settings():\n    return {'timeout': -5}\n",
        "def settings():\n    raise RuntimeError('broken')\n",
    ]
    for n, body in enumerate(broken, start=len(resolved)):
        with pytest.raises(sea_commands.SeaScriptError):
            sea_commands.sea_settings(_script(n, body))


def test_writer_task_first_round_and_revision_rounds() -> None:
    """Round 1 writes from the instructions; later rounds answer the review and carry notes."""
    first = sea.writer_task("  Write for NeurIPS 2027 at /w/p/p.tex.\n", 1)
    assert first.startswith("You ARE the paper-writing agent. Writing round 1:")
    assert "Writing instructions from the user:\nWrite for NeurIPS 2027 at /w/p/p.tex.\n" in first
    assert "Notes from the loop coordinator" not in first
    assert "review at" not in first
    assert first.endswith("report what is left undone.")
    assert "Do not run a review of your own" in first
    assert "AI-discovery process" in first and "ablation" in first

    later = sea.writer_task(
        "Write for NeurIPS 2027.", 3, "/w/reports/p-round2-review.txt", " YCSB allowed. \n"
    )
    assert later.startswith("You ARE the paper-writing agent. Revision round 3: edit the paper")
    assert "answer the review at /w/reports/p-round2-review.txt" in later
    assert "Notes from the loop coordinator:\nYCSB allowed.\n" in later
    assert later.index("Writing instructions") < later.index("Notes from the loop")
    assert later.index("Notes from the loop") < later.index("Rules for this round:")
    # A writer task must keep the full toolset even under a reviewer parent: the fan-out
    # guard reads "edit" and "build" as implementation verbs.
    assert fanout_guard.is_implementation_task(first)
    assert fanout_guard.is_implementation_task(later)


def test_reviewer_task_is_fresh_and_keeps_the_write_tool() -> None:
    """The reviewer task names the staged copy only, forbids earlier notes, asks for the verdict."""
    text = sea.reviewer_task(
        " Review for ICLR 2027, cutoff June 2026. ",
        "/w/tmp/revise_review/round-2/p.pdf",
        "/w/reports/p-round2-review.txt",
        2,
        word_limit=800,
    )
    assert text.startswith("You ARE the paper review agent. Review the paper at /w/tmp/")
    assert "write the review to /w/reports/p-round2-review.txt (word_limit=800)" in text
    assert "Review instructions from the user:\nReview for ICLR 2027, cutoff June 2026.\n" in text
    assert "round 2 of a revision cycle; do not mention rounds" in text
    assert "Do not open any other file of the repository, any earlier review" in text
    assert "under /w/tmp/revise_review/round-2/notes/" in text
    assert "do not ask the user questions" in text
    assert text.endswith('"strong accept" only for a paper you would argue for at the meeting.')
    assert "Recommendation: <" + " | ".join(sea.VERDICTS) + ">" in text
    default = sea.reviewer_task("Review for ICLR.", "/w/r/1/p.pdf", "/w/reports/r1.txt", 1)
    assert f"(word_limit={sea.DEFAULT_WORD_LIMIT})" in default
    # The text reads as a review task, and its "write the review to" keeps the child on
    # the full tool profile so it can write the review file.
    assert fanout_guard.is_review_task(text)
    assert fanout_guard.is_implementation_task(text)


def test_loop_status_sorts_a_glob_by_the_round_suffix_and_reads_scores(tmp_path: Path) -> None:
    """A glob is ordered by the ``round<N>-review`` suffix (10 after 2, whatever the name has)."""
    _review(
        tmp_path, "p-round1-review.txt", "Rating: 5\nConfidence: 4\nRecommendation: Weak Reject\n"
    )
    _review(
        tmp_path,
        "p-round2-study-round2-review.txt",
        "Rating: 6\nConfidence: 4\nRecommendation: borderline accept\n",
    )
    _review(
        tmp_path,
        "p-round2-study-round10-review.txt",
        "Overall: 4 / 5\nConfidence: 5\nRecommendation: **Accept** (8) after the fixes\n",
    )
    report = sea.loop_status(str(tmp_path / "p-round*-review.txt"), max_rounds=6)
    assert report.splitlines() == [
        "round 1: weak reject (rank 2/6); scores: Rating: 5, Confidence: 4",
        "round 2: borderline (rank 3/6); scores: Rating: 6, Confidence: 4",
        "round 3: accept (rank 5/6); scores: Overall: 4, Confidence: 5",
        "CONTINUE: revise the paper for round 4.",
    ]


def test_loop_status_keeps_the_order_of_a_path_list_and_stops_at_strong_accept(
    tmp_path: Path,
) -> None:
    """Paths separated by commas or newlines keep the given order; strong accept ends the loop."""
    first = _review(tmp_path, "z-first.txt", "Rating: 6\nRecommendation: weak accept\n")
    second = _review(tmp_path, "a-second.txt", "Rating: 10\nRecommendation: Strong accept.\n")
    report = sea.loop_status(f"{first},\n{second}\n")
    assert report.splitlines() == [
        "round 1: weak accept (rank 4/6); scores: Rating: 6",
        "round 2: strong accept (rank 6/6); scores: Rating: 10",
        "STOP: the review says strong accept; the target is reached.",
    ]


def test_loop_status_stops_after_two_rounds_without_progress(tmp_path: Path) -> None:
    """Two rounds in a row that beat no earlier round on verdict, then scores, end the loop."""
    tails = [
        "Recommendation: weak reject\n",
        "Recommendation: accept\n",
        "Recommendation: accept\n",
    ]
    for n, tail in enumerate(tails, 1):
        _review(tmp_path, f"p-round{n}-review.txt", tail)
    pattern = str(tmp_path / "p-round*-review.txt")
    assert sea.loop_status(pattern).endswith("CONTINUE: revise the paper for round 4.")
    _review(tmp_path, "p-round4-review.txt", "Recommendation: weak accept\n")
    four = sea.loop_status(pattern)
    assert "STOP: the last two rounds raised neither the recommendation nor the scores" in four
    assert four.endswith("Report the best round.")


def test_loop_status_counts_rising_scores_as_progress_and_ignores_confidence(
    tmp_path: Path,
) -> None:
    """Same verdict with higher paper scores is progress; a higher confidence alone is not."""
    for n, (rating, confidence) in enumerate([(6, 2), (7, 2), (8, 2)], 1):
        tail = f"Rating: {rating}\nConfidence: {confidence}\nRecommendation: accept\n"
        _review(tmp_path, f"p-round{n}-review.txt", tail)
    pattern = str(tmp_path / "p-round*-review.txt")
    assert sea.loop_status(pattern).endswith("CONTINUE: revise the paper for round 4.")
    _review(tmp_path, "p-round4-review.txt", "Rating: 8\nConfidence: 5\nRecommendation: accept\n")
    _review(tmp_path, "p-round5-review.txt", "Rating: 8\nConfidence: 5\nRecommendation: accept\n")
    assert "STOP: the last two rounds raised neither" in sea.loop_status(pattern)


def test_loop_status_stops_at_the_round_cap(tmp_path: Path) -> None:
    """The round cap stops the loop even while the verdict is still rising."""
    _review(tmp_path, "p-round1-review.txt", "Recommendation: reject\n")
    _review(tmp_path, "p-round2-review.txt", "Recommendation: weak accept\n")
    report = sea.loop_status(str(tmp_path / "p-round*-review.txt"), max_rounds=2)
    assert report.endswith("STOP: 2 rounds reached the cap of 2.")


def test_loop_status_asks_for_an_override_when_a_verdict_is_unreadable(tmp_path: Path) -> None:
    """No recommendation, a negated one and a look-alike word give CHECK lines and no decision."""
    _review(tmp_path, "p-round1-review.txt", "Rating: 8\nConfidence: 3\n")
    _review(
        tmp_path, "p-round2-review.txt", "Recommendation: not strong accept; reject\nRating: 7\n"
    )
    _review(tmp_path, "p-round3-review.txt", "Recommendation: unacceptable\n")
    _review(tmp_path, "p-round4-review.txt", "Recommendation: strong accept\n")
    pattern = str(tmp_path / "p-round*-review.txt")
    lines = sea.loop_status(pattern).splitlines()
    assert lines[0] == "round 1: no readable Recommendation line; scores: Rating: 8, Confidence: 3"
    assert lines[1] == "round 2: no readable Recommendation line; scores: Rating: 7"
    assert lines[2] == "round 3: no readable Recommendation line; scores: none found"
    assert lines[3] == "round 4: strong accept (rank 6/6); scores: none found"
    assert lines[4].startswith("CHECK: round 1, 2, 3 has no readable Recommendation line.")
    assert lines[4].endswith('call loop_status again with overrides="1=<verdict>".')
    # With the coordinator's readings the rounds count and the target is reached.
    forced = sea.loop_status(pattern, overrides="1=accept, 2 = Reject,3=weak reject")
    assert forced.splitlines()[0] == (
        "round 1: accept (rank 5/6) (verdict from overrides); scores: Rating: 8, Confidence: 3"
    )
    assert (
        forced.splitlines()[2]
        == "round 3: weak reject (rank 2/6) (verdict from overrides); scores: none found"
    )
    assert forced.endswith("STOP: the review says strong accept; the target is reached.")
    # Overridden rounds count towards the cap like any other round.
    three = ",".join(str(tmp_path / f"p-round{n}-review.txt") for n in range(1, 4))
    assert sea.loop_status(three, max_rounds=3, overrides="1=reject,2=reject,3=reject").endswith(
        "STOP: 3 rounds reached the cap of 3."
    )
    assert sea.loop_status(pattern, overrides="2=maybe") == (
        "error: override '2=maybe' is not <round>=<verdict> with a verdict among "
        + ", ".join(sea.VERDICTS)
    )
    assert sea.loop_status(pattern, overrides="two=accept").startswith(
        "error: override 'two=accept'"
    )


def test_loop_status_reports_missing_files_and_empty_input(tmp_path: Path) -> None:
    """A path that is not a file is reported at once; no matching files is a message, no crash."""
    present = _review(tmp_path, "p-round1-review.txt", "Recommendation: accept\n")
    assert sea.loop_status(str(present)).endswith("CONTINUE: revise the paper for round 2.")
    missing = tmp_path / "p-round2-review.txt"
    assert sea.loop_status(f"{present}\n{missing}") == (
        f"round 2: no such file: {missing}; fix the path and call loop_status again"
    )
    assert sea.loop_status(str(tmp_path / "nothing-*.txt")).startswith("no review files match")
    assert sea.loop_status("  ").startswith("no review files match")


def test_agent_gets_the_rules_and_the_tools_and_the_real_results(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees the three tools, ``run_agent`` and real results.

    The scripted model builds a writer task, reads the loop status of one
    review on disk, and finishes.  The test checks the offered tools (no
    browser, no ``run_parallel``), that the default system prompt was kept
    and the coordinator rules appended, and that both tool results flowed
    back through the tool-result messages.
    """
    review = _review(tmp_path, "p-round1-review.txt", "Rating: 10\nRecommendation: strong accept\n")
    script = [
        tool_call_body(
            "writer_task",
            {"instructions": "Write for ICLR 2027 at p.tex.", "round_number": 1},
            prompt_tokens=500,
        ),
        tool_call_body(
            "loop_status", {"reviews": str(tmp_path / "p-round*-review.txt")}, prompt_tokens=600
        ),
        finish_body("<pre>STOP: strong accept after round 1</pre>", prompt_tokens=700),
    ]
    settings = sea.settings()
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("revise-review-sea-test")
        result = agent.run(
            prompt_template="Writing: Write for ICLR 2027 at p.tex. Review: for ICLR 2027.",
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=5,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            system_prompt=sea.add_to_system_prompt(),
            tools=sea.add_to_tools(),
            tool_profile=settings["tool_profile"],
            web_tools=settings["use_web_tools"],
            is_parallel=settings["allow_fan_out"],
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert "strong accept after round 1" in parsed["summary"]

    agentic: list[dict[str, Any]] = [r for r in requests if r.get("tools")]
    assert len(agentic) == 3, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert {
            "writer_task",
            "reviewer_task",
            "loop_status",
            "run_agent",
            "Bash",
            "finish",
        } <= names
        assert not names & {"go_to_url", "run_parallel"}, names
        system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
        assert not system.startswith(sea.SYSTEM_PROMPT)
        assert "# Revise-and-review loop coordinator" in system
        assert sea.REVIEW_PAPER_SEA in system
    tool_results = [m for m in agentic[2]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 2, agentic[2]["messages"]
    assert str(tool_results[0]["content"]).startswith(
        "You ARE the paper-writing agent. Writing round 1:"
    )
    assert "round 1: strong accept (rank 6/6); scores: Rating: 10" in str(
        tool_results[1]["content"]
    )
    assert review.exists()
