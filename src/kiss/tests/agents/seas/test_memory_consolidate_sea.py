# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``/memory_consolidate`` SEA.

Every test runs the SEA as a real task: the task's model is the scripted
local server (it calls the job's tool, answers the merge requests and
finishes) and the checking model is a second local server registered as
an endpoint model, so nothing is mocked and no provider is called.  The
memory is a directory of the test's own with the offline embedder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kiss.agents.seas.memory_consolidate import memory_consolidate_sea
from kiss.agents.seas.memory_consolidate.memory_consolidate_sea import (
    MemoryConsolidateSea,
    consolidate_memory,
)
from kiss.agents.sorcar.cron_agent import PROMPT_PREAMBLE, RUN_CONTEXT_END, run_context
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.memoryfield import consolidate
from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.memoryfield.tools import MemoryTools
from kiss.core.models.model_info import calculate_cost
from kiss.tests.agents.seas.conftest import JUDGE, judge_model, prompts, run_memory_job
from kiss.tests.agents.seas.sea_contract import assert_worker_contract, system_message, tool_names
from kiss.tests.agents.sorcar.local_model_server import MODEL, ScriptedHandler, text_body

# Four pairs of near-duplicate pages; the pairs are examined in this order
# (the most similar first), which the tails are written to give.
PG = (
    "Staging Postgres runs on host db-stg-2 in the eu-west region. The primary listens on "
    "port 6432 through pgbouncer and the replica on port 6433. Backups run nightly at 02:00 "
    "UTC with pgbackrest to the bucket stg-backups and are kept for fourteen days. "
)
CI = (
    "The CI pipeline of the project runs the unit tests on four shards, then the linters, "
    "then the packaging step that builds the wheel and uploads it to the internal index. "
    "A run takes about eleven minutes and is cancelled when a newer commit arrives. "
)
DNS = (
    "The internal DNS zone corp.example is served by two CoreDNS pods behind the address "
    "10.0.0.53. Records are generated from the service catalogue every five minutes by the "
    "job dns-sync, and a change takes at most ten minutes to reach every resolver. "
)
NOTES = (
    "Release notes of the billing service. The deploy went through the canary stage in "
    "twenty minutes, the database migration ran before the rollout, and the on-call "
    "engineer watched the error budget dashboard for one hour afterwards. "
)
PAIRS = {
    "pg": (PG, "Owner: platform.", "Owner: data."),
    "proj/ci": (CI, "Maintainer: build team.", "Maintainer: release team."),
    "dns": (DNS, "The owner is the network team.", "The owner is the platform team since March."),
    "notes": (NOTES, "Version 4.1 of the third of May.", "Version 4.2 of the ninth of June."),
}

SUPPORTED = text_body('[{"n": 1, "verdict": "SUPPORTED"}, {"n": 2, "verdict": "SUPPORTED"}]')
QUESTIONS = text_body(json.dumps([
    {"q": "Which port does the staging replica use?", "a": "6433"},
    {"q": "Who owns staging Postgres?", "a": "the data team"},
    {"q": "", "a": "a question without text is dropped"},
]))
# What the judge answers for a merge that passes: questions, verdicts, no fact found missing.
CHECKS = [QUESTIONS, SUPPORTED, text_body("[]")]


def merged(stem: str) -> bytes:
    """The task model's reply that merges pair *stem*: its newer page with a title and summary."""
    base, _, newer = PAIRS[stem]
    return text_body(f"TITLE: {stem} merged\nSUMMARY: One page about {stem}.\n\n{base}{newer}")


def write_pairs(root: Path, *stems: str) -> MemoryTools:
    """Write the pages ``<stem>-a`` (older) and ``<stem>-b`` of each of *stems* under *root*."""
    tools = MemoryTools(root, domains={"proj": ""}, task_id="writer-task")
    for stem in stems:
        base, older, newer = PAIRS[stem]
        tools.memory_write(f"{stem}-a", base + older, title=stem, sources="runbook.md")
        tools.memory_write(f"{stem}-b", base + newer, title=stem)
    return tools


def page_names(root: Path) -> list[str]:
    """The pages of the general memory and of the domain memory ``proj`` under *root*."""
    return MemoryDir(root).page_names() + [
        f"proj/{name}" for name in MemoryDir(root / "proj").page_names()
    ]


def test_the_sea_follows_the_worker_contract() -> None:
    sea = MemoryConsolidateSea()
    assert [tool.__name__ for tool in sea.tools([print])] == ["print", "consolidate_memory"]
    assert sea.prompt("--apply --limit 3").startswith("--apply --limit 3\n\n")
    assert sea.prompt("") == memory_consolidate_sea.PROMPT
    assert sea.system_prompt("the assistant's prompt") == memory_consolidate_sea.SYSTEM_PROMPT
    assert_worker_contract(
        memory_consolidate_sea,
        sea,
        {"tool_profile": "none", "max_budget": 10.0, "timeout": 3600, "locked": ["tool_profile"]},
    )
    for option in ("--apply", "--list", "--restore", "--memory", "--limit", "--judge"):
        assert option in sea.description()
    # Outside a running task there is no task text to read the options from.
    assert consolidate_memory() == (
        "Error: consolidate_memory is a tool of a running /memory_consolidate task"
    )


def test_a_dry_run_reports_the_merge_it_would_make_and_writes_nothing(
    memory_root: Path, tmp_path: Path
) -> None:
    write_pairs(memory_root, "pg")
    before = {path.name: path.read_bytes() for path in memory_root.glob("*.md")}
    with judge_model(CHECKS) as judged:
        report, requests = run_memory_job(
            MemoryConsolidateSea(), f"--judge {JUDGE}", [merged("pg")], tmp_path
        )

    body = PG + "Owner: data."
    stored = sum(len(MemoryDir(memory_root).read(name).body) for name in ("pg-a", "pg-b"))
    sizes = f"{stored} to {len(body)} bytes"
    assert report == (
        "Memory consolidation, dry run: families examined: 1; would merge: 1. "
        "Nothing was written; the option --apply merges.\n\n"
        f"- `pg-a` + `pg-b` (similarity 0.98): would become `pg-b` ({sizes}; "
        "check questions answered: 2 of 2; no fact of the originals found missing).\n"
        "Proposed page (not written):\n~~~\n"
        f"TITLE: pg merged\nSUMMARY: One page about pg.\n\n{body}\n~~~\n\n"
        f"Model calls: 4, $0.00; written by {MODEL}, checked by {JUDGE}."
    )
    assert {path.name: path.read_bytes() for path in memory_root.glob("*.md")} == before
    assert not (memory_root / ".archive").exists()

    # The agent gets the job's tool and finish, and the job's own instructions.  The tool
    # takes no argument: only the task text can make the job write.
    first, last = (r for r in requests if r.get("tools"))
    assert tool_names(first) == {"consolidate_memory", "finish"}
    (job,) = (t["function"] for t in first["tools"] if t["function"]["name"] != "finish")
    assert job["parameters"]["properties"] == {}
    assert system_message(first).startswith(memory_consolidate_sea.SYSTEM_PROMPT)
    assert report in last["messages"][-1]["content"]
    # The task's model merges; it sees both pages, the older first, and the size limit.
    (merge_prompt,) = prompts(requests)
    assert merge_prompt.index("=== page pg-a") < merge_prompt.index("=== page pg-b")
    assert "Owner: platform." in merge_prompt and "At most 8192 bytes." in merge_prompt
    # The judge writes its questions from the originals alone, then reads the merged page;
    # last it reads both sides for a fact no question asked about.
    asked, checked, compared = prompts(judged)
    assert "Owner: platform." in asked and "pg merged" not in asked
    page = "title: pg merged\nsummary: One page about pg.\n\n" + body
    assert page in checked
    assert "1. Q: Which port does the staging replica use?\n   A: 6433" in checked
    assert "2. Q: Who owns staging Postgres?" in checked and "3. Q:" not in checked
    assert "Owner: platform." not in checked
    assert f"<merged>\n{page}\n</merged>" in compared
    assert compared.index("<originals>\n=== page pg-a") < compared.index("=== page pg-b")

    # A fact it finds missing stops the merge, however the questions were answered; a reply
    # that is no list stops it too, for this run.
    lost = text_body('[{"fact": "The replica lags by up to 30 s."}, {"fact": "Owner: infra."}]')
    for compare, reason in (
        (lost, "facts of the originals found missing: 2 "
               "(The replica lags by up to 30 s.; Owner: infra.)"),
        (text_body("Nothing is lost."), "the check reply could not be used"),
    ):
        with judge_model([QUESTIONS, SUPPORTED, compare]):
            report, _ = run_memory_job(
                MemoryConsolidateSea(), f"--judge {JUDGE}", [merged("pg")], tmp_path
            )
        assert f"- `pg-a` + `pg-b` (similarity 0.98): left as they are, {reason}." in report
    assert not (memory_root / ".archive").exists()  # a dry run records no refusal either


def test_apply_merges_what_passes_and_records_what_it_leaves(
    memory_root: Path, tmp_path: Path
) -> None:
    write_pairs(memory_root, "pg", "proj/ci", "dns", "notes")
    original = (memory_root / "pg-a.md").read_text(encoding="utf-8")
    written = MemoryDir(memory_root).read("pg-a").frontmatter["verified"]
    three = text_body(json.dumps([
        {"q": "Which address serves the zone corp.example?", "a": "10.0.0.53"},
        {"q": "Who owns the internal DNS?", "a": "the platform team"},
        {"q": "How often are records generated?", "a": "every five minutes"},
    ]))
    # The second answer differs, and the third is judged twice, once as absent: a verdict
    # against a fact is not undone by another one for it.
    one_kept = text_body(json.dumps([
        {"n": 1, "verdict": "SUPPORTED"}, {"n": 2, "verdict": "DIFFERENT"},
        {"n": 3, "verdict": "ABSENT"}, {"n": 3, "verdict": "SUPPORTED"},
    ]))
    sea = MemoryConsolidateSea()
    with judge_model([*CHECKS, *CHECKS, three, one_kept]) as judged:
        report, _ = run_memory_job(
            sea,
            f"Nobody can answer questions.\n--apply --judge {JUDGE}",
            [merged("pg"), merged("proj/ci"), merged("dns"), text_body("SEPARATE.")],
            tmp_path,
        )
    assert len(judged) == 8  # no check is made of pages the model says are separate records
    general_id, proj_id = re.findall(r"--restore (\S+)`", report)
    head, lines, tail = report.split("\n\n")
    assert head == "Memory consolidation: families examined: 4; merged: 2."
    checked = "check questions answered: 2 of 2; no fact of the originals found missing"
    assert lines.split("\n") == [
        f"- `pg-a` + `pg-b` (similarity 0.98): merged into `pg-b` (530 to 262 bytes; {checked}). "
        f"Undo: `/memory_consolidate --restore {general_id}`",
        "- `proj/ci-a` + `proj/ci-b` (similarity 0.97): merged into `proj/ci-b` (546 to 273 "
        f"bytes; {checked}). Undo: `/memory_consolidate --restore {proj_id}`",
        "- `dns-a` + `dns-b` (similarity 0.96): left as they are, 2 of 3 facts not kept "
        "(Who owns the internal DNS?; How often are records generated?).",
        "- `notes-a` + `notes-b` (similarity 0.91): left as they are, separate records.",
    ]
    assert tail == f"Model calls: 12, $0.00; written by {MODEL}, checked by {JUDGE}."

    # One page replaced each merged pair; the originals are in the archive, byte for byte.
    assert page_names(memory_root) == ["dns-a", "dns-b", "notes-a", "notes-b", "pg-b", "proj/ci-b"]
    page = MemoryDir(memory_root).read("pg-b")
    assert (page.title, page.summary) == ("pg merged", "One page about pg.")
    assert page.body == PG + "Owner: data.\n"
    assert page.frontmatter["writer"] == "consolidation"
    assert page.frontmatter["verified"] == written and page.frontmatter["sources"] == "runbook.md"
    assert re.fullmatch(r"[0-9a-f]{32}", page.frontmatter["task"])
    assert proj_id.startswith("proj/") and "/" not in general_id
    archived = memory_root / ".archive" / general_id
    assert sorted(path.name for path in archived.iterdir()) == ["merge.json", "pg-a.md", "pg-b.md"]
    assert (archived / "pg-a.md").read_text(encoding="utf-8") == original
    assert "pg-a" not in MemoryTools(memory_root).memory_search("staging postgres replica port")

    # What was examined and left is recorded, so the next run does not ask about it again.
    left = json.loads((memory_root / ".archive" / "left-alone.json").read_text(encoding="utf-8"))
    assert sorted(left.values()) == [
        "2 of 3 facts not kept (Who owns the internal DNS?; How often are records generated?)",
        "separate records",
    ]
    again, requests = run_memory_job(sea, f"--apply --judge {MODEL}", [], tmp_path)
    assert again == (
        "Memory consolidation: families examined: 0; merged: 0.\n\n"
        "Families not examined (left alone before: 2).\n"
        f"Model calls: 0, $0.00; written by {MODEL}, checked by the same model."
    )
    assert prompts(requests) == []


def test_a_merge_is_listed_and_can_be_undone(memory_root: Path, tmp_path: Path) -> None:
    write_pairs(memory_root, "pg", "proj/ci")
    original = (memory_root / "proj" / "ci-b.md").read_text(encoding="utf-8")
    sea = MemoryConsolidateSea()
    assert run_memory_job(sea, "--list", [], tmp_path)[0] == "No merges are recorded."
    with judge_model(CHECKS * 2):
        report, _ = run_memory_job(
            sea, f"--apply --judge {JUDGE}", [merged("pg"), merged("proj/ci")], tmp_path
        )
    general_id, proj_id = re.findall(r"--restore (\S+)`", report)
    assert run_memory_job(sea, "--list", [], tmp_path)[0] == (
        f"- `{general_id}`: pg-a, pg-b into pg-b (in place)\n"
        f"- `{proj_id}`: ci-a, ci-b into ci-b (in place)"
    )

    restored, _ = run_memory_job(sea, f"--restore {proj_id}", [], tmp_path)
    assert restored == f"Merge `{proj_id}`: restored ci-a, ci-b."
    assert page_names(memory_root) == ["pg-b", "proj/ci-a", "proj/ci-b"]
    back = MemoryDir(memory_root / "proj").read("ci-b")
    assert back.body == CI + "Maintainer: release team.\n" and back.title == "proj/ci"
    assert back.frontmatter["writer"] == "agent" and back.raw != original  # a fresh `updated`
    listed, _ = run_memory_job(sea, "--list --memory proj", [], tmp_path)
    assert re.fullmatch(rf"- `{proj_id}`: ci-a, ci-b into ci-b \(restored 20\S+Z\)", listed)

    # A second restore touches nothing, and the undone merge is not made again.
    untouched = "a page of this name exists that the merge did not write"
    assert run_memory_job(sea, f"--restore {proj_id}", [], tmp_path)[0] == (
        f"Merge `{proj_id}`: restored no page.\n"
        f"- `ci-a` not restored: {untouched}\n- `ci-b` not restored: {untouched}"
    )
    again, _ = run_memory_job(sea, f"--apply --judge {MODEL}", [], tmp_path)
    assert again.split("\n\n")[1].startswith(
        "Families not examined (a merge of these pages was undone: 1).\n"
    )


def test_a_reply_that_cannot_be_used_leaves_the_family_for_another_run(
    memory_root: Path, tmp_path: Path
) -> None:
    write_pairs(memory_root, "pg")
    sea = MemoryConsolidateSea()
    task = f"--apply --judge {JUDGE}"
    refused = text_body("I cannot.").replace(b'"stop"', b'"content_filter"')
    cut = text_body('[{"n": 1, "verd').replace(
        b'"finish_reason": "stop"', b'"finish_reason": "length"'
    )
    one = text_body('[{"q": "Which port does the replica use?", "a": "6433"}, "not a question"]')
    runs: list[tuple[bytes, list[bytes], str]] = [
        (text_body("Here is the page you asked for."), [], "the merge reply could not be used"),
        (refused, [], "the merge reply could not be used"),
        (merged("pg"), [text_body("I have no questions.")],
         "no questions could be written to check the merge"),
        (merged("pg"), [one, cut], "the check reply could not be used"),
        (merged("pg"), [QUESTIONS, text_body("[not json]")], "the check reply could not be used"),
        # A question left without a verdict, or with one that is none, is not a fact found
        # lost: nothing is recorded, and another run asks again.
        (merged("pg"), [QUESTIONS, text_body('[{"n": 1, "verdict": "SUPPORTED"}]')],
         "the check reply could not be used"),
        (merged("pg"), [QUESTIONS, text_body(
            '[{"n": 1, "verdict": "SUPPORTED"}, {"n": 2, "verdict": "PROBABLY"}]')],
         "the check reply could not be used"),
        # Nor is a reading of both sides that gives no list of facts, found or not.
        (merged("pg"), [QUESTIONS, SUPPORTED, text_body("Nothing is lost.")],
         "the check reply could not be used"),
        (merged("pg"), [QUESTIONS, SUPPORTED, text_body('["Owner: platform."]')],
         "the check reply could not be used"),
        (merged("pg"), [QUESTIONS, SUPPORTED, refused], "the check reply could not be used"),
    ]
    for merge_reply, judge_replies, reason in runs:
        with judge_model(judge_replies or [text_body("never asked")]) as judged:
            report, _ = run_memory_job(sea, task, [merge_reply], tmp_path)
        assert f"`pg-a` + `pg-b` (similarity 0.98): left as they are, {reason}." in report
        assert len(judged) == len(judge_replies)
    assert not (memory_root / ".archive").exists()

    # Reasons that hold for the pages as they are end the matter, but only a run that may write
    # records them.
    too_large = text_body("TITLE: t\nSUMMARY: s\n\n" + "x" * 9000)
    for _ in range(2):
        report, _ = run_memory_job(sea, f"--judge {MODEL}", [too_large], tmp_path)
        assert "left as they are, the merged page is too large (9004 bytes)." in report
    report, _ = run_memory_job(sea, f"--apply --judge {MODEL}", [too_large], tmp_path)
    assert "families examined: 1; merged: 0." in report
    report, requests = run_memory_job(sea, f"--apply --judge {MODEL}", [], tmp_path)
    assert "families examined: 0; merged: 0." in report and prompts(requests) == []
    assert "Families not examined (left alone before: 1)." in report

    # A fact the reading of both sides finds missing is such a reason.
    write_pairs(memory_root, "dns")
    lost = text_body('[{"fact": "A change reaches every resolver in ten minutes."}]')
    with judge_model([QUESTIONS, SUPPORTED, lost]):
        report, _ = run_memory_job(sea, task, [merged("dns")], tmp_path)
    assert (
        "`dns-a` + `dns-b` (similarity 0.96): left as they are, facts of the originals found "
        "missing: 1 (A change reaches every resolver in ten minutes.)."
    ) in report
    report, requests = run_memory_job(sea, f"--apply --judge {MODEL}", [], tmp_path)
    assert "Families not examined (left alone before: 2)." in report and prompts(requests) == []
    assert page_names(memory_root) == ["dns-a", "dns-b", "pg-a", "pg-b"]


class _AnotherTaskWrites(ScriptedHandler):
    """The judge's server: before its first reply, another task rewrites the page ``name``."""

    root: Path
    name: str

    def do_POST(self) -> None:  # noqa: N802
        if not type(self).requests:
            MemoryTools(self.root).memory_write(self.name, PG + "Owner: infra.", title="pg")
        super().do_POST()


def test_a_page_another_task_rewrote_during_the_merge_is_not_replaced(
    memory_root: Path, tmp_path: Path
) -> None:
    write_pairs(memory_root, "pg")
    sea = MemoryConsolidateSea()
    _AnotherTaskWrites.root = memory_root

    # The newest page, whose name the merged page takes, changed: nothing is merged.
    _AnotherTaskWrites.name = "pg-b"
    with judge_model(CHECKS, _AnotherTaskWrites):
        report, _ = run_memory_job(sea, f"--apply --judge {JUDGE}", [merged("pg")], tmp_path)
    assert "): not merged, Page pg-b.md changed since it was read." in report
    assert "families examined: 1; merged: 0." in report
    assert page_names(memory_root) == ["pg-a", "pg-b"]
    assert MemoryDir(memory_root).read("pg-b").body == PG + "Owner: infra.\n"
    assert consolidate.merges(MemoryDir(memory_root)) == []

    # An older page changed: the others are merged and that page stays as the task wrote it.
    _AnotherTaskWrites.name = "pg-a"
    with judge_model(CHECKS, _AnotherTaskWrites):
        report, _ = run_memory_job(sea, f"--apply --judge {JUDGE}", [merged("pg")], tmp_path)
    assert "found missing; pg-a changed meanwhile and stayed). Undo:" in report
    assert page_names(memory_root) == ["pg-a", "pg-b"]
    assert MemoryDir(memory_root).read("pg-a").body == PG + "Owner: infra.\n"
    assert MemoryDir(memory_root).read("pg-b").frontmatter["writer"] == "consolidation"


def test_the_job_stops_while_the_task_can_still_pay_for_its_report(
    memory_root: Path, tmp_path: Path
) -> None:
    write_pairs(memory_root, "pg", "dns")
    # Five calls of the task's model are paid when the first family is done (the step that
    # calls the tool and the family's four): that leaves less than a tenth of this budget.
    budget = 5 * calculate_cost(MODEL, 500, 100) / 0.95
    report, requests = run_memory_job(
        MemoryConsolidateSea(),
        f"--judge {MODEL}",
        [merged("pg"), *CHECKS],
        tmp_path,
        max_budget=budget,
    )
    assert report.startswith("Memory consolidation, dry run: families examined: 1; would merge: 1.")
    assert report.endswith(
        "~~~\n\nStopped early: the task's budget is nearly spent.\n"
        "Families waiting for another run: 1.\n"
        f"Model calls: 4, $0.00; written by {MODEL}, checked by the same model."
    )
    assert len(prompts(requests)) == 4

    # The budget is watched call by call: a family it stops in the middle of is not settled.
    report, requests = run_memory_job(
        MemoryConsolidateSea(),
        f"--apply --judge {MODEL}",
        [merged("pg"), QUESTIONS],
        tmp_path,
        max_budget=3 * calculate_cost(MODEL, 500, 100) / 0.95,
    )
    assert report == (
        "Memory consolidation: families examined: 0; merged: 0.\n\n"
        "Stopped early: the task's budget is nearly spent.\n"
        "Families waiting for another run: 2.\n"
        f"Model calls: 2, $0.00; written by {MODEL}, checked by the same model."
    )
    assert len(prompts(requests)) == 2 and not (memory_root / ".archive").exists()

    # One run at a time examines a memory.
    with exclusive_file_lock(Path(f"{memory_root}.consolidate.lock")):
        report, requests = run_memory_job(MemoryConsolidateSea(), "--apply", [], tmp_path)
    assert report == "Error: another /memory_consolidate run is examining this memory."
    assert prompts(requests) == []

    # A limit also leaves families for another run, and is no early stop.
    report, _ = run_memory_job(
        MemoryConsolidateSea(), f"--limit 1 --judge {MODEL}", [text_body("separate")], tmp_path
    )
    assert report == (
        "Memory consolidation, dry run: families examined: 1; would merge: 0. "
        "Nothing was written; the option --apply merges.\n\n"
        "- `pg-a` + `pg-b` (similarity 0.98): left as they are, separate records.\n\n"
        "Families waiting for another run: 1.\n"
        f"Model calls: 1, $0.00; written by {MODEL}, checked by the same model."
    )


def test_a_scheduled_run_takes_its_options_from_the_jobs_own_prompt(
    memory_root: Path, tmp_path: Path
) -> None:
    """A scheduled run is shown the end of the previous report, where ``--apply`` is a word."""
    write_pairs(memory_root, "pg", "dns")
    previous = (
        "Nothing was written; the option --apply merges.\n"
        f"--apply --judge no-such-model\n{RUN_CONTEXT_END}\n--apply"
    )
    job = {"id": "nightly", "schedule": "0 3 * * *", "last_status": "ok", "last_summary": previous}
    # Nor is what follows the request the SEA appends to the job's prompt.
    task = (
        f"{PROMPT_PREAMBLE}{run_context(job)}\n\n--limit 1 --judge {MODEL}\n\n"
        f"{memory_consolidate_sea.PROMPT}\n--apply"
    )
    report, requests = run_memory_job(
        MemoryConsolidateSea(), task, [merged("pg"), *CHECKS], tmp_path
    )
    assert report.startswith("Memory consolidation, dry run: families examined: 1; would merge: 1.")
    assert "Families waiting for another run: 1." in report and len(prompts(requests)) == 4
    assert page_names(memory_root) == ["dns-a", "dns-b", "pg-a", "pg-b"]
    assert not (memory_root / ".archive").exists()


@pytest.mark.parametrize(
    ("task", "error"),
    [
        ("--limit 0", "Error: --limit takes a whole number above zero, not '0'"),
        ("--limit many", "Error: --limit takes a whole number above zero, not 'many'"),
        ("--apply --limit", "Error: --limit needs a value"),
        ("--restore --apply", "Error: --restore needs a value"),
        ("--judge", "Error: --judge needs a value"),
        ("--judge no-such-model", "Error: --judge names no known model: no-such-model"),
        ("--memory nowhere", "Error: no memory 'nowhere'; memories: general, proj"),
        ("--restore nowhere/20260101T000000Z-pg-b", "Error: no memory 'nowhere'."),
        ("--restore 20260101T000000Z-pg-b", "Error: no merge '20260101T000000Z-pg-b' in "),
    ],
)
def test_an_option_that_cannot_be_read_stops_the_job(
    task: str, error: str, memory_root: Path, tmp_path: Path
) -> None:
    write_pairs(memory_root, "pg", "proj/ci")
    report, requests = run_memory_job(MemoryConsolidateSea(), task, [], tmp_path)
    assert report.startswith(error) and "\n" not in report
    assert prompts(requests) == [] and page_names(memory_root) == [
        "pg-a", "pg-b", "proj/ci-a", "proj/ci-b"]
    assert not (memory_root / ".archive").exists()


def test_a_memory_that_is_empty_or_switched_off_gives_nothing_to_do(
    memory_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sea = MemoryConsolidateSea()
    assert not memory_root.exists()
    report, _ = run_memory_job(sea, f"--apply --judge {MODEL}", [], tmp_path)
    assert report == (
        "Memory consolidation: families examined: 0; merged: 0.\n\n"
        f"Model calls: 0, $0.00; written by {MODEL}, checked by the same model."
    )
    monkeypatch.setenv("KISS_USE_MEMORY", "0")
    report, _ = run_memory_job(sea, "--apply", [], tmp_path)
    assert report == "Memory is switched off; there is nothing to consolidate."
