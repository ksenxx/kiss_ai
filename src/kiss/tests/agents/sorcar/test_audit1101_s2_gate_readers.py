# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S2, F11 and F12: the coding gate's readers of results and reviews.

* F11: a host ``Bash`` result opens with ``Error (exit code N):`` when the
  command failed; a ``[exit code: M]`` the command itself printed (a nested
  tool's output, a fixture) must not override that status in the ledger,
  or a failing check is shown to the reviewer as evidence that passed.
* F12: a reviewer answer of ``VERDICT: NOT DONE`` with nothing after it is
  an incomplete review (a truncated answer), not an approval.  Since the
  gate redesign of 2026-11-04 it refuses nothing either (a reviewer that
  cannot answer must not cost the run a round): it keeps the previous
  findings open and counts under ``MAX_FAILED_REVIEWS``, as a call that
  fails does; only ``VERDICT: OK`` clears the findings, and only a review
  with findings refuses.

The scripted local model server of the coding-gate tests plays the reviewer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.coding_gate import (
    MAX_FAILED_REVIEWS,
    REVIEW_VERDICT,
    CodingGate,
    bash_exit_code,
    run_on_host,
)
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, text_body
from kiss.tests.conftest import posix_only

pytestmark = posix_only("bash runs the commands")


def test_host_failure_prefix_wins_over_a_marker_in_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 0)
    command = "printf '[exit code: 0]\\n'; exit 7"
    result = UsefulTools(work_dir=str(tmp_path)).Bash(command, "exit marker", timeout_seconds=5)
    assert result.startswith("Error (exit code 7):")
    assert bash_exit_code(result) == "7"
    gate = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                      str(tmp_path), "tmp")
    gate.result("Bash", {"command": command}, result, False)
    assert gate.exit_codes[command] == "7"
    assert gate.evidence_report({"evidence": [command]}) == f"- exit 7: `{command}`"
    # The container tool's marker still reads as before, a marker in a passing
    # host result too (the host tool never opens a passing result with the prefix).
    assert bash_exit_code("some output\n[exit code: 3]") == "3"
    assert bash_exit_code("[exit code: 0]\n") == "0"
    assert bash_exit_code("fixture says [exit code: 5]\n") == "5"
    assert bash_exit_code("Error (exit code 2):\nno marker") == "2"
    assert bash_exit_code("Error (exit code 2):\n[exit code: 4]") == "2"


def test_a_review_without_a_verdict_or_whose_call_fails_refuses_nothing(
    tmp_path: Path,
) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("exit 0\n")
    previous = ["1. The endpoint returns 500 on an empty body."]
    answers = ["VERDICT: NOT DONE", "Looks fine.",
               "VERDICT: NOT DONE\n1. The empty-body case is not handled.", "VERDICT: OK"]
    with serve([text_body(answer) for answer in answers]) as (url, requests):
        gate = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                          str(tmp_path), "tmp", reviewer=MODEL,
                          model_config={"base_url": url, "api_key": "local"})
        gate.open_findings = list(previous)
        # a bare NOT DONE, then an answer without a VERDICT line: no verdict, no retry,
        # the previous findings stay open and the review is still owed
        for failed in range(1, MAX_FAILED_REVIEWS + 1):
            assert gate.review_verdict({"summary_in_html": "done"}, "task") is None
            assert (gate.reviews, gate.failed_reviews) == (0, failed)
            assert gate.open_findings == previous and gate.changed_since_review
            assert len(requests) == failed
        assert not gate.review_possible()
        # A NOT DONE with its gaps is a finding; an OK clears the findings.
        gate.failed_reviews = 0
        verdict = gate.review_verdict({"summary_in_html": "done"}, "task")
        assert verdict == REVIEW_VERDICT.format(findings="1. The empty-body case is not handled.")
        assert gate.open_findings == ["1. The empty-body case is not handled."]
        assert (gate.reviews, gate.failed_reviews) == (1, 0) and not gate.changed_since_review
        assert gate.review_verdict({"summary_in_html": "done"}, "task") is None
        assert gate.open_findings == [] and (gate.reviews, gate.failed_reviews) == (2, 0)
        assert len(requests) == 4
    # a call that fails (a model nothing serves) is a failed review the same way: the
    # finish is not refused, and past the cap no review is asked for again
    broken = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                        str(tmp_path), "tmp", reviewer="no-such-model-xyz")
    broken.observe("Edit", {"file_path": str(tmp_path / "a.py"), "old_string": "a",
                            "new_string": "b"})
    done = {"success": True, "summary_in_html": "<p>Done.</p>"}
    for finish in (1, 2, 3):
        assert broken.finish_verdict(dict(done), "task") is None
        assert broken.failed_reviews == min(finish, MAX_FAILED_REVIEWS)
        assert broken.open_findings == [] and broken.gate_rounds == 0
    assert not broken.review_possible() and broken.reviews == 0
