# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S2, F11 and F12: the coding gate's readers of results and reviews.

* F11: a host ``Bash`` result opens with ``Error (exit code N):`` when the
  command failed; a ``[exit code: M]`` the command itself printed (a nested
  tool's output, a fixture) must not override that status in the ledger,
  or a failing check is accepted as evidence.
* F12: a reviewer answer of ``VERDICT: NOT DONE`` with nothing after it is
  an incomplete review (a truncated answer), not an approval: it is
  retried once, then refused as a review that did not complete, and the
  previous findings stay open; only ``VERDICT: OK`` clears them.

The scripted local model server of the coding-gate tests plays the reviewer;
it is registered as the provider of the reviewer model too, so the retry on
the default model configuration reaches it as well, not the vendor.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from kiss.agents.sorcar.coding_gate import (
    REVIEW_FAILED_VERDICT,
    REVIEW_VERDICT,
    CodingGate,
    bash_exit_code,
    container_run,
    run_on_host,
)
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models import model_info
from kiss.tests.agents.sorcar.local_model_server import serve, text_body
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
    verdict = gate.evidence_field_verdict({"evidence": [command]})
    assert verdict is not None and "exit 7" in verdict
    # The container tool's marker still reads as before, a marker in a passing
    # host result too (the host tool never opens a passing result with the prefix).
    assert bash_exit_code("some output\n[exit code: 3]") == "3"
    assert bash_exit_code("[exit code: 0]\n") == "0"
    assert bash_exit_code("fixture says [exit code: 5]\n") == "5"
    assert bash_exit_code("Error (exit code 2):\nno marker") == "2"
    assert container_run("Error (exit code 2):\n[exit code: 4]") == (2, "Error (exit code 2):")


#: A catalogued OpenAI chat model without the Responses-API flag: the default
#: configuration builds the v1 adapter the scripted server speaks.
REVIEWER = "gpt-5-chat-latest"


def test_bare_not_done_is_an_incomplete_review_not_an_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("exit 0\n")
    answers = ["VERDICT: NOT DONE", "VERDICT: NOT DONE\n\n",
               "VERDICT: NOT DONE\n1. The empty-body case is not handled.", "VERDICT: OK"]
    with serve([text_body(answer) for answer in answers]) as (url, requests):
        openai = next(p for p in model_info.OPENAI_COMPATIBLE_PROVIDERS if p.name == "openai")
        local = dataclasses.replace(openai, name="local", base_url=url, prefixes=(REVIEWER,))
        monkeypatch.setattr(model_info, "OPENAI_COMPATIBLE_PROVIDERS",
                            (local, *model_info.OPENAI_COMPATIBLE_PROVIDERS))
        monkeypatch.setattr(DEFAULT_CONFIG, "OPENAI_API_KEY", "local")
        gate = CodingGate(lambda command, timeout: run_on_host(command, str(tmp_path), timeout),
                          str(tmp_path), "tmp", reviewer=REVIEWER,
                          model_config={"base_url": url, "api_key": "local"})
        gate.open_findings = ["1. The endpoint returns 500 on an empty body."]
        verdict = gate.review_verdict({"summary_in_html": "done"}, "task")
        assert verdict == REVIEW_FAILED_VERDICT
        assert (gate.reviews, gate.failed_reviews) == (0, 1)
        assert gate.open_findings == ["1. The endpoint returns 500 on an empty body."]
        assert gate.changed_since_review
        # The bare verdict was retried once (on the fallback configuration), like a missing one.
        assert len(requests) == 2
        # A NOT DONE with its gaps is a finding; an OK clears the findings.
        verdict = gate.review_verdict({"summary_in_html": "done"}, "task")
        assert verdict == REVIEW_VERDICT.format(findings="1. The empty-body case is not handled.")
        assert gate.open_findings == ["1. The empty-body case is not handled."]
        assert (gate.reviews, gate.failed_reviews) == (1, 1)
        assert gate.review_verdict({"summary_in_html": "done"}, "task") is None
        assert gate.open_findings == [] and (gate.reviews, gate.failed_reviews) == (2, 1)
        assert len(requests) == 4
