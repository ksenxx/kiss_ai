# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests (real models) for the SYSTEM.md Internet-search-by-default policy.

SYSTEM.md instructs the agent to use Internet search extensively for ALL
tasks unless it is confident it can complete the task correctly without
searching. These tests run real models with the real SYSTEM_PROMPT and a
real web-fetching ``go_to_url`` tool and verify:

1. For a task that depends on current information, the agent DOES call
   ``go_to_url`` before finishing.
2. For a trivial task (arithmetic) the agent is confident about, it does
   NOT search and still produces the correct answer.
"""

from __future__ import annotations

import urllib.request

import pytest

from kiss.core.base import SYSTEM_PROMPT
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.conftest import has_api_key_for_model

TEST_MODELS = ["gemini-3-flash-preview", "gpt-5.4-mini", "claude-haiku-4-5"]

VISITED_URLS: list[str] = []


def go_to_url(url: str) -> str:
    """Navigate the browser to a URL and return the page content.

    Args:
        url: Full URL to open (e.g. a search engine query or a website).

    Returns:
        The textual content of the page (truncated).
    """
    VISITED_URLS.append(url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data: bytes = resp.read(20000)
            return data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error fetching {url}: {e}"


def _run_task(model_name: str, task: str) -> str:
    """Run a KISSAgent with the real SYSTEM_PROMPT and a real go_to_url tool."""
    VISITED_URLS.clear()
    agent = KISSAgent("InternetSearchPolicyTest")
    return agent.run(
        model_name=model_name,
        system_prompt=SYSTEM_PROMPT,
        prompt_template=task,
        tools=[go_to_url],
        max_steps=15,
        max_budget=2.0,
        verbose=False,
    )


@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_current_info_task_triggers_internet_search(model_name: str) -> None:
    """A task needing current information must trigger real go_to_url calls."""
    if not has_api_key_for_model(model_name):
        pytest.skip(f"No API key for {model_name}")
    task = (
        "What is the latest stable version of the Python programming "
        "language released as of today? Reply with the exact version number."
    )
    result = ""
    for attempt in range(2):  # retry once to absorb model/API nondeterminism
        try:
            result = _run_task(model_name, task)
        except KISSError:
            # The agent may exhaust max_steps/budget without calling
            # finish (nondeterministic live-model behavior).  The policy
            # under test is only that a search happened, so accept the
            # run if ``go_to_url`` was called before the error.
            if VISITED_URLS:
                break
            if attempt == 1:
                raise
            continue  # transient live-API failure; retry once
        if VISITED_URLS:
            break
    assert VISITED_URLS, (
        f"{model_name} finished without any Internet search; "
        f"SYSTEM.md search-by-default policy not followed. Result: {result}"
    )


@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_confident_trivial_task_skips_internet_search(model_name: str) -> None:
    """A trivial arithmetic task the model is confident about must not search."""
    if not has_api_key_for_model(model_name):
        pytest.skip(f"No API key for {model_name}")
    result = _run_task(
        model_name,
        "Compute 17 * 23 and finish immediately with just the numeric result.",
    )
    assert not VISITED_URLS, (
        f"{model_name} searched the Internet ({VISITED_URLS}) for trivial "
        f"arithmetic; the confidence exception in SYSTEM.md was not honored."
    )
    assert "391" in result
