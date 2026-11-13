# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The memory pages relevant to a task are listed in its first message, not searched by the model.

``MemoryTools.recall`` runs the search the memory protocol used to ask
the model to make as its first turn; ``SorcarAgent.run`` appends the
block to the task prompt.  The tests use the offline hashed embedder
(no ``OPENAI_API_KEY``) and the scripted local model, and read the
block off the first request's user message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.tools import MEMORY_PROTOCOL, RECALL_HEADING, MemoryTools
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve

PAGE = """---
title: Postgres pool exhaustion root cause
summary: The pool ran dry because sessions were never returned on timeout.
updated: '2026-10-20T10:00:00+00:00'
---
The pool exhaustion came from timeouts not releasing sessions.
"""


def _memory_root(tmp_path: Path) -> Path:
    root = tmp_path / "memories"
    root.mkdir()
    (root / "postgres-pool-exhaustion.md").write_text(PAGE, encoding="utf-8")
    return root


def test_recall_lists_matching_pages_with_date_and_summary(tmp_path: Path) -> None:
    tools = MemoryTools(_memory_root(tmp_path), embed=hashed_embedding)
    block = tools.recall("Why did the postgres pool exhaustion happen? sessions timeout")
    assert block.startswith(RECALL_HEADING)
    assert "- `postgres-pool-exhaustion` (2026-10-20): Postgres pool exhaustion root cause" in block
    assert "— The pool ran dry because sessions were never returned on timeout." in block
    assert block.endswith("memory_pull(query).")


def test_recall_is_empty_without_pages(tmp_path: Path) -> None:
    tools = MemoryTools(tmp_path / "empty", embed=hashed_embedding)
    assert tools.recall("anything") == ""


def test_protocol_no_longer_asks_for_a_search_first() -> None:
    assert "Before starting work, call `memory_search`" not in MEMORY_PROTOCOL
    assert "Memory pages relevant to this task" in MEMORY_PROTOCOL
    assert "at most three sites" in MEMORY_PROTOCOL


def test_first_message_of_a_run_carries_the_recall_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run with memory on gets the block in its prompt; the system prompt has the protocol."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    monkeypatch.setenv("KISS_USE_MEMORY", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _memory_root(home)
    work = tmp_path / "work"
    work.mkdir()
    agent = ChatSorcarAgent("recall-test")
    with serve([finish_body("<p>done</p>")]) as (url, requests):
        agent.run(
            prompt_template="Explain the postgres pool exhaustion and the sessions timeout",
            model_name=MODEL,
            work_dir=str(work),
            model_config={"base_url": url, "api_key": "local"},
            use_memory=True,
            verbose=False,
        )
    messages = requests[0]["messages"]
    user = [m for m in messages if m["role"] == "user"]
    assert user, messages
    first = str(user[0]["content"])
    assert RECALL_HEADING in first
    assert "`postgres-pool-exhaustion` (2026-10-20)" in first
    assert first.index("postgres pool exhaustion and the sessions timeout") < first.index(
        RECALL_HEADING
    )


PLACEHOLDER_PAGE = """---
title: Prompt templates fill the {task} placeholder
summary: The task argument replaces {task} in the prompt template.
updated: '2026-10-21T10:00:00+00:00'
---
A prompt template names its task argument as the {task} placeholder.
"""


def test_recalled_pages_are_not_filled_with_the_run_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page that quotes ``{task}`` reaches the model as written: only the template is filled."""
    home = tmp_path / "home"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "prompt-template-placeholder.md").write_text(
        PLACEHOLDER_PAGE, encoding="utf-8"
    )
    monkeypatch.setenv("KISS_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    task = "Explain how prompt templates fill the task placeholder with the task argument"
    with serve([finish_body("<p>done</p>")]) as (url, requests):
        SorcarAgent("recall-arguments").run(
            prompt_template="Task: {task}",
            arguments={"task": task},
            model_name=MODEL,
            work_dir=str(work),
            model_config={"base_url": url, "api_key": "local"},
            use_memory=True,
            verbose=False,
        )
    first = str(next(m for m in requests[0]["messages"] if m["role"] == "user")["content"])
    assert first.index(f"Task: {task}\n") < first.index(RECALL_HEADING)
    assert "Prompt templates fill the {task} placeholder" in first
    assert "The task argument replaces {task} in the prompt template." in first
