# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the 2026-10-23 cost audit fixes.

1. Spend by model: a task that switched models with ``set_model`` used to
   be filed entirely under the model it was launched with.  Its row now
   records its own ``llm_call`` events per model (``model_costs``) when
   the final cost is saved, and the Spend report splits the task's own
   share by that record.
2. Anthropic cache writes: when a streamed ``message_delta`` raises the
   aggregate ``cache_creation_input_tokens`` past a ``message_start``
   split that shows only five-minute writes (KISS requests the
   five-minute default), the excess is billed at the five-minute rate
   (1.25x input), not the one-hour rate (2x).

No mocks: a temporary history database driven through the persistence
API the daemon uses, and a real local ``/v1/messages`` server streaming
to the real Anthropic SDK under a real ``KISSAgent``.
"""

from __future__ import annotations

import json
import shutil
import tempfile

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import calculate_cost
from kiss.tests.agents.sorcar.test_history_date_range import (
    _redirect,
    _restore,
    _set_timestamp,
)
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer, sse


@pytest.fixture
def history_db():
    """Point the persistence DB at an empty temporary directory."""
    tmpdir = tempfile.mkdtemp()
    saved = _redirect(tmpdir)
    yield
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    _restore(saved)
    shutil.rmtree(tmpdir, ignore_errors=True)


def _llm_call(model: str, cost: float, tokens: int) -> dict[str, object]:
    """An ``llm_call`` event as ``JsonPrinter`` broadcasts it."""
    return {
        "type": "llm_call",
        "model": model,
        "step": 1,
        "duration_ms": 10,
        "input_tokens": tokens // 2,
        "output_tokens": tokens - tokens // 2,
        "cache_read": 0,
        "cache_write": 0,
        "cost": cost,
    }


def _run_task(
    model: str,
    cost: float,
    tokens: int,
    calls: list[tuple[str, float, int]],
    parent: str = "",
) -> str:
    """Start a task row, persist its *calls*, then save its final usage."""
    extra: dict[str, object] = {"model": model}
    if parent:
        extra["parent_task_id"] = parent
    task_id, _ = th._add_task(f"task {model} {cost}", "", extra)
    for call in calls:
        th._queue_chat_event(_llm_call(*call), task_id)
    th._save_task_extra({"cost": cost, "tokens": tokens, "steps": 1}, task_id=task_id)
    _set_timestamp(task_id, 1_790_000_000.0)
    return task_id


def _by_model() -> dict[str, tuple[float, int]]:
    """The Spend report's per-model ``(cost, tokens)``."""
    return {
        str(r["model"]): (round(float(str(r["cost"])), 9), int(str(r["tokens"])))
        for r in th._spend_by_day_and_model()
    }


def _model_costs(task_id: str) -> str:
    row = th._get_db().execute(
        "SELECT model_costs FROM task_history WHERE id = ?", (task_id,)
    ).fetchone()
    return str(row[0])


@pytest.mark.usefixtures("history_db")
class TestSpendFollowsTheModelThatRan:
    def test_a_switched_task_is_split_by_its_calls(self) -> None:
        """Launched on Fable, switched to Opus: each model gets its calls' spend.

        The $0.50 no call explains (a decide() charge) stays with the
        launch model, and the token split follows the calls the same way.
        """
        task = _run_task(
            "claude-fable-5-1",
            9.5,
            10_000,
            [
                ("claude-fable-5-1", 0.25, 400),
                ("claude-fable-5-1", 0.75, 600),
                ("claude-opus-5-5", 8.0, 8_000),
            ],
        )
        assert json.loads(_model_costs(task)) == {
            "claude-fable-5-1": [1.0, 1_000],
            "claude-opus-5-5": [8.0, 8_000],
        }
        assert _by_model() == {
            "claude-fable-5-1": (1.5, 2_000),
            "claude-opus-5-5": (8.0, 8_000),
        }

    def test_sub_agents_and_parent_are_each_split(self) -> None:
        """A child on Astra keeps its share; the parent's own share is split."""
        parent = _run_task(
            "claude-fable-5-1",
            6.0,
            6_000,
            [("claude-fable-5-1", 1.0, 1_000), ("claude-opus-5-5", 3.0, 3_000)],
        )
        _run_task("gpt-6-astra", 2.0, 2_000, [("gpt-6-astra", 2.0, 2_000)], parent=parent)
        assert _by_model() == {
            "claude-fable-5-1": (1.0, 1_000),
            "claude-opus-5-5": (3.0, 3_000),
            "gpt-6-astra": (2.0, 2_000),
        }

    def test_calls_above_the_own_share_are_scaled_to_fit(self) -> None:
        """The row total is authoritative: calls summing above it are scaled down."""
        _run_task(
            "claude-fable-5-1",
            2.0,
            100,
            [("claude-fable-5-1", 1.0, 100), ("claude-opus-5-5", 3.0, 300)],
        )
        assert _by_model() == {
            "claude-fable-5-1": (0.5, 25),
            "claude-opus-5-5": (1.5, 75),
        }

    def test_a_provider_prefixed_launch_name_is_one_model(self) -> None:
        """Launched as anthropic/claude-sonnet-4-6, served as claude-sonnet-4-6.

        The adapter drops the redundant prefix, so the calls carry the bare
        name; the row's spend, unexplained remainder and task count are
        filed under that one name too.
        """
        _run_task(
            "anthropic/claude-sonnet-4-6",
            1.25,
            1_100,
            [("claude-sonnet-4-6", 1.0, 1_000)],
        )
        rows = th._spend_by_day_and_model()
        assert [(r["model"], r["tasks"]) for r in rows] == [("claude-sonnet-4-6", 1)]
        assert _by_model() == {"claude-sonnet-4-6": (1.25, 1_100)}

    def test_a_row_without_a_record_stays_with_its_model(self) -> None:
        """No llm_call events (a legacy row): everything goes to the row's model."""
        task = _run_task("claude-opus-5-5", 3.0, 300, [])
        assert _model_costs(task) == ""
        assert _by_model() == {"claude-opus-5-5": (3.0, 300)}

    def test_an_unreadable_record_stays_with_its_model(self) -> None:
        """A malformed model_costs value falls back to the row's model."""
        for bad in ("not json", '["a list"]', '{"m": "x"}'):
            task = _run_task("claude-opus-5-5", 3.0, 300, [])
            db = th._get_db()
            db.execute("UPDATE task_history SET model_costs = ? WHERE id = ?", (bad, task))
            db.commit()
            assert _by_model() == {"claude-opus-5-5": (3.0, 300)}
            db.execute("DELETE FROM task_history WHERE id = ?", (task,))
            db.commit()


_MODEL = "claude-opus-5-5"


def _streamed_turn(start_cache_write: int, final_cache_write: int) -> list[bytes]:
    """SSE chunks of one ``finish`` turn whose cache writes grow mid-stream.

    ``message_start`` splits *start_cache_write* tokens, all five-minute;
    ``message_delta`` reports the cumulative aggregate *final_cache_write*
    (the SDK updates the aggregate, not the split).
    """
    start_usage = {
        "input_tokens": 1_000,
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": start_cache_write,
        "cache_creation": {
            "ephemeral_5m_input_tokens": start_cache_write,
            "ephemeral_1h_input_tokens": 0,
        },
    }
    message: dict[str, object] = {
        "id": "msg_cache",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": _MODEL,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": start_usage,
    }
    return [
        sse("message_start", {"type": "message_start", "message": message}),
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_finish", "name": "finish"},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"result": "done"}'},
            },
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {
                    "output_tokens": 100,
                    "input_tokens": 1_000,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": final_cache_write,
                },
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]


def test_streamed_cache_writes_keep_the_five_minute_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 five-minute writes at message_start, 1,000 in total at message_delta.

    All 1,000 are five-minute writes ($5/MTok on Opus 5.5), not 200 at
    $5 plus 800 at the one-hour $8.
    """
    with ScriptedAnthropicServer(_streamed_turn(200, 1_000)) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        agent = KISSAgent("cache-ttl-cost")
        result = agent.run(
            model_name=_MODEL,
            prompt_template="hi",
            max_steps=3,
            max_budget=1.0,
            verbose=False,
            model_config={"api_key": "sk-test"},
        )
    assert result == "done"
    expected = calculate_cost(_MODEL, 1_000, 100, 0, 1_000, 0)
    assert expected == pytest.approx(1_000 * 4e-6 + 100 * 20e-6 + 1_000 * 5e-6)
    assert agent.budget_used == pytest.approx(expected)
    assert agent.last_call_usage is not None
    assert agent.last_call_usage["cache_write"] == 1_000
