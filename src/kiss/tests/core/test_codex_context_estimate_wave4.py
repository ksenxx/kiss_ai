# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a Codex turn that reads token-dense text still reaches the hand-off line.

Review round 2 (Q1_core).  The Codex hook sizes the context by the
transcript the next turn sends, which it measured at four characters per
token.  That holds for ASCII text, but GPT-5's tokenizer (``o200k_base``)
spends 0.5 to 0.75 tokens on each CJK character and 0.35 on each
Devanagari one.  A bridged tool returning 400K tokens of Chinese (700K
characters) was sized at 175K, under the 201K average of the turn's two
requests, so the agent reported 201K, never reached its 350K hand-off
line, and sent all 400K again on the next turn.

Each case's text is one line repeated: the line's ``o200k_base`` token
count (measured with tiktoken, which this test does not load) times the
repeats is 400K, the usage the stand-in reports for the turn's second
request.  The stand-in ``codex`` and its MCP bridge are those of
``test_codex_context_estimate_wave3.py``.
"""

from pathlib import Path
from typing import Any

import pytest

from kiss.core.kiss_error import ContextWindowExceededError
from kiss.core.utils import finish as structured_finish
from kiss.tests.core.test_codex_context_estimate_wave3 import FINISH_TEXT, _run_codex_agent


def dense(line: str, repeats: int) -> str:
    """Return *line* repeated *repeats* times, as a big file read would.

    Args:
        line: The text of one line.
        repeats: How many times to repeat it.

    Returns:
        The text.
    """
    return line * repeats


@pytest.mark.parametrize(
    ("line", "tokens_per_line"),
    [
        ("你好，世界！\n", 4),
        ("これは日本語のテキストです。\n", 9),
        ("नमस्ते दुनिया! यह हिंदी पाठ है।\n", 11),
    ],
    ids=["chinese", "japanese", "hindi"],
)
def test_a_huge_dense_tool_result_reaches_the_hand_off_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, tokens_per_line: int
) -> None:
    """Requests of 1K and 401K tokens of dense text hand off; their 201K average did not."""
    repeats = 400_000 // tokens_per_line
    assert len(line * repeats) // 4 < 300_000, "four characters per token undercounts the text"
    turns: list[dict[str, Any]] = [
        {
            "mcp_calls": [["dense", {"line": line, "repeats": repeats}]],
            "text": "I read it.",
            "usage": {"input_tokens": 401_500, "output_tokens": 500},
        },
        {"mcp_calls": [], "text": FINISH_TEXT, "usage": {"input_tokens": 9, "output_tokens": 1}},
    ]
    agent, record_dir = _run_codex_agent(tmp_path, monkeypatch, turns)

    with pytest.raises(ContextWindowExceededError):
        agent.run(
            model_name="codex/gpt-5.5",
            prompt_template="Read the material with dense, then finish.",
            tools=[structured_finish, dense],
            is_agentic=True,
            verbose=False,
        )

    assert 400_000 <= agent.context_tokens_used <= 402_000, "the text read, within the usage"
    assert len(list(record_dir.glob("prompt-*.txt"))) == 1, "no turn was sent past the line"
