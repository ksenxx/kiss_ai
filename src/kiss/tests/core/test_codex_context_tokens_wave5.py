# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a Codex turn is sized with GPT-5's tokenizer, not a byte ratio.

Review round 3 (P1_core).  ``codex exec --json`` sums the usage of a
turn's requests, so the hook sizes what the next turn sends.  Wave 4
counted four UTF-8 bytes per token, which undercounts token-dense text:
GPT-5's tokenizer (``o200k_base``) spends 17 tokens on a 42-byte JSON
line, 36 on a 73-byte line of dense Chinese, 16 on a 41-byte line of
emoji and 21 on a 37-byte base64 line.  A bridged tool returning 400K
tokens of such text was sized at 176K to 256K, under the 350K hand-off
line, so the next turn sent all 400K again.  The hook now counts the
tokens with ``o200k_base``, read offline from the copy of its ranks
litellm installs, and without that tokenizer counts the UTF-8 bytes,
which no byte-level BPE token count exceeds.

Each line's token count was measured with tiktoken's own ``o200k_base``.
The stand-in ``codex`` and its MCP bridge are those of
``test_codex_context_estimate_wave3.py``.  The test of the fallback runs
in a child process that blocks ``import tiktoken`` through
``sys.modules`` (Python's documented way to make an import fail): this
install always has tiktoken, so an install without it cannot be had
otherwise.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import ContextWindowExceededError
from kiss.core.utils import finish as structured_finish
from kiss.tests.core.test_codex_context_estimate_wave3 import (
    FINISH_TEXT,
    _reported_context,
    _run_codex_agent,
)
from kiss.tests.core.test_codex_context_estimate_wave4 import dense

JSON_LINE = '{"id":12345,"status":true,"value":0.1234}\n'
JSON_LINE_TOKENS = 17


def _run(agent: KISSAgent) -> str:
    """Run the agentic task on ``codex/gpt-5.5`` with the ``dense`` tool; return its result."""
    return agent.run(
        model_name="codex/gpt-5.5",
        prompt_template="Read the material with dense, then finish.",
        tools=[structured_finish, dense],
        is_agentic=True,
        verbose=False,
    )


def read_400k_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, tokens_per_line: int
) -> tuple[KISSAgent, Path]:
    """Run a turn whose bridged tool returns 400K tokens of *line*; it must hand off.

    The turn's usage is that of a first request of 1K tokens and a second
    one of 401K, and the line is so dense that four bytes per token size
    it under the hand-off line.

    Args:
        tmp_path: A fresh directory for the stand-in ``codex``.
        monkeypatch: Puts the stand-in on PATH.
        line: The text of one line of the tool result.
        tokens_per_line: The line's ``o200k_base`` token count.

    Returns:
        The agent and the directory of the prompts the stand-in received.
    """
    repeats = 400_000 // tokens_per_line
    assert len((line * repeats).encode()) // 4 < 300_000, "four bytes per token undercount it"
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
        _run(agent)
    return agent, record_dir


@pytest.mark.parametrize(
    ("line", "tokens_per_line"),
    [
        (JSON_LINE, JSON_LINE_TOKENS),
        ("甲乙丙丁戊己庚辛壬癸，子丑寅卯辰巳午未申酉戌亥。\n", 36),
        ("😀😃😄😁😆😅😂🤣😊😇\n", 16),
        ("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=\n", 21),
    ],
    ids=["json", "chinese", "emoji", "base64"],
)
def test_a_huge_token_dense_tool_result_reaches_the_hand_off_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, tokens_per_line: int
) -> None:
    """400K tokens of dense text hand off; four bytes per token sized them at 176K-256K."""
    agent, record_dir = read_400k_tokens(tmp_path, monkeypatch, line, tokens_per_line)

    assert 400_000 <= agent.context_tokens_used <= 402_000
    assert len(list(record_dir.glob("prompt-*.txt"))) == 1, "no turn was sent past the line"


def test_a_bill_past_the_line_over_a_smaller_transcript_is_sized_by_its_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five calls of 25K tokens: requests of 1K to 126K bill 384K, the context is ~126K."""
    repeats = 25_000 // JSON_LINE_TOKENS
    turns: list[dict[str, Any]] = [
        {
            "mcp_calls": [["dense", {"line": JSON_LINE, "repeats": repeats}]] * 5,
            "text": "Read.",
            "usage": {"input_tokens": 381_000, "output_tokens": 3_000},
        },
        {
            "mcp_calls": [],
            "text": FINISH_TEXT,
            "usage": {"input_tokens": 127_000, "output_tokens": 100},
        },
    ]
    agent, record_dir = _run_codex_agent(tmp_path, monkeypatch, turns)

    assert yaml.safe_load(_run(agent))["success"] is True, "the summed bill is no context"

    context = _reported_context((record_dir / "prompt-1.txt").read_text())
    assert 5 * repeats * JSON_LINE_TOKENS <= context < 150_000, context
    assert agent.context_tokens_used == 127_100, "a one-request turn is its whole usage"


def padded(spaces: int) -> str:
    """Return a line padded with *spaces* spaces, as a whitespace-filled file would read.

    Args:
        spaces: How many spaces pad the line.

    Returns:
        The padded line.
    """
    return "start" + " " * spaces + "end"


def test_a_whitespace_run_of_a_million_characters_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1.2M spaces in one run (~9.4K tokens): counted whole, tiktoken's regex panicked."""
    turns: list[dict[str, Any]] = [
        {
            "mcp_calls": [["padded", {"spaces": 1_200_000}]],
            "text": "Read.",
            "usage": {"input_tokens": 10_000, "output_tokens": 100},
        },
        {
            "mcp_calls": [],
            "text": FINISH_TEXT,
            "usage": {"input_tokens": 9_900, "output_tokens": 100},
        },
    ]
    agent, record_dir = _run_codex_agent(tmp_path, monkeypatch, turns)

    result = agent.run(
        model_name="codex/gpt-5.5",
        prompt_template="Read the material with padded, then finish.",
        tools=[structured_finish, padded],
        is_agentic=True,
        verbose=False,
    )

    assert yaml.safe_load(result)["success"] is True
    context = _reported_context((record_dir / "prompt-1.txt").read_text())
    assert 9_000 <= context <= 10_100, context


# Run in a child process: this install cannot be without tiktoken otherwise.
_WITHOUT_TIKTOKEN = """
import sys
sys.modules["tiktoken"] = None
from pathlib import Path
import pytest
from kiss.tests.core.test_codex_context_tokens_wave5 import (
    JSON_LINE, JSON_LINE_TOKENS, read_400k_tokens)
with pytest.MonkeyPatch.context() as monkeypatch:
    agent, record_dir = read_400k_tokens(
        Path(sys.argv[1]), monkeypatch, JSON_LINE, JSON_LINE_TOKENS)
print(agent.context_tokens_used, len(list(record_dir.glob("prompt-*.txt"))))
"""


def test_without_the_tokenizer_the_bytes_still_hand_off(tmp_path: Path) -> None:
    """Without tiktoken the UTF-8 bytes, more than the tokens, size the turn: it hands off."""
    done = subprocess.run(
        [sys.executable, "-c", _WITHOUT_TIKTOKEN, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.split()[-2:] == ["402000", "1"], "the whole usage bounds the bytes"
