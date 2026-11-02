# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A function call Gemini never signed must still carry a thought signature.

Gemini 3 and 4 models reject a request whose history holds a function
call without a thought signature (``400 INVALID_ARGUMENT: Function call
is missing a thought_signature in functionCall parts``).  Such calls
reach a GeminiModel whenever a conversation is handed off from another
provider's model (Sorcar's ``set_model``, the fallback swap) or when a
Gemini parallel batch is replayed (only its first call is signed).  The
documented way through is the dummy signature
``skip_thought_signature_validator`` on every unsigned call.  Gemini
validates the calls of the turn being continued (the request that
follows their tool results), which is exactly the first request after a
hand-off.

Seen live on 2026-11-01: a Sorcar task switching from a Claude model to
``gemini-4-argon`` failed on its first request with that 400.
"""

from __future__ import annotations

import copy

from kiss.core.models.gemini_model import FOREIGN_CALL_SIGNATURE, GeminiModel
from kiss.core.models.model_info import model
from kiss.tests.conftest import requires_gemini_api_key

_ANTHROPIC_STYLE_HISTORY: list[dict] = [
    {"role": "user", "content": "Call reveal_secret with index 1, then reply with the word only."},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Calling the tool."},
            {"type": "tool_use", "id": "toolu_01", "name": "reveal_secret", "input": {"index": 1}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": "The secret word is QUARTZ-PELICAN",
            }
        ],
    },
]

_OPENAI_STYLE_HISTORY: list[dict] = [
    {"role": "user", "content": "Call reveal_secret for index 1 and 2 in parallel."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "function": {"name": "reveal_secret", "arguments": '{"index": 1}'}},
            {"id": "call_2", "function": {"name": "reveal_secret", "arguments": '{"index": 2}'}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "The secret word is QUARTZ-PELICAN"},
    {"role": "tool", "tool_call_id": "call_2", "content": "The secret word is VELVET-COMPASS"},
]


def _function_call_parts(m: GeminiModel) -> list:
    contents = m._convert_conversation_to_gemini_contents(m._chat_messages())
    return [p for c in contents for p in c.parts or [] if p.function_call is not None]


def reveal_secret(index: int) -> str:
    """Return the secret word stored at *index*."""
    return f"The secret word is {['QUARTZ-PELICAN', 'VELVET-COMPASS'][index - 1]}"


def test_a_handed_off_anthropic_tool_use_gets_the_dummy_signature() -> None:
    m = GeminiModel("gemini-3-flash-preview", api_key="test")
    m.model_config = {}
    m.conversation = copy.deepcopy(_ANTHROPIC_STYLE_HISTORY)

    parts = _function_call_parts(m)

    assert [p.function_call.name for p in parts] == ["reveal_secret"]
    assert parts[0].thought_signature == FOREIGN_CALL_SIGNATURE


def test_a_stored_signature_is_kept_and_only_unsigned_calls_get_the_dummy() -> None:
    m = GeminiModel("gemini-3-flash-preview", api_key="test")
    m.model_config = {}
    m.conversation = copy.deepcopy(_OPENAI_STYLE_HISTORY)
    # A Gemini parallel batch signs its first call only.
    m._thought_signatures = {"call_1": b"real-signature"}

    parts = _function_call_parts(m)

    assert [p.thought_signature for p in parts] == [b"real-signature", FOREIGN_CALL_SIGNATURE]


@requires_gemini_api_key
def test_live_gemini_3_answers_after_a_hand_off_from_another_provider() -> None:
    """The exact failure seen live: a Claude-produced tool call replayed to Gemini 3.

    Tools are declared, as on every agent request, and the history ends
    with the tool result: the model continues its own turn, the one
    request shape Gemini validates.
    """
    m = model("gemini-3-flash-preview")
    assert isinstance(m, GeminiModel)
    m.initialize("placeholder")
    m.conversation = copy.deepcopy(_ANTHROPIC_STYLE_HISTORY)

    _calls, content, _response = m.generate_and_process_with_tools({"reveal_secret": reveal_secret})

    assert "QUARTZ-PELICAN" in content
