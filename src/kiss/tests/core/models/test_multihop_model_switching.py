# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Live multi-hop model-switching regression test.

The Sorcar ``set_model`` tool hands a live conversation over verbatim between
providers: ``new_model.conversation = old_model.conversation``.  Two-hop
hand-offs are covered by the per-pair tests
(``test_*_conversation_handoff.py``, ``TestModelSwitching*`` in
``test_openai_compatible_model2.py``).  This module exercises a *five-hop*
chain within a single conversation against the real provider APIs:

    OpenAI Chat Completions (v1) -> OpenAI Responses (v2) -> Anthropic
    -> Gemini -> OpenAI Responses (v2)

Each hop makes a real tool call that reveals a distinct secret word, so the
history accumulates every provider's native storage format in sequence
(chat ``tool_calls``/``role="tool"`` messages, Responses ``function_call`` /
``function_call_output`` / ``reasoning`` items, Anthropic ``tool_use`` /
``tool_result`` (and possibly ``thinking``) blocks, Gemini dict-argument
tool calls).  The final hop must (a) still be able to call tools on top of
that mixed history and (b) recall *all* secret words from every earlier hop,
proving that the format-conversion chain is lossless across more than two
hand-offs.  A structural assertion additionally checks that every secret
survives verbatim in the serialized conversation.
"""

import json
import os
from typing import Any

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model import Model
from kiss.core.models.model_info import model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)

# One distinct, unguessable secret per hop.  The final model can only know
# these words if every earlier hop's tool result survived every conversion.
_SECRETS: dict[int, str] = {
    1: "PLUTONIUM-KITE",
    2: "MARIGOLD-ANVIL",
    3: "ZEPHYR-LANTERN",
    4: "COBALT-MERIDIAN",
}


def reveal_secret(index: int) -> str:
    """Reveal the secret word stored at a given index.

    Args:
        index: The 1-based index of the secret to reveal (1 through 4).

    Returns:
        The secret word at that index, or ``UNKNOWN`` for a bad index.
    """
    return _SECRETS.get(int(index), "UNKNOWN")


def _run_tool_turn(m: Model, prompt: str, secret_index: int) -> None:
    """Drive one user turn on ``m``, executing ``reveal_secret`` calls.

    Mirrors the KissAgent loop: add the user message, generate with tools,
    execute each returned call, submit results via
    ``add_function_results_to_conversation_and_return``, and repeat until
    the model answers with plain text.

    Args:
        m: The live model to drive.
        prompt: The user instruction for this turn.
        secret_index: The index the model is asked to reveal; used only for
            the failure message.
    """
    m.add_message_to_conversation("user", prompt)
    saw_call = False
    for _ in range(4):
        function_calls, _content, _resp = m.generate_and_process_with_tools(
            {"reveal_secret": reveal_secret}
        )
        if not function_calls:
            break
        results: list[tuple[str, dict[str, Any]]] = []
        for fc in function_calls:
            raw_args = fc.get("arguments")
            if isinstance(raw_args, str):
                args = json.loads(raw_args) if raw_args.strip() else {}
            else:
                args = raw_args or {}
            if fc["name"] == "reveal_secret":
                saw_call = True
                result = reveal_secret(**args)
            else:  # pragma: no cover - models only see one tool
                result = f"unknown tool {fc['name']}"
            results.append((fc["name"], {"result": str(result)}))
        m.add_function_results_to_conversation_and_return(results)
    assert saw_call, (
        f"{type(m).__name__}({m.model_name}) never called reveal_secret "
        f"for secret #{secret_index}"
    )


def _handoff(old: Model, new: Model) -> Model:
    """Hand the live conversation from ``old`` to ``new`` like ``set_model``.

    Args:
        old: The model currently holding the conversation.
        new: The freshly-constructed replacement model.

    Returns:
        ``new``, initialized and holding ``old``'s conversation object.
    """
    new.initialize("")
    new.conversation = old.conversation
    new.usage_info_for_messages = old.usage_info_for_messages
    return new


def _make_v2() -> OpenAICompatibleModel2:
    """Build a live Responses-API model against api.openai.com.

    Returns:
        An (uninitialized) ``OpenAICompatibleModel2`` for ``gpt-5-mini``
        with low reasoning effort, so reasoning items appear in history.
    """
    return OpenAICompatibleModel2(
        model_name="gpt-5-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model_config={"reasoning_effort": "low"},
    )


@requires_openai_api_key
@requires_anthropic_api_key
@requires_gemini_api_key
class TestMultiHopModelSwitchingLive:
    """Five-hop live hand-off chain: chat -> v2 -> Anthropic -> Gemini -> v2."""

    def test_five_hop_chain_is_lossless(self) -> None:
        """Every hop tool-calls on the accumulated mixed-format history and
        the final model recalls all secrets from all previous hops."""
        # Hop 1: OpenAI Chat Completions (v1) native chat format.
        m1 = model("gpt-4o")
        assert isinstance(m1, OpenAICompatibleModel)
        m1.initialize(
            "We will play a memory game across this conversation. "
            "Follow each instruction exactly."
        )
        _run_tool_turn(
            m1,
            "Call the reveal_secret tool with index 1, then state the "
            "secret word you received.",
            1,
        )

        # Hop 2: OpenAI Responses API (v2) — history gains reasoning /
        # function_call / function_call_output items.
        m2 = _handoff(m1, _make_v2())
        _run_tool_turn(
            m2,
            "Call the reveal_secret tool with index 2, then state the "
            "secret word you received.",
            2,
        )

        # Hop 3: Anthropic Messages API — must digest chat + Responses items.
        m3 = _handoff(m2, model("claude-haiku-4-5"))
        assert isinstance(m3, AnthropicModel)
        _run_tool_turn(
            m3,
            "Call the reveal_secret tool with index 3, then state the "
            "secret word you received.",
            3,
        )

        # Hop 4: Gemini — must digest chat + Responses + Anthropic blocks.
        m4 = _handoff(m3, model("gemini-2.5-flash"))
        assert isinstance(m4, GeminiModel)
        _run_tool_turn(
            m4,
            "Call the reveal_secret tool with index 4, then state the "
            "secret word you received.",
            4,
        )

        # Hop 5: back to OpenAI Responses (v2).  First prove tools still
        # work on top of the four-provider history...
        m5 = _handoff(m4, _make_v2())
        _run_tool_turn(
            m5,
            "Call the reveal_secret tool with index 1 one more time to "
            "double-check it, then state the secret word you received.",
            1,
        )

        # ...then prove the history is semantically lossless: the model can
        # only list all four secrets if every hop's tool result survived
        # every conversion.
        m5.add_message_to_conversation(
            "user",
            "List ALL the secret words that were revealed by the "
            "reveal_secret tool at any point in this conversation, in "
            "order of their index. Reply with the words only.",
        )
        content, _resp = m5.generate()
        for secret in _SECRETS.values():
            assert secret in content, (
                f"final model failed to recall {secret!r}; got: {content!r}"
            )

        # Structural losslessness: every secret must still be present
        # verbatim somewhere in the handed-off conversation.  (Note the
        # list object itself may be rebound: v2 rebuilds the conversation
        # when converting foreign items to native Responses items.)
        serialized = json.dumps(m5.conversation, default=str)
        for secret in _SECRETS.values():
            assert secret in serialized
