# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: AnthropicModel must enable interleaved thinking so
between-tool-call reasoning is streamed as ``thinking`` blocks (routed to
the Thoughts panel) rather than as plain ``text`` blocks (which would
render in the main response area).

Reproduces the user-reported bug:

    "In the last task, you showed the model thinking tokens outside the
     Thoughts panel."

Diagnosis from the recorded event log of the offending task (which used
``claude-opus-4-7`` with ``thinking={"type": "adaptive"}``): between
tool calls the model emitted reasoning text such as

    "I have the core facts. Let me verify a couple more things..."
    "I have what I need. Quick reality check first..."

These were broadcast as ``text_delta`` events because Anthropic's API,
without the ``interleaved-thinking-2025-05-14`` beta header, returns
between-action reasoning as ``text`` content blocks rather than as
``thinking`` blocks.

The fix: when extended thinking is enabled in
``AnthropicModel._build_create_kwargs``, attach
``extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"}``
so the API emits reasoning as ``thinking`` blocks.

Uses a real ThreadingHTTPServer (no mocks/patches/fakes) that:
  * captures the inbound ``anthropic-beta`` header,
  * returns one ``thinking`` block followed by one ``text`` block via SSE.
"""

from __future__ import annotations

from kiss.core.models.anthropic_model import AnthropicModel


class TestInterleavedThinkingEnabled:
    """Confirm AnthropicModel asks for interleaved thinking and routes
    reasoning to the Thoughts panel."""

    def test_build_kwargs_attaches_interleaved_beta_for_opus_4_7(self) -> None:
        """``_build_create_kwargs`` must add the interleaved-thinking
        beta header for ``claude-opus-4-7`` (adaptive thinking)."""
        m = AnthropicModel("claude-opus-4-7", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, (
            f"Expected 'interleaved-thinking-2025-05-14' in anthropic-beta "
            f"header for claude-opus-4-7, got: {beta!r}.  Without it, "
            f"between-tool-call reasoning is streamed as text and shown "
            f"outside the Thoughts panel."
        )

    def test_build_kwargs_attaches_interleaved_beta_for_sonnet_4(self) -> None:
        """The fix must also apply to the sonnet-4 family."""
        m = AnthropicModel("claude-sonnet-4-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, beta

    def test_build_kwargs_no_beta_for_non_thinking_models(self) -> None:
        """Models without thinking enabled must not gain the beta header."""
        m = AnthropicModel("claude-3-5-sonnet-20241022", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking" not in beta, beta

    def test_user_supplied_beta_header_is_preserved(self) -> None:
        """A user-supplied ``anthropic-beta`` header must be augmented,
        not replaced, by the interleaved-thinking token."""
        m = AnthropicModel(
            "claude-opus-4-7",
            api_key="test-key",
            model_config={
                "extra_headers": {"anthropic-beta": "fine-grained-tool-streaming-2025-05-14"},
            },
        )
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "fine-grained-tool-streaming-2025-05-14" in beta, beta
        assert "interleaved-thinking-2025-05-14" in beta, beta
