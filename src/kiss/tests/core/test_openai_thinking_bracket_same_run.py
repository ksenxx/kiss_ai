# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `cut_then_answer` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
"""End-to-end tests: a broken OpenAI stream must close its thinking bracket.

``_stream_chat_completion`` opens the thinking bracket the moment a
provider emits its first ``reasoning_content`` delta, and used to close
it only when the loop ran to exhaustion.  ``stop_aware_events`` calls
``on_abort`` for a user stop and for a stall, but deliberately re-raises
every *other* transport failure untouched, so a provider that simply
dropped the connection mid-reasoning left the bracket open.

``KISSAgent._run_agentic_loop`` classifies such a failure as retryable
and takes the next step **in the same run**, without
``reset_conversation()``.  The printer was therefore still in thinking
mode when the retry's ordinary answer text arrived, and rendered the
answer as reasoning — grey italics under a "Thinking" rule that never
closed.

No mocks, patches, fakes or test doubles: a real ``ThreadingHTTPServer``
speaks genuine Chat Completions SSE to the real ``openai`` SDK, the real
``OpenAICompatibleModel`` streams it, and the real ``ConsolePrinter``
renders it into a ``StringIO``.
"""

from __future__ import annotations

import io

from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.core.models.openai_sse_harness import (
    ScriptedOpenAIServer,
)
from kiss.tests.core.models.test_openai_thinking_bracket_same_run import (  # noqa: F401
    _ANSWER,
    _DEADLINE,
    _MODEL,
    _PLAIN_ANSWER,
    _REASONING_THEN_CUT,
    _STALL_TIMEOUT,
    _CutThenAnswerPolicy,
    _delta_chunk,
    _make_model,
    _run_bounded,
    cut_then_answer,
)


def _closing_rule_between(rendered: str, opener: str, answer: str) -> bool:
    """Whether a bare rule line separates the *opener* rule from *answer*.

    ``ConsolePrinter`` draws ``──── Thinking ────`` to open a reasoning
    block and an unlabelled rule of the same character to close it, so a
    bare rule line between the two is exactly "the block ended before
    this text".

    Args:
        rendered: Everything the printer wrote.
        opener: The label on the opening rule.
        answer: The answer text that must fall outside the block.

    Returns:
        ``True`` when a closing rule precedes *answer*.
    """
    lines = rendered.splitlines()
    start = next(i for i, line in enumerate(lines) if opener in line)
    end = next(i for i, line in enumerate(lines) if answer in line)
    return any(
        set(lines[i].strip()) == {"─"} for i in range(start + 1, end) if lines[i].strip()
    )


def test_a_retry_in_the_same_run_does_not_render_answers_as_reasoning(
    cut_then_answer: tuple[ScriptedOpenAIServer, _CutThenAnswerPolicy],
) -> None:
    """The user-visible symptom, reproduced through a real ConsolePrinter.

    ``KISSAgent`` retries a dropped stream without resetting the model,
    so the second turn runs against the very same adapter and printer.
    The answer must arrive as plain text, after a *closed* thinking
    block — not inside the one the dead stream opened.
    """
    server, policy = cut_then_answer
    out = io.StringIO()
    printer = ConsolePrinter(file=out)
    model = _make_model(
        server.base_url, printer.token_callback, printer.thinking_callback
    )
    model.initialize("Think, then answer.")

    def retry() -> None:
        """Do exactly what the agent does on a retryable transport error."""
        # Same model, same printer, same run — no reset_conversation().
        model.add_message_to_conversation("user", "Failed to get response. Try again.")
        model.generate()

    cut, retried = _run_bounded(model.generate, retry)

    assert cut is not None, "the cut stream should have failed the turn"
    assert retried is None, f"the retry failed: {retried!r}"
    assert policy.requests == 2, "the retry never reached the provider"
    rendered = out.getvalue()
    assert _ANSWER in rendered
    assert _closing_rule_between(rendered, "Thinking", _ANSWER), (
        f"the answer was rendered inside the dead stream's thinking "
        f"block — no closing rule precedes it:\n{rendered!r}"
    )
