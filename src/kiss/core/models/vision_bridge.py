# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Eyes for a text-only model: a vision model describes what a tool returned.

A tool that returns an image or a PDF (the Read tool on a screenshot, a
chart, a scanned page) embeds the bytes in its result, and every adapter
sends them to the model as image or document parts.  A model that cannot
see (DeepSeek, Kimi, a text Qwen, a local model) then gets raw base64 or
nothing, and image-centred work fails on it where a model that sees
succeeds.

:func:`describe_attachments` is what
:meth:`~kiss.core.models.model.Model.tool_result_text_and_attachments`
calls instead of sending the bytes to a model that cannot see them: one
non-agentic call per attachment to the first configured model of
:data:`VISION_MODELS`, whose description replaces the attachment in the
tool message.  It never raises on its own: a missing vision model or a
failed call becomes a line that tells the model why it has no picture
(a Stop, and the caller's own limit check, do propagate).  What the
calls cost is handed, call by call, to the caller's ``credit`` callback:
the calling model keeps it for its agent, which bills it against its own
budget like every other spend made on its behalf.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence

from kiss.core.models.model import Attachment
from kiss.core.models.model_families import _available

logger = logging.getLogger(__name__)

VISION_MODELS: tuple[str, ...] = (
    "claude-sonnet-5-5",
    "gpt-6-luna",
    "gemini-3.5-flash-lite",
    "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/google/gemini-3.5-flash-lite",
)
"""Models that describe an attachment, in the order they are preferred."""

DESCRIBE_PROMPT = """\
The attached {kind} was returned by a tool to an engineer who cannot see it. Describe it \
so that they can work from your words alone: its pixel dimensions (or page count) if known, \
the layout, every piece of text verbatim, labelled regions, axes and legends, the colours \
and shapes that carry meaning, and anything that looks like an error, a chart value or a \
code listing. State only what is in the {kind}; give no advice.{context}"""
"""The description request; ``kind`` is ``image`` or ``PDF``, ``context`` the tool's text."""

DESCRIPTION_HEADER = (
    "[Description of {mime} attachment {index} by {model}; "
    "the model you run on ({caller}) cannot see images]"
)
"""Opens each description so the model knows who saw the attachment and why."""

FAILURE_LINE = (
    "[{mime} attachment {index} could not be described: {reason}; "
    "the model you run on ({caller}) cannot see images, use a vision model or an OCR tool]"
)
"""Replaces an attachment nothing could describe."""

CONTEXT_LIMIT = 2000
"""Characters of the tool result's text shown to the vision model as context."""


def vision_model() -> str | None:
    """Return the first model of :data:`VISION_MODELS` whose provider is configured.

    Returns:
        A model name, or ``None`` when no vision model is available.
    """
    for candidate in VISION_MODELS:
        if _available(candidate):
            return candidate
    return None


def describable(attachment: Attachment) -> bool:
    """Return whether the bridge is for *attachment*: an image or a PDF.

    An audio or video clip is not: a model that cannot see may still
    hear, so the adapter carries the clip as its own content part.
    :func:`describe_attachments` takes describable attachments only.

    Args:
        attachment: One attachment lifted out of a tool result.
    """
    return attachment.mime_type.startswith("image/") or attachment.mime_type == "application/pdf"


def _describe_one(
    attachment: Attachment,
    index: int,
    model_name: str | None,
    caller_model: str,
    what: str,
    credit: Callable[[float, int, str, str], None] | None,
    limit_check: Callable[[], None] | None,
) -> str:
    """Describe one attachment with *model_name*, or say why it could not be described.

    Args:
        attachment: The image or PDF to describe (see :func:`describable`).
        index: Its 1-based position among the tool result's attachments.
        model_name: The vision model, or ``None`` when none is configured.
        caller_model: The text-only model the description is for.
        what: The text of the tool result the attachment came with.
        credit: Called with the call's cost, tokens, a key unique to
            the call and *model_name* as soon as it ends, however it ends (a stop
            included), so its billed spend reaches the caller even when
            nothing is returned; called again with the same key when a
            stop interrupts it, so it must add a key once.  ``None``
            when nobody collects them.
        limit_check: Called before the call is made; ``None`` for none.

    Returns:
        The description under :data:`DESCRIPTION_HEADER`, or a
        :data:`FAILURE_LINE`.

    Raises:
        KeyboardInterrupt: The user stopped the task during the call.
        KISSError: *limit_check* found the caller's budget spent.
    """
    kind = "PDF" if attachment.mime_type == "application/pdf" else "image"
    failure = {"mime": attachment.mime_type, "index": index, "caller": caller_model}
    if model_name is None:
        logger.warning(
            "Vision bridge: no vision model is configured; %s attachment %d was not described "
            "for %s, which cannot see images",
            attachment.mime_type,
            index,
            caller_model,
        )
        return FAILURE_LINE.format(reason="no vision model is configured", **failure)
    from kiss.core.kiss_agent import KISSAgent

    if limit_check is not None:
        limit_check()
    context = f"\nThe tool result it came with says:\n{what[:CONTEXT_LIMIT]}" if what else ""
    agent = KISSAgent("Vision bridge")
    key = f"vision-bridge:{uuid.uuid4().hex}"

    def bill() -> None:
        if credit is not None:
            credit(agent.budget_used, agent.total_tokens_used, key, model_name)

    try:
        try:
            description = agent.run(
                model_name=model_name,
                prompt_template="{p}",
                arguments={"p": DESCRIBE_PROMPT.format(kind=kind, context=context)},
                is_agentic=False,
                verbose=False,
                print_prompts=False,
                attachments=[attachment],
            )
        finally:
            bill()
    except BaseException as failed:
        if not isinstance(failed, Exception):
            # A stop can land inside the finally's own credit: credit
            # again (once per key) before it propagates.
            bill()
            raise
        # A call that raised may still have been billed (the agent's
        # partial-usage bill): its spend is reported like a success's.
        logger.warning(
            "Vision bridge: %s could not describe attachment %d", model_name, index, exc_info=True
        )
        return FAILURE_LINE.format(reason=f"{model_name} failed: {failed!r:.200}", **failure)
    logger.info(
        "Vision bridge: %s described %s attachment %d (%d bytes) for %s; "
        "cost $%.4f, %d tokens, billed to the calling agent",
        model_name,
        attachment.mime_type,
        index,
        len(attachment.data),
        caller_model,
        agent.budget_used,
        agent.total_tokens_used,
    )
    header = DESCRIPTION_HEADER.format(
        mime=attachment.mime_type, index=index, model=model_name, caller=caller_model
    )
    return f"{header} {description.strip() or '(the vision model returned no text)'}"


def describe_attachments(
    attachments: Sequence[Attachment],
    caller_model: str,
    what: str = "",
    *,
    credit: Callable[[float, int, str, str], None] | None = None,
    limit_check: Callable[[], None] | None = None,
) -> str:
    """Describe tool-result attachments in words for a model that cannot see them.

    One non-agentic call per image or PDF to :func:`vision_model`, with
    the attachment and :data:`DESCRIBE_PROMPT`; the cost of each call is
    logged at INFO with the model's name.  A caller that bills through
    *credit* gets each call's spend as the call ends, so a stop (which
    propagates) loses none of it.  An attachment that cannot be
    described (no vision model configured, an API error) becomes a
    :data:`FAILURE_LINE` that tells the model why it has no picture and
    what to do instead; only a Stop and *limit_check* raise.

    Args:
        attachments: The images and PDFs lifted out of one tool result
            (the :func:`describable` ones; the caller keeps the rest).
        caller_model: The name of the model the text is for; quoted in
            every line so the model knows it is the one that cannot see.
        what: The text of the tool result the attachments came with (the
            Read tool names the file there), shown to the vision model as
            context; empty when there is none.
        credit: Called with each call's cost, tokens, unique key and
            vision model (for the caller's split by model) as soon as
            that call ends, a stopped call's partial usage
            included, so the spend reaches the caller even when this
            function never returns (``Model._credit_helper_usage``);
            possibly twice per key (it must add a key once).
        limit_check: Called before each call is made; the calling
            agent's limit check, which raises once what ``credit``
            received has spent the budget, so a batch of descriptions
            stops at the cap instead of paying for every attachment.

    Returns:
        The descriptions, one block per attachment separated by blank
        lines.

    Raises:
        KeyboardInterrupt: The user stopped the task during a call.
        KISSError: *limit_check* found the caller's budget spent.
    """
    model_name = vision_model()
    return "\n\n".join(
        _describe_one(attachment, index, model_name, caller_model, what, credit, limit_check)
        for index, attachment in enumerate(attachments, 1)
    )
