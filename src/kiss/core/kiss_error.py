# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Custom error class for KISS framework exceptions."""


class KISSError(ValueError):
    """Custom exception class for KISS framework errors."""

    def __str__(self) -> str:
        return f"KISS Error: {super().__str__()}"


class BudgetExceededError(KISSError):
    """Raised when an agent reaches or exceeds its configured budget.

    A distinct type lets orchestrators stop immediately without invoking
    recovery LLMs (for example ``RelentlessAgent``'s trajectory summarizer),
    which would otherwise spend still more money after the hard limit.
    """


class StepLimitExceededError(KISSError):
    """Raised when an agent's loop ends because it ran out of steps.

    A distinct type lets a caller tell exhaustion (the agent did its
    allotted work and found no answer) from a failure at the last step
    (a credential or transport error), which the step count alone cannot
    do because the count is advanced before the step runs.
    """


class ContextWindowExceededError(KISSError):
    """Raised when the conversation no longer fits the model's context window.

    Raised in two places in ``KISSAgent``:

    * proactively by ``_check_limits`` when the live conversation size
      (``context_tokens_used``) reaches ``CONTEXT_LIMIT_FRACTION`` of the
      model's maximum context length, and
    * reactively by ``_run_agentic_loop`` when the provider rejects a
      request with a context-overflow error (e.g. Anthropic's "Your input
      exceeds the context window of this model" or OpenAI's
      ``context_length_exceeded``).  Retrying such a request is pointless —
      the retry handler would append yet another message, growing the
      conversation further — so the loop converts it to this typed error
      immediately instead of burning ``MAX_CONSECUTIVE_ERRORS`` retries.

    A distinct type lets ``RelentlessAgent`` route the failure to its
    trajectory-summarizer/continuation path (starting a fresh session with a
    summary of progress) instead of hard-failing the whole task.
    """


class ModelRefusalError(KISSError):
    """Raised when a model refuses to answer for safety reasons.

    Anthropic adaptive-thinking models (e.g. ``claude-fable-5``) can return
    ``stop_reason="refusal"`` with an EMPTY ``content`` list when their
    safety layer declines a request (observed in production on benign
    security-research prompts).  Retrying the identical request is useless —
    the refusal is deterministic for the same content — so the adapter
    raises this error instead of silently returning an empty turn, and
    ``KISSAgent._run_agentic_loop`` immediately switches to a model of
    another family (see :mod:`kiss.core.models.model_families`) when one
    is configured.
    """

    stop_reason = "refusal"
    """The provider's stop reason, named by the agent's fail-over warning."""


class ContentFilterError(ModelRefusalError):
    """Raised when the provider's content filter stopped the response.

    OpenAI's Responses API reports it as an ``incomplete`` response with
    ``incomplete_details.reason == "content_filter"``, Chat Completions as
    ``finish_reason == "content_filter"``.  Like a safety refusal it is
    deterministic for the same request and shared by every model of the
    family, so the agentic loop fails over to another family instead of
    retrying the same one.
    """

    stop_reason = "content_filter"


class OutputCapError(KISSError):
    """Raised when the response hit the output-token cap before a usable answer.

    Responses API ``incomplete_details.reason == "max_output_tokens"``,
    Chat Completions ``finish_reason == "length"`` and Anthropic
    ``stop_reason == "max_tokens"`` with no complete tool call: the
    adapters raise instead of returning the cut-off text, since the
    agent cannot act on it.  The agentic loop retries the same request
    once at a lower reasoning effort and then fails over to a model of
    another family.
    """

    def __init__(self, message: str, stop_reason: str = "max_tokens") -> None:
        """Create the error.

        Args:
            message: What was cut short; it names the stop reason.
            stop_reason: The provider's stop reason (``max_output_tokens``,
                ``length`` or ``max_tokens``), named by the fail-over warning.
        """
        super().__init__(message)
        self.stop_reason = stop_reason
