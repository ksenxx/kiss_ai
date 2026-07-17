# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Custom error class for KISS framework exceptions."""


class KISSError(ValueError):
    """Custom exception class for KISS framework errors."""

    def __init__(self, message: str, code: int | None = None) -> None:
        """Initializes a new instance of the KISSError class.

        Args:
            message: The error message.
            code: The error code.
        """
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        if self.code is not None:
            return f"KISS Error (Code: {self.code}): {super().__str__()}"
        return f"KISS Error: {super().__str__()}"


class BudgetExceededError(KISSError):
    """Raised when an agent reaches or exceeds its configured budget.

    A distinct type lets orchestrators stop immediately without invoking
    recovery LLMs (for example ``RelentlessAgent``'s trajectory summarizer),
    which would otherwise spend still more money after the hard limit.
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
    ``KISSAgent._run_agentic_loop`` immediately switches to the registered
    fallback model (see ``MODEL_INFO.json``) when one exists.
    """
