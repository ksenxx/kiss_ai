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
