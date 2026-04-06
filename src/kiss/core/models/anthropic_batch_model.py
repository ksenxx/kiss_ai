# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Anthropic Batch API model — routes requests through the Message Batches API.

The Batch API provides a **50% discount** over standard API pricing in exchange
for asynchronous processing (up to 24 hours, though typically minutes).  This
model wraps the standard ``AnthropicModel`` and replaces the real-time
``messages.stream()`` call with a batch submission + poll loop.

Use the ``batch/`` model prefix to select this path::

    m = model("batch/claude-opus-4-6")
"""

import time
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model import TokenCallback


class AnthropicBatchModel(AnthropicModel):
    """A model that uses the Anthropic Message Batches API for 50% cheaper inference.

    Submits each request as a single-item batch, polls until completion,
    and returns the result.  Token streaming is not supported (the
    ``token_callback`` is ignored).

    Model names use the ``batch/`` prefix.  The part after the prefix is
    the underlying Anthropic model name (e.g. ``batch/claude-opus-4-6``
    → ``claude-opus-4-6``).
    """

    #: Seconds between batch status polls.
    POLL_INTERVAL: float = 2.0

    #: Maximum seconds to wait for a batch to complete.
    POLL_TIMEOUT: float = 86400.0  # 24 hours (Anthropic's SLA)

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
    ):
        """Initialize an AnthropicBatchModel instance.

        Args:
            model_name: Full model name including ``batch/`` prefix.
            api_key: The Anthropic API key for authentication.
            model_config: Optional configuration.  Recognised extra keys:
                - ``poll_interval`` (float): Seconds between polls (default 2.0).
                - ``poll_timeout`` (float): Max seconds to wait (default 86400).
            token_callback: Ignored — batch API does not support streaming.
        """
        # Strip "batch/" prefix for the actual Anthropic API model name
        self._api_model_name = model_name.removeprefix("batch/")
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            model_config=model_config,
            token_callback=None,  # Batch API does not stream
        )

    def _build_create_kwargs(self) -> dict[str, Any]:
        """Build kwargs, substituting the stripped API model name.

        The parent checks ``self.model_name`` for feature detection (thinking,
        max_tokens) — temporarily swap in the stripped name so those checks work,
        then ensure the final ``model`` key uses the API name.
        """
        saved = self.model_name
        self.model_name = self._api_model_name
        try:
            kwargs = super()._build_create_kwargs()
        finally:
            self.model_name = saved
        kwargs["model"] = self._api_model_name
        return kwargs

    def _create_message(self, kwargs: dict[str, Any]) -> Any:
        """Submit a single-item batch and poll until the result is ready.

        Args:
            kwargs: Keyword arguments for the Anthropic Messages API
                (same as ``messages.create()``).

        Returns:
            The Anthropic ``Message`` object from the completed batch result.

        Raises:
            KISSError: If the batch fails, is cancelled, or expires.
        """
        poll_interval = self.model_config.get("poll_interval", self.POLL_INTERVAL)
        poll_timeout = self.model_config.get("poll_timeout", self.POLL_TIMEOUT)

        # Remove keys not accepted by MessageCreateParamsNonStreaming
        exclude = {"cache_control", "poll_interval", "poll_timeout"}
        params = {k: v for k, v in kwargs.items() if k not in exclude}

        batch = self.client.messages.batches.create(
            requests=[{"custom_id": "req_0", "params": params}]
        )

        elapsed = 0.0
        while batch.processing_status != "ended":
            if elapsed >= poll_timeout:
                # Try to cancel the timed-out batch
                try:
                    self.client.messages.batches.cancel(batch.id)
                except Exception:  # pragma: no cover
                    pass
                raise KISSError(
                    f"Batch {batch.id} did not complete within {poll_timeout}s"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
            batch = self.client.messages.batches.retrieve(batch.id)

        # Retrieve the single result
        for result in self.client.messages.batches.results(batch.id):
            if result.custom_id == "req_0":
                if result.result.type == "succeeded":
                    return result.result.message
                elif result.result.type == "errored":
                    error = getattr(result.result, "error", None)
                    msg = str(error) if error else "Unknown batch error"
                    raise KISSError(f"Batch request failed: {msg}")
                else:
                    raise KISSError(
                        f"Batch request ended with status: {result.result.type}"
                    )

        raise KISSError(f"No result found for batch {batch.id}")  # pragma: no cover

    def __str__(self) -> str:
        return f"AnthropicBatchModel(name={self.model_name})"

    __repr__ = __str__
