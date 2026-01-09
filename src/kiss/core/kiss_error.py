# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Custom error class for KISS framework exceptions."""

import traceback

from kiss.core.config import DEFAULT_CONFIG


class KISSError(ValueError):
    def __init__(self, message: str):
        super().__init__(message)
        self._message = message

    def __str__(self) -> str:
        extra = traceback.format_exc() if DEFAULT_CONFIG.agent.debug else ""
        return f"KISS Error: {self._message}\n{extra}"
