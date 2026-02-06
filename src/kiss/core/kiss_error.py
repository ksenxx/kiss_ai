# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Custom error class for KISS framework exceptions."""

import traceback

from kiss.core import config as config_module


class KISSError(ValueError):
    """Custom exception class for KISS framework errors."""

    def __init__(self, message: str):
        """Initialize a KISSError instance.

        Args:
            message: The error message describing the issue.
        """
        super().__init__(message)
        self._message = message

    def __str__(self) -> str:
        """Return a string representation of the error.

        Returns:
            str: The formatted error message, including traceback if debug mode is enabled.
        """
        extra = traceback.format_exc() if config_module.DEFAULT_CONFIG.agent.debug else ""
        return f"KISS Error: {self._message}\n{extra}"
