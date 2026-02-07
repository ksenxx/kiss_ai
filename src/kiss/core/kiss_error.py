# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Custom error class for KISS framework exceptions."""

import traceback

from kiss.core import config as config_module


class KISSError(ValueError):
    """Custom exception class for KISS framework errors."""

    def __str__(self) -> str:
        extra = traceback.format_exc() if config_module.DEFAULT_CONFIG.agent.debug else ""
        return f"KISS Error: {super().__str__()}\n{extra}"
