# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Custom error class for KISS framework exceptions."""


class KISSError(ValueError):
    """Custom exception class for KISS framework errors."""

    def __str__(self) -> str:
        return f"KISS Error: {super().__str__()}"
