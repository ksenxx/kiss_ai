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
