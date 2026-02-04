# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract formatter interface for message formatting and printing."""

from abc import ABC, abstractmethod
from typing import Any


class Formatter(ABC):
    """Abstract base class for message formatting and printing."""

    @abstractmethod
    def format_message(self, message: dict[str, Any]) -> str:
        """Format a single message.

        Args:
            message: A dictionary containing message data with 'role' and 'content' keys.

        Returns:
            str: The formatted message string.
        """
        pass

    @abstractmethod
    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format a list of messages.

        Args:
            messages: A list of message dictionaries.

        Returns:
            str: The formatted messages as a single string.
        """
        pass

    @abstractmethod
    def print_message(self, message: dict[str, Any]) -> None:
        """Print a single message.

        Args:
            message: A dictionary containing message data with 'role' and 'content' keys.
        """
        pass

    @abstractmethod
    def print_messages(self, messages: list[dict[str, Any]]) -> None:
        """Print a list of messages.

        Args:
            messages: A list of message dictionaries.
        """
        pass

    @abstractmethod
    def print_status(self, message: str) -> None:
        """Print a status message.

        Args:
            message: The status message to print.
        """
        pass

    @abstractmethod
    def print_error(self, message: str) -> None:
        """Print an error message.

        Args:
            message: The error message to print.
        """
        pass

    @abstractmethod
    def print_warning(self, message: str) -> None:
        """Print a warning message.

        Args:
            message: The warning message to print.
        """
        pass

    @abstractmethod
    def print_label_and_value(self, label: str, value: str) -> None:
        """Print a label and value pair.

        Args:
            label: The label text.
            value: The value text.
        """
        pass
