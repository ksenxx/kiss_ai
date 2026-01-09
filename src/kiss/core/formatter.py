# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract formatter interface for message formatting and printing."""

from abc import ABC, abstractmethod


class Formatter(ABC):
    @abstractmethod
    def format_message(self, message: dict[str, str]) -> str:
        """Format a single message."""
        pass

    @abstractmethod
    def format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format a list of messages."""
        pass

    @abstractmethod
    def print_message(self, message: dict[str, str]) -> None:
        """Print a single message."""
        pass

    @abstractmethod
    def print_messages(self, messages: list[dict[str, str]]) -> None:
        """Print a list of messages."""
        pass

    @abstractmethod
    def print_status(self, message: str) -> None:
        """Print a status message."""
        pass

    @abstractmethod
    def print_error(self, message: str) -> None:
        """Print an error message."""
        pass

    @abstractmethod
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        pass
