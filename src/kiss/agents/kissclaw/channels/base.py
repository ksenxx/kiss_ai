"""Abstract channel interface for KISSClaw."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from kiss.agents.kissclaw.types import Message


class Channel(ABC):
    """Abstract base class for messaging channels."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def send_message(self, jid: str, text: str) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def owns_jid(self, jid: str) -> bool: ...

    def set_typing(self, jid: str, is_typing: bool) -> None:
        pass

    def set_on_message(self, callback: Callable[[str, Message], None]) -> None:
        self._on_message = callback

    def set_on_chat_metadata(self, callback: Callable[[str, str], None]) -> None:
        self._on_chat_metadata = callback
