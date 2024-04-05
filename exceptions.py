"""Custom exceptions module."""
from abc import ABC, abstractmethod


class IncorrectMode(ABC, Exception):
    """Abstract class of exception for incorrect mode."""

    @abstractmethod
    def __init__(self, msg: str) -> None:
        """Initialize abstract exception.

        Args:
            msg (str): exception message.
        """
        super().__init__(msg)
