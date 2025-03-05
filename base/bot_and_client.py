from abc import ABC, abstractmethod
from typing import Any

from base.results import ILPException


class BaseBot(ABC):
    """
    The BaseBot class provides a structure for generating messages to prompt an LLM
    in a conversational/chat setting, including in ways that depend on the LLM's responses.
    """

    @abstractmethod
    def get_message(self, llm_response: str) -> tuple[str | None, dict[str, Any]]:
        """
        Produces the next message to send to the LLM given the LLM's most recent response,
        along with a dictionary of keyword arguments to provide to the prompt method.
        The llm_response argument is typically ignored for the bot's first message.
        A return value of None indicates that the conversation is over.
        """
        ...


class BaseLLMClient(ABC):
    """
    The BaseLLMClient class is effectively a wrapper for LLM APIs and provides a method for sending messages and receiving responses.
    """

    @abstractmethod
    def prompt(self, message_history: tuple[str, ...] = (), **kwargs) -> str: ...

    @abstractmethod
    def bot_prompt(self, bot: BaseBot) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Prompts the LLM with messages one at a time, updating the context
        with the LLM's responses.
        Returns a tuple of the messages sent and the responses received.
        """
        history: list[str] = []
        messages: list[str] = []
        responses: list[str] = []

        try:
            message, kwargs = bot.get_message("")
            while message is not None:
                history.append(message)
                response = self.prompt(tuple(history), **kwargs)
                history.append(response)
                messages.append(message)
                responses.append(response)
                message, kwargs = bot.get_message(response)
        except ILPException as e:
            raise ILPException(str(e), tuple(messages), tuple(responses))
        else:
            return tuple(messages), tuple(responses)
