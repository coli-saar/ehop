from abc import ABC, abstractmethod
from typing import Any, Iterable

from base.results import ILPException


class BaseBot(ABC):
    """
    The BaseBot class provides a structure for generating messages to prompt an LLM
    in a conversational/chat setting, including in ways that depend on the LLM's responses.
    """

    @abstractmethod
    def is_conversational(self) -> bool:
        """
        Returns True if the bot is conversational, i.e., if messages should
        be accumulated in the prompting context. Returns False if messages
        should be sent individually without additional context.
        """
        ...

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
    def get_model(self) -> str:
        """
        Returns the name of the model being used.
        """
        ...

    @abstractmethod
    def is_reasoning(self) -> bool:
        """
        Returns True if the model produces reasoning output in addition to its responses.
        """
        ...

    @abstractmethod
    def set_history(self, history: Iterable[str]) -> None:
        """
        Replaces the client's internal history of messages with the given history.
        This can be used for few-shot prompting and is also used internally
        to clear the history when keep_history is False in the prompt method.
        """
        ...

    @abstractmethod
    def prompt(
        self, message: str, keep_history: bool = False, **kwargs
    ) -> tuple[str, str | None]:
        """
        Prompts the language model with a message and returns the response.
        If keep_history is True, the message is added to the past history of messages
        and this history is passed as context to the LLM. Otherwise, the model's history
        is reset and only the message is passed as context.
        Other kwargs are passed to the LLM API call.
        """
        ...

    def bot_prompt(
        self, bot: BaseBot, **kwargs
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """
        Prompts the LLM with messages one at a time, updating the context
        with the LLM's responses.
        Returns a tuple of the messages sent and the responses received.
        """
        messages: list[str] = []
        responses: list[str] = []
        reasons: list[str] = []

        message, extra_kwargs = bot.get_message("")
        while message is not None:
            messages.append(message)
            response, reasoning = self.prompt(
                message,
                keep_history=bot.is_conversational(),
                **{**kwargs, **extra_kwargs}
            )
            responses.append(response)
            if reasoning is not None:
                reasons.append(reasoning)
            try:
                message, extra_kwargs = bot.get_message(response)
            except ILPException as e:
                raise ILPException(
                    str(e), tuple(messages), tuple(responses), tuple(reasons)
                )
        return tuple(messages), tuple(responses), tuple(reasons)
