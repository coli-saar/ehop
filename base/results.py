from abc import ABC
from dataclasses import dataclass


class Result(ABC): ...


@dataclass
class ValidResult(Result):
    solution_string: str
    summary_value: int | bool | None = None


class OptimalResult(ValidResult): ...


class SuboptimalResult(ValidResult): ...


@dataclass
class InvalidResult(Result):
    error: str


class ErroneousResult(InvalidResult): ...


class IncompatibleFormatResult(InvalidResult): ...


class ILPException(Exception):
    def __init__(
        self,
        message: str,
        prompts: tuple[str, ...] | None = None,
        responses: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.prompts = prompts
        self.responses = responses
