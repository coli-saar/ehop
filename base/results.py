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


def classify_result(result: Result) -> tuple[str, int | None, str]:
    match result:
        case OptimalResult(solution_string, summary_value):
            return "OPTIMAL", summary_value, solution_string
        case SuboptimalResult(solution_string, summary_value):
            return "SUBOPTIMAL", summary_value, solution_string
        case ErroneousResult(reason):
            return "ERRONEOUS", None, reason
        case IncompatibleFormatResult(reason):
            return "INCOMPATIBLE", None, reason
        case _:
            raise ValueError("Unrecognized result type")


class ILPException(Exception):
    def __init__(
        self,
        message: str,
        prompts: tuple[str, ...] | None = None,
        responses: tuple[str, ...] | None = None,
        reasons: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.prompts = prompts
        self.responses = responses
        self.reasons = reasons
