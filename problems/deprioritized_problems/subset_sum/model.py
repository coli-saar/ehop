from dataclasses import dataclass, field

from base.problem_structures import (
    BaseGenerator,
    BaseInstance,
    BaseLoader,
    BaseSolution,
)
from base.results import InvalidResult, Result, ValidResult
from utils.utils import register


@dataclass
class SubsetSumSolution(BaseSolution):
    solvable: bool = field(default=False)  # * See question below about solvability
    selected_values: list[int] = field(default_factory=list)  # [3, -1]

    def __str__(self) -> str:
        # TODO: Possibly change if we want to include solvability in the string representation
        return str(self.selected_values)


# @dataclass
# class SubsetSumInstance(BaseInstance[SubsetSumSolution]):
#     values: list[int]
#     target: int
#     solvable: bool | None = (
#         None  ##? Do we want to make this a decision problem? Or should all instances be solvable?
#     )

#     # TODO: Implement methods

#     def evaluate(self, solution: SubsetSumSolution, verbose: bool = False) -> Result:
#         raise NotImplementedError

#     def reasonable_encoding(self) -> str:
#         raise NotImplementedError
