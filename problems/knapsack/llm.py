import sys
from pathlib import Path

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.llm_solver import BaseLLMSolver
from problems.knapsack.alt import KnapsackRandomValid
from problems.knapsack.model import (
    KnapsackInstance,
    KnapsackLLMSolution,
    KnapsackLoader,
)
from utils.utils import register


@register("knapsack-llm")
class KnapsackLLM(BaseLLMSolver[KnapsackLLMSolution, KnapsackInstance]):
    here = Path(__file__).parent
    default_demo = KnapsackLoader.load(
        "data/problem_instances/knapsack/demo/problem.in",
        "data/problem_instances/knapsack/demo/solution.sol",
    )
    random_solver = KnapsackRandomValid()

    def extract_solution(
        self,
        inst: KnapsackInstance,
        prompting: tuple[str, ...],
        response: str | tuple[str, ...],
        reasoning: str | tuple[str, ...] | None,
        loi: list[int],
    ) -> KnapsackLLMSolution:
        return KnapsackLLMSolution(
            prompting=prompting,
            response=response,
            reasoning=reasoning,
            selected_items=[num - 1 for num in loi],
        )
