import sys
from pathlib import Path

from base.llm_solver import BaseLLMSolver
from utils.utils import register

from .model import KnapsackInstance, KnapsackLLMSolution, KnapsackLoader

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from utils.llm_output_utils import extract_csloi


@register("knapsack-llm")
class KnapsackLLM(BaseLLMSolver[KnapsackLLMSolution, KnapsackInstance]):
    here = Path(__file__).parent
    default_demo = KnapsackLoader.load(
        "data/problem_instances/knapsack/demo/problem.in",
        "data/problem_instances/knapsack/demo/solution.sol",
    )

    def solve(self, inst: KnapsackInstance) -> KnapsackLLMSolution:
        prompt, response = self.prompt_response(inst)

        object_nums = extract_csloi(
            response if isinstance(response, str) else response[-1]
        )

        selected_items = [num - 1 for num in object_nums]

        return KnapsackLLMSolution(
            prompt=prompt, response=response, selected_items=selected_items
        )
