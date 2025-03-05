import sys
from pathlib import Path

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.llm_solver import BaseLLMSolver
from problems.traveling_salesman.model import (
    TravelingSalesmanInstance,
    TravelingSalesmanLLMSolution,
    TravelingSalesmanLoader,
)
from utils.llm_output_utils import extract_csloi
from utils.utils import register


@register("traveling-salesman-llm")
class TravelingSalesmanLLM(
    BaseLLMSolver[TravelingSalesmanLLMSolution, TravelingSalesmanInstance]
):
    here = Path(__file__).parent
    default_demo = TravelingSalesmanLoader.load(
        "data/problem_instances/traveling_salesman/demo/problem.tsp",
        "data/problem_instances/traveling_salesman/demo/solution.sol",
    )

    def solve(self, inst: TravelingSalesmanInstance) -> TravelingSalesmanLLMSolution:
        prompt, response = self.prompt_response(inst)

        ordering = extract_csloi(
            response if isinstance(response, str) else response[-1]
        )

        if len(ordering) == inst.graph.number_of_nodes() + 1 and ordering[-1] == 1:
            ordering = ordering[:-1]  # remove explicit return to node 1

        return TravelingSalesmanLLMSolution(
            prompt=prompt, response=response, ordering=ordering
        )
