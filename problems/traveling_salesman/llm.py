import sys
from pathlib import Path

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.llm_solver import BaseLLMSolver
from problems.traveling_salesman.alt import TravelingSalesmanRandom
from problems.traveling_salesman.model import (
    TravelingSalesmanInstance,
    TravelingSalesmanLLMSolution,
    TravelingSalesmanLoader,
)
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
    random_solver = TravelingSalesmanRandom()

    def extract_solution(
        self,
        inst: TravelingSalesmanInstance,
        prompting: tuple[str, ...],
        response: str | tuple[str, ...],
        reasoning: str | tuple[str, ...] | None,
        loi: list[int],
    ) -> TravelingSalesmanLLMSolution:
        if len(loi) == inst.graph.number_of_nodes() + 1 and loi[-1] == 1:
            loi = loi[:-1]  # remove explicit return to node 1

        return TravelingSalesmanLLMSolution(
            prompting=prompting, response=response, reasoning=reasoning, ordering=loi
        )
