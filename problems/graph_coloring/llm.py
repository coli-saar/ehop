import sys
from pathlib import Path

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.llm_solver import BaseLLMSolver
from problems.graph_coloring.alt import GraphColoringRandomValid
from problems.graph_coloring.model import (
    GraphColoringInstance,
    GraphColoringLLMSolution,
    GraphColoringLoader,
)
from utils.utils import register


@register("graph-coloring-llm")
class GraphColoringLLM(BaseLLMSolver[GraphColoringLLMSolution, GraphColoringInstance]):
    here = Path(__file__).parent
    default_demo = GraphColoringLoader.load(
        "data/problem_instances/graph_coloring/demo/problem.col",
        "data/problem_instances/graph_coloring/demo/solution.sol",
    )
    random_solver = GraphColoringRandomValid()

    def extract_solution(
        self,
        inst: GraphColoringInstance,
        prompting: tuple[str, ...],
        response: str | tuple[str, ...],
        reasoning: str | tuple[str, ...] | None,
        loi: list[int],
    ) -> GraphColoringLLMSolution:
        return GraphColoringLLMSolution(
            prompting=prompting, response=response, reasoning=reasoning, coloring=loi
        )
