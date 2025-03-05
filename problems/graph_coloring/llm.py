import sys
from pathlib import Path

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.llm_solver import BaseLLMSolver
from problems.graph_coloring.model import (
    GraphColoringInstance,
    GraphColoringLLMSolution,
    GraphColoringLoader,
)
from utils.llm_output_utils import extract_csloi
from utils.utils import register


@register("graph-coloring-llm")
class GraphColoringLLM(BaseLLMSolver[GraphColoringLLMSolution, GraphColoringInstance]):
    here = Path(__file__).parent
    default_demo = GraphColoringLoader.load(
        "data/problem_instances/graph_coloring/demo/problem.col",
        "data/problem_instances/graph_coloring/demo/solution.sol",
    )

    def solve(self, inst: GraphColoringInstance) -> GraphColoringLLMSolution:
        prompt, response = self.prompt_response(inst)

        coloring = extract_csloi(
            response if isinstance(response, str) else response[-1]
        )

        return GraphColoringLLMSolution(
            prompt=prompt, response=response, coloring=coloring
        )
