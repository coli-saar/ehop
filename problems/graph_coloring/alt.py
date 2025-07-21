from random import randint

import networkx as nx

from base.problem_structures import BaseSolver
from base.results import ValidResult
from utils.utils import register

from .model import GraphColoringInstance, GraphColoringSolution


@register("graph-coloring-greedy")
class GraphColoringGreedy(BaseSolver[GraphColoringSolution, GraphColoringInstance]):
    """
    A greedy solver for finding a valid (not necessarily optimal) graph coloring.
    The default strategy is "largest_first". This means that the nodes are colored
    in descending order of degree (number of neighbors).
    """

    def __init__(self, variant: str = "standard", strategy: str = "largest_first"):
        super().__init__(variant)
        self.strategy = (
            strategy
            if strategy != "node_order"
            else (lambda G, colors: list(range(1, G.number_of_nodes() + 1)))
        )

    def solve(self, inst: GraphColoringInstance) -> GraphColoringSolution:
        d = nx.greedy_color(inst.graph, strategy=self.strategy)  # type: ignore
        coloring = [d[v] + 1 for v in range(1, inst.graph.number_of_nodes() + 1)]

        return GraphColoringSolution(coloring=coloring)


@register("graph-coloring-random")
class GraphColoringRandom(BaseSolver[GraphColoringSolution, GraphColoringInstance]):
    """
    A random solver for generating a sequence of numbers that looks like a coloring,
    with no guarantees of validity.
    """

    def solve(self, inst: GraphColoringInstance) -> GraphColoringSolution:
        n = inst.graph.number_of_nodes()
        sol = GraphColoringSolution(coloring=[randint(1, n) for _ in range(n)])
        sol.simplify()
        return sol


@register("graph-coloring-random-valid")
class GraphColoringRandomValid(
    BaseSolver[GraphColoringSolution, GraphColoringInstance]
):
    """
    A random solver for generating a sequence of numbers that looks like a coloring
    guaranteed to be a valid solution.
    """

    def solve(self, inst: GraphColoringInstance) -> GraphColoringSolution:
        n = inst.graph.number_of_nodes()
        sol = GraphColoringSolution(coloring=[randint(1, n) for _ in range(n)])
        sol.simplify()
        while not isinstance(inst.evaluate(sol, variant=self.variant), ValidResult):
            for x, y in inst.graph.edges:
                if sol.coloring[x - 1] == sol.coloring[y - 1]:
                    sol.coloring[x - 1] = max(sol.coloring) + 1
            sol.simplify()
        return sol
