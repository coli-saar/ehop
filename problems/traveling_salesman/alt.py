from random import sample

import networkx as nx

from base.problem_structures import BaseSolver
from utils.utils import register

from .model import TravelingSalesmanInstance, TravelingSalesmanSolution


@register("traveling-salesman-greedy")
class TravelingSalesmanGreedy(
    BaseSolver[TravelingSalesmanSolution, TravelingSalesmanInstance]
):
    """
    An implementation of a greedy solver that repeatedly adds the
    closest unvisited location until all locations are visited.
    """

    def solve(self, inst: TravelingSalesmanInstance) -> TravelingSalesmanSolution:
        adj_mat = nx.adjacency_matrix(inst.graph).todense()
        to_visit = set(range(2, inst.graph.number_of_nodes() + 1))
        ordering = [1]
        optimization = min if self.variant == "standard" else max

        while to_visit:
            current = ordering[-1]
            next_node = optimization(
                to_visit,
                key=lambda x: adj_mat[current - 1, x - 1],
            )
            ordering.append(next_node)
            to_visit.remove(next_node)

        return TravelingSalesmanSolution(ordering)


@register("traveling-salesman-random")
class TravelingSalesmanRandom(
    BaseSolver[TravelingSalesmanSolution, TravelingSalesmanInstance]
):
    """
    A random solver for generating a sequence of locations,
    with no guarantees of optimality.
    """

    def solve(self, inst: TravelingSalesmanInstance) -> TravelingSalesmanSolution:
        n = inst.graph.number_of_nodes()
        return TravelingSalesmanSolution(
            ordering=[1] + sample(list(range(2, n + 1)), n - 1)
        )
