import sys
import unittest

import networkx as nx

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory
from base.results import (
    ErroneousResult,
    IncompatibleFormatResult,
    OptimalResult,
    SuboptimalResult,
)
from problems.traveling_salesman.model import (
    TravelingSalesmanInstance,
    TravelingSalesmanSolution,
)

g = nx.complete_graph(4)
graph = nx.relabel_nodes(g, {i: i + 1 for i in range(4)})

weights = [1, 3, 3, 4, 5, 7]

nx.set_edge_attributes(
    graph,
    {edge: weight for edge, weight in zip(graph.edges(), weights)},
    "weight",
)

inst = TravelingSalesmanInstance(graph=graph, minimum_ordering=[1, 2, 3, 4])


class TravelingSalesmanValid(unittest.TestCase):
    def test_valid_perfect(self):
        solution = TravelingSalesmanSolution(ordering=[1, 2, 3, 4])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, OptimalResult)

    def test_valid_perfect_reversed(self):
        solution = TravelingSalesmanSolution(ordering=[1, 4, 3, 2])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, OptimalResult)

    def test_valid_perfect_alternate(self):
        solution = TravelingSalesmanSolution(ordering=[1, 3, 2, 4])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, OptimalResult)

    def test_valid_suboptimal(self):
        solution = TravelingSalesmanSolution(ordering=[1, 3, 4, 2])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, SuboptimalResult)


class TravelingSalesmanInvalid(unittest.TestCase):
    def test_invalid_empty(self):
        solution = TravelingSalesmanSolution(ordering=[])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, IncompatibleFormatResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(
                evaluation.error,
                "Solution has ordering with the wrong number of locations (0 instead of 4).",
            )

    def test_invalid_start(self):
        solution = TravelingSalesmanSolution(ordering=[2, 3, 4, 1])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, ErroneousResult)
        if isinstance(evaluation, ErroneousResult):
            self.assertEqual(evaluation.error, "Solution does not start at location 1.")

    def test_invalid_uses_zero(self):
        solution = TravelingSalesmanSolution(ordering=[0, 1, 2, 3])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, IncompatibleFormatResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(
                evaluation.error,
                "Solution uses a bad set of locations (non-integers or integers outside the expected range).",
            )


if __name__ == "__main__":
    unittest.main()
