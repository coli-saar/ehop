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
from problems.graph_coloring.model import GraphColoringInstance, GraphColoringSolution

inst = GraphColoringInstance(
    nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)]),
    chromatic_number=3,
)


class GraphColoringValid(unittest.TestCase):
    def test_valid_perfect(self):
        solution = GraphColoringSolution(coloring=[1, 2, 3, 1, 2])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, OptimalResult)

    def test_valid_perfect_alternate(self):
        solution = GraphColoringSolution(coloring=[1, 3, 2, 1, 3])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, OptimalResult)

    def test_valid_suboptimal(self):
        solution = GraphColoringSolution(coloring=[1, 2, 3, 4, 1])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, SuboptimalResult)

    def test_valid_skips_color(self):
        solution = GraphColoringSolution(coloring=[1, 2, 3, 5, 1])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, SuboptimalResult)


class GraphColoringInvalid(unittest.TestCase):
    def test_invalid_empty(self):
        solution = GraphColoringSolution(coloring=[])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, IncompatibleFormatResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(
                evaluation.error,
                "Solution has colors for the wrong number of nodes (0 instead of 5).",
            )

    def test_invalid_monochromatic_edge(self):
        solution = GraphColoringSolution(coloring=[1, 2, 3, 4, 4])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, ErroneousResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(
                evaluation.error,
                "Solution has adjacent nodes with the same color. Nodes 4 and 5 are both colored 4.",
            )

    def test_invalid_uses_zero(self):
        solution = GraphColoringSolution(coloring=[0, 1, 2, 3, 0])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, IncompatibleFormatResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(
                evaluation.error,
                "Solution uses a bad set of colors (non-integers or integers outside the expected range).",
            )

    def test_invalid_beyond_max(self):
        solution = GraphColoringSolution(coloring=[1, 2, 3, 1, 6])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, IncompatibleFormatResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(
                evaluation.error,
                "Solution uses a bad set of colors (non-integers or integers outside the expected range).",
            )


if __name__ == "__main__":
    unittest.main()
