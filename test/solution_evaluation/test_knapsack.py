import sys
import unittest

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory
from base.results import (
    ErroneousResult,
    IncompatibleFormatResult,
    OptimalResult,
    SuboptimalResult,
)
from problems.knapsack.model import KnapsackInstance, KnapsackSolution

inst = KnapsackInstance(
    num_items=5,
    capacity=10,
    weights=[1, 2, 3, 4, 5],
    profits=[1, 2, 3, 4, 5],
    optimal_profit=10,
)


class KnapsackValid(unittest.TestCase):
    def test_valid_perfect(self):
        solution = KnapsackSolution(selected_items=[1, 2, 4])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, OptimalResult)

    def test_valid_perfect_alternate(self):
        solution = KnapsackSolution(selected_items=[0, 1, 2, 3])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, OptimalResult)

    def test_valid_suboptimal(self):
        solution = KnapsackSolution(selected_items=[0, 1])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, SuboptimalResult)

    def test_valid_empty(self):
        solution = KnapsackSolution(selected_items=[])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, SuboptimalResult)


class KnapsackInvalid(unittest.TestCase):
    def test_invalid_duplicate(self):
        solution = KnapsackSolution(selected_items=[0, 2, 0])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, IncompatibleFormatResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(evaluation.error, "Solution contains duplicate items.")

    def test_invalid_bad_number(self):
        solution = KnapsackSolution(selected_items=[0, 2, 5])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, IncompatibleFormatResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(evaluation.error, "Solution contains invalid items.")

    def test_invalid_too_heavy(self):
        solution = KnapsackSolution(selected_items=[1, 3, 4])
        evaluation = inst.evaluate(solution)
        self.assertIsInstance(evaluation, ErroneousResult)
        if isinstance(evaluation, IncompatibleFormatResult):
            self.assertEqual(evaluation.error, "Selected items are too heavy.")


if __name__ == "__main__":
    unittest.main()
