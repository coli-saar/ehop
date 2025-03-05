# from mknapsack import solve_single_knapsack
from ortools.algorithms.python import knapsack_solver

from base.problem_structures import BaseSolver
from utils.utils import register

from .model import KnapsackInstance, KnapsackSolution


@register("knapsack-symbolic-brute")
class KnapsackBrute(BaseSolver[KnapsackSolution, KnapsackInstance]):
    """A brute-force solver for the knapsack problem."""

    def solve(self, inst: KnapsackInstance) -> KnapsackSolution:
        n = inst.num_items

        best_selection: list[int] = []
        best_profit = float("-inf")

        for mask in range(1 << n):
            selection = [i for i in range(n) if (mask & (1 << i))]

            weight = sum([inst.weights[i] for i in selection])
            if weight > inst.capacity:
                continue

            profit = sum([inst.profits[i] for i in selection])
            if profit > best_profit:
                best_profit = profit
                best_selection = selection

        solution = (
            best_selection
            if self.variant == "standard"
            else [i for i in range(inst.num_items) if i not in best_selection]
        )

        return KnapsackSolution(solution)


@register("knapsack-symbolic-ortools")
class KnapsackORTools(BaseSolver[KnapsackSolution, KnapsackInstance]):
    def solve(self, inst: KnapsackInstance) -> KnapsackSolution:
        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",
        )
        solver.init(inst.profits, [inst.weights], [inst.capacity])
        solver.solve()

        packed_item_ids = [
            i for i in range(len(inst.profits)) if solver.best_solution_contains(i)
        ]

        solution = (
            packed_item_ids
            if self.variant == "standard"
            else [i for i in range(inst.num_items) if i not in packed_item_ids]
        )

        return KnapsackSolution(solution)


# @register("knapsack-symbolic-mknapsack")
# class KnapsackMKnapsack(BaseSolver[KnapsackSolution, KnapsackInstance]):
#     def solve(self, inst: KnapsackInstance) -> KnapsackSolution:
#         #! This solver is currently unable to handle zero-weight items
#         solution = solve_single_knapsack(
#             inst.profits, inst.weights, inst.capacity, method="mt1r"
#         )

#         selected_items = [i for i in range(len(inst.profits)) if solution[i]]

#         return KnapsackSolution(selected_items)
