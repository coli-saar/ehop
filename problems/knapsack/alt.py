from random import randint, shuffle

from base.problem_structures import BaseSolver
from utils.utils import register

from .model import KnapsackInstance, KnapsackSolution


@register("knapsack-greedy")
class KnapsackGreedy(BaseSolver[KnapsackSolution, KnapsackInstance]):
    """
    An implementation of a greedy solver that goes through the items in a specific order
    and adds an item to the knapsack if it fits. The order of the items is determined by
    the strategy parameter.
    """

    def __init__(self, variant: str = "standard", strategy: str = "density"):
        super().__init__(variant)
        if strategy not in {"density", "value", "random"}:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        self.strategy: str = strategy

    def solve(self, inst: KnapsackInstance) -> KnapsackSolution:
        match self.strategy:
            case "density":
                # sort by value/weight ratio
                value_weight_ratio = [
                    (i, inst.profits[i] / inst.weights[i])
                    for i in range(inst.num_items)
                ]
                value_weight_ratio.sort(
                    key=lambda x: x[1], reverse=(self.variant == "standard")
                )
                indices = [i for i, _ in value_weight_ratio]
            case "value":
                # sort by value
                value_list = [(i, inst.profits[i]) for i in range(inst.num_items)]
                value_list.sort(
                    key=lambda x: x[1], reverse=(self.variant == "standard")
                )
                indices = [i for i, _ in value_list]
            case "item_order":
                indices = [i for i in range(inst.num_items)]
            case "random":
                indices = [i for i in range(inst.num_items)]
                shuffle(indices)
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # greedy selection
        selected_items = []

        if self.variant == "standard":
            remaining_capacity = inst.capacity
            for i in indices:
                if inst.weights[i] <= remaining_capacity:
                    selected_items.append(i)
                    remaining_capacity -= inst.weights[i]
        else:
            total_weight = sum(inst.weights)
            for i in indices:
                if total_weight - inst.weights[i] >= inst.complement_capacity:
                    selected_items.append(i)
                    total_weight -= inst.weights[i]

        return KnapsackSolution(selected_items)


@register("knapsack-random")
class KnapsackRandom(BaseSolver[KnapsackSolution, KnapsackInstance]):
    """
    A random solver for generating a sequence of selected items,
    with no guarantees of validity.
    """

    def solve(self, inst: KnapsackInstance) -> KnapsackSolution:
        return KnapsackSolution(
            selected_items=[i for i in range(inst.num_items) if randint(0, 1) == 1]
        )
