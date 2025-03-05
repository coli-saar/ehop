from dataclasses import dataclass
from pathlib import Path

from base.problem_structures import (
    BaseInstance,
    BaseLLMSolution,
    BaseLoader,
    BaseSolution,
)
from base.results import (
    ErroneousResult,
    IncompatibleFormatResult,
    OptimalResult,
    Result,
    SuboptimalResult,
)
from utils.llm_output_utils import extract_csloi
from utils.utils import register


@dataclass
class KnapsackSolution(BaseSolution):
    selected_items: list[int]  # 0-based indices of selected items

    def __str__(self) -> str:
        return str(self.selected_items)


class KnapsackLLMSolution(BaseLLMSolution, KnapsackSolution): ...


@dataclass
class KnapsackInstance(BaseInstance[KnapsackSolution]):
    num_items: int
    profits: list[int]
    weights: list[int]
    capacity: int
    complement_capacity: int
    optimal_items: list[int] | None = None
    complement_optimal_items: list[int] | None = None
    optimal_profit: int | None = None
    complement_optimal_profit: int | None = None

    def __init__(
        self,
        num_items: int,
        profits: list[int],
        weights: list[int],
        capacity: int,
        optimal_items: list[int] | None = None,
        optimal_profit: int | None = None,
    ):
        self.num_items = num_items

        self.profits = profits
        self.weights = weights

        self.capacity = capacity
        self.complement_capacity = sum(weights) - capacity

        self.optimal_items = optimal_items
        if optimal_profit is None and optimal_items is not None:
            self.optimal_profit = sum(profits[i] for i in optimal_items)
        else:
            self.optimal_profit = optimal_profit

        if optimal_items is not None:
            self.complement_optimal_items = [
                i for i in range(num_items) if i not in optimal_items
            ]
        if self.optimal_profit:
            self.complement_optimal_profit = sum(self.profits) - self.optimal_profit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KnapsackInstance):
            return NotImplemented
        return (
            self.profits == other.profits
            and self.weights == other.weights
            and self.capacity == other.capacity
        )

    def evaluate(
        self,
        solution: KnapsackSolution,
        variant: str = "standard",
        verbose: bool = False,
    ) -> Result:
        if variant not in {"standard", "inverted"}:
            raise ValueError('Variant must be either "standard" or "inverted"')

        standard = variant == "standard"

        # check for validity
        if len(set(solution.selected_items)) != len(solution.selected_items):
            return IncompatibleFormatResult("Solution contains duplicate items.")
        if any(
            not 0 <= item_num < self.num_items for item_num in solution.selected_items
        ):
            return IncompatibleFormatResult("Solution contains invalid items.")

        selected_weights = [
            self.weights[object_id] for object_id in solution.selected_items
        ]
        if (standard and sum(selected_weights) > self.capacity) or (
            not standard and sum(selected_weights) < self.complement_capacity
        ):
            return ErroneousResult(
                f"Selected items are too {'heavy' if standard else 'light'}. ({sum(selected_weights)} {'>' if standard else '<'} {self.capacity if standard else self.complement_capacity})"
            )

        # result is valid, compare it to optimum
        total_value = sum(
            [self.profits[object_id] for object_id in solution.selected_items]
        )

        optimum = self.optimal_value(variant)

        if verbose:
            print(f"Found profit {total_value}, vs optimum {optimum}")

        if total_value == optimum:
            return OptimalResult(str(solution), total_value)
        else:
            return SuboptimalResult(str(solution), total_value)

    def optimal_value(self, variant: str = "standard") -> int:
        match variant:
            case "standard":
                if self.optimal_profit is None:
                    raise RuntimeError("Optimal value not known/provided")
                return self.optimal_profit
            case "inverted":
                if self.complement_optimal_profit is None:
                    self.complement_optimal_profit = (
                        sum(self.profits) - self.optimal_value()
                    )
                return self.complement_optimal_profit
            case _:
                raise ValueError('Variant must be either "standard" or "inverted"')

    def reasonable_encoding(self) -> str:
        capacity = self.capacity
        profit_weight_pairs = [
            str((self.profits[i], self.weights[i])).replace(" ", "")
            for i in range(self.num_items)
        ]
        return f"{capacity}{''.join(profit_weight_pairs)}"


@register("knapsack-loader")
class KnapsackLoader(BaseLoader[KnapsackInstance]):
    @staticmethod
    def load(problem_path: str, solution_path: str | None) -> KnapsackInstance:
        num_items: int = 0
        profits: list[int] = []
        weights: list[int] = []
        capacity: int = 0

        with open(problem_path, "r") as f:
            for line in f:
                if line.strip():  # skip empty lines
                    if not num_items:
                        num_items = int(line)
                    elif len(profits) < num_items:
                        _, profit, weight = line.split()
                        profits.append(int(profit))
                        weights.append(int(weight))
                    else:
                        capacity = int(line)

        if not all([num_items, profits, weights, capacity]):
            raise ValueError("Problem file does not specify all information correctly")

        inst = KnapsackInstance(num_items, profits, weights, capacity)

        if solution_path is not None:
            with open(solution_path, "r") as f:
                for line in f:
                    if inst.optimal_profit is None:
                        inst.optimal_profit = int(line)
                    elif inst.optimal_items is None:
                        inst.optimal_items = extract_csloi(line)
                    else:
                        raise ValueError("Unexpected line in solution file")

        return inst

    @staticmethod
    def store(instance: KnapsackInstance, folder: Path, header: str = "") -> None:
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / "problem.in", "w") as f:
            if header:
                f.write(header + "\n")
            f.write(f"{instance.num_items}\n\n")
            for i in range(instance.num_items):
                f.write(f"{i + 1} {instance.profits[i]} {instance.weights[i]}\n")
            f.write(f"\n{instance.capacity}\n")

        if instance.optimal_profit is not None and instance.optimal_items is not None:
            with open(folder / "solution.sol", "w") as f:
                f.write(str(instance.optimal_profit) + "\n")
                f.write(", ".join(map(str, instance.optimal_items)) + "\n")
