import random
import sys

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import BaseGenerator, BaseLoader, BaseSolver
from base.results import OptimalResult
from problems.knapsack.alt import KnapsackGreedy
from problems.knapsack.model import KnapsackInstance, KnapsackLoader, KnapsackSolution
from problems.knapsack.symbolic import KnapsackORTools


class KnapsackGenerator(
    BaseGenerator[
        KnapsackInstance,
        BaseSolver[KnapsackSolution, KnapsackInstance],
        BaseLoader[KnapsackInstance],
    ],
):
    problem_folder = "knapsack"
    scale_descriptor = "items"

    def generate(
        self,
        scale: int,
        solve: bool = True,
    ) -> KnapsackInstance:
        n = scale
        profits = [random.randint(1, n) for _ in range(n)]
        weights = [random.randint(1, n) for _ in range(n)]
        capacity = random.randint(min(weights), sum(weights) - 1)

        inst = KnapsackInstance(
            num_items=n,
            profits=profits,
            weights=weights,
            capacity=capacity,
        )

        if solve:
            optimal_solution = self.solver.solve(inst)

            inst.optimal_items = optimal_solution.selected_items
            inst.optimal_profit = sum(
                [inst.profits[i] for i in optimal_solution.selected_items]
            )

        return inst

    def generate_multiple(
        self,
        scale: int,
        num_instances: int,
        distribution: str = "random",
        solve: bool = True,
        verbose: bool = False,
    ) -> list[KnapsackInstance]:
        instances: list[KnapsackInstance] = []
        match distribution:
            case "random":
                while len(instances) < num_instances:
                    if verbose:
                        print(
                            f"{len(instances)} instances generated...{' '*10}", end="\r"
                        )
                    inst = self.generate(scale, solve)
                    if inst not in instances:
                        instances.append(inst)
            case "random non-greedy":
                if not solve:
                    raise ValueError(
                        "Instances must be solved to determine whether they are solved greedily"
                    )
                if self.greedy_solver is None:
                    raise ValueError(
                        "Greedy solver must be provided to generate non-greedy instances"
                    )
                while len(instances) < num_instances:
                    if verbose:
                        print(
                            f"{len(instances)} instances generated...{' '*10}", end="\r"
                        )
                    inst = self.generate(scale, solve)
                    if inst not in instances:
                        greedy_solution = self.greedy_solver.solve(inst)
                        result = inst.evaluate(greedy_solution)
                        if not isinstance(result, OptimalResult):
                            instances.append(inst)
            case "uniform":
                counts = [0] * (scale - 1)
                max_count = num_instances // (scale - 1)
                while sum(counts) < num_instances:
                    if verbose:
                        print(
                            f"{len(instances)} instances generated...{' '*10}", end="\r"
                        )
                    print(
                        counts,
                        f"{sum(counts) / num_instances:.0%}",
                        end="\r",
                    )
                    inst = self.generate(scale, solve)
                    if inst.optimal_profit is None or inst.optimal_items is None:
                        raise ValueError("Optimal profit not found")
                    item_count = len(inst.optimal_items)
                    if item_count in [0, scale]:
                        continue
                    elif (
                        counts[item_count - 1] < max_count
                        or (
                            sum(counts) >= max_count * (scale - 1)
                            and counts[item_count - 1] == min(counts)
                        )
                    ) and inst not in instances:
                        instances.append(inst)
                        counts[item_count - 1] += 1
                print(str(counts).ljust(80))
            case _:
                raise ValueError(
                    f"Unrecognized/unsupported distribution: {distribution}"
                )
        return instances


if __name__ == "__main__":
    demo_inst = KnapsackLoader().load(
        "data/problem_instances/knapsack/demo/problem.in",
        "data/problem_instances/knapsack/demo/solution.sol",
    )
    generator = KnapsackGenerator(KnapsackORTools(), KnapsackLoader(), KnapsackGreedy())
    generator.generate_and_store(
        [4, 8, 12, 16, 20, 24],
        25,
        "random non-greedy",
        to_exclude=[demo_inst],
        verbose=True,
    )
