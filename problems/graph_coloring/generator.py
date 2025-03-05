import sys

import networkx as nx

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import BaseGenerator, BaseLoader, BaseSolver
from base.results import OptimalResult
from problems.graph_coloring.alt import GraphColoringGreedy
from problems.graph_coloring.model import (
    GraphColoringInstance,
    GraphColoringLoader,
    GraphColoringSolution,
)
from problems.graph_coloring.symbolic import GraphColoringILP


class GraphColoringGenerator(
    BaseGenerator[
        GraphColoringInstance,
        BaseSolver[GraphColoringSolution, GraphColoringInstance],
        BaseLoader[GraphColoringInstance],
    ]
):
    problem_folder = "graph_coloring"
    scale_descriptor = "nodes"

    def generate(
        self, scale: int, solve: bool = True, p: float = 0.5
    ) -> GraphColoringInstance:
        g = nx.binomial_graph(scale, p)
        graph = nx.relabel_nodes(g, {i: i + 1 for i in range(scale)})
        optimal_solution = (
            self.solver.solve(GraphColoringInstance(graph=graph)) if solve else None
        )
        return GraphColoringInstance(
            graph=graph,
            chromatic_number=(
                len(set(optimal_solution.coloring)) if optimal_solution else None
            ),
            optimal_coloring=optimal_solution.coloring if optimal_solution else None,
        )

    def generate_multiple(
        self,
        scale: int,
        num_instances: int,
        distribution: str = "random",
        solve: bool = True,
        verbose: bool = False,
    ) -> list[GraphColoringInstance]:
        instances: list[GraphColoringInstance] = []
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
                p = 0.5
                step_size = 0.05
                counts = [0] * (scale - 2)
                max_count = num_instances // (scale - 2)
                while sum(counts) < num_instances:
                    if verbose:
                        print(
                            f"{len(instances)} instances generated...{' '*10}", end="\r"
                        )
                    print(
                        counts,
                        f"Progress: {sum(counts) / num_instances:.0%}",
                        f"| p={p:.1%}",
                        end="\r",
                    )
                    inst = self.generate(scale, solve, p=p)
                    if inst.chromatic_number is None:
                        raise ValueError("Chromatic number not found")
                    X: int = inst.chromatic_number
                    if X in [1, scale]:
                        continue
                    elif (
                        counts[X - 2] < max_count
                        or (
                            sum(counts) >= max_count * (scale - 2)
                            and counts[X - 2] == min(counts)
                        )
                    ) and inst not in instances:
                        instances.append(inst)
                        counts[X - 2] += 1
                    if p < 0.005:
                        p = 0.005
                    elif p > 0.995:
                        p = 0.995
                    elif counts[0] < max(counts) and X != 2:
                        p *= 1 - step_size
                    elif counts[-1] < max(counts) and X != scale - 1:
                        p *= 1 + step_size
                print(str(counts).ljust(80))
            case _:
                raise ValueError(
                    f"Unrecognized/unsupported distribution: {distribution}"
                )
        return instances


if __name__ == "__main__":
    demo_inst = GraphColoringLoader().load(
        "data/problem_instances/graph_coloring/demo/problem.col",
        "data/problem_instances/graph_coloring/demo/solution.sol",
    )
    generator = GraphColoringGenerator(
        GraphColoringILP(), GraphColoringLoader(), GraphColoringGreedy()
    )
    generator.generate_and_store(
        [6, 7, 8, 9],
        25,
        "random non-greedy",
        to_exclude=[demo_inst],
        verbose=True,
    )
