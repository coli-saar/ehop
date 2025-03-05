import random
import sys

import networkx as nx

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import BaseGenerator, BaseLoader, BaseSolver
from base.results import OptimalResult
from problems.traveling_salesman.alt import TravelingSalesmanGreedy
from problems.traveling_salesman.model import (
    TravelingSalesmanInstance,
    TravelingSalesmanLoader,
    TravelingSalesmanSolution,
    invert_tsp_graph,
)
from problems.traveling_salesman.symbolic import TravelingSalesmanILP


class TravelingSalesmanGenerator(
    BaseGenerator[
        TravelingSalesmanInstance,
        BaseSolver[TravelingSalesmanSolution, TravelingSalesmanInstance],
        BaseLoader[TravelingSalesmanInstance],
    ],
):
    problem_folder = "traveling_salesman"
    scale_descriptor = "cities"

    def generate(self, scale: int, solve: bool = True) -> TravelingSalesmanInstance:
        n = scale

        g = nx.complete_graph(n)
        graph = nx.relabel_nodes(g, {i: i + 1 for i in range(n)})

        nx.set_edge_attributes(
            graph,
            {edge: random.randint(1, n**2) for edge in graph.edges()},
            "weight",
        )

        shift = random.randint(1, n)

        minimum_ordering, maximum_ordering = None, None

        if solve:
            minimum_solution = self.solver.solve(TravelingSalesmanInstance(graph))
            minimum_ordering = minimum_solution.ordering

            maximum_solution = self.solver.solve(
                TravelingSalesmanInstance(invert_tsp_graph(graph))
            )
            maximum_ordering = maximum_solution.ordering

        return TravelingSalesmanInstance(
            graph,
            minimum_ordering=minimum_ordering,
            maximum_ordering=maximum_ordering,
            inversion_shift=shift,
        )

    def generate_multiple(
        self,
        scale: int,
        num_instances: int,
        distribution: str = "random",
        solve: bool = True,
        verbose: bool = False,
    ) -> list[TravelingSalesmanInstance]:
        instances: list[TravelingSalesmanInstance] = []
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
            case _:
                raise ValueError(
                    f"Unrecognized/unsupported distribution: {distribution}"
                )
        return instances


if __name__ == "__main__":
    demo_inst = TravelingSalesmanLoader().load(
        "data/problem_instances/traveling_salesman/demo/problem.tsp",
        "data/problem_instances/traveling_salesman/demo/solution.sol",
    )
    generator = TravelingSalesmanGenerator(
        TravelingSalesmanILP(), TravelingSalesmanLoader(), TravelingSalesmanGreedy()
    )
    generator.generate_and_store(
        [4, 5, 6, 7, 8, 9],
        25,
        "random non-greedy",
        to_exclude=[demo_inst],
        verbose=True,
    )
