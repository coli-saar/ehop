import sys

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import BaseGenerator, BaseInstance, BaseSolver
from base.results import OptimalResult
from problems.graph_coloring.generator import GraphColoringGenerator
from problems.graph_coloring.model import GraphColoringLoader
from problems.graph_coloring.symbolic import GraphColoringBrute, GraphColoringILP
from problems.knapsack.generator import KnapsackGenerator
from problems.knapsack.model import KnapsackLoader
from problems.knapsack.symbolic import KnapsackBrute, KnapsackORTools
from problems.traveling_salesman.generator import TravelingSalesmanGenerator
from problems.traveling_salesman.model import TravelingSalesmanLoader
from problems.traveling_salesman.symbolic import (
    TravelingSalesmanBrute,
    TravelingSalesmanILP,
)

INSTANCES_PER_SCALE = 10

problems = ["Graph Coloring", "Knapsack", "Traveling Salesman"]

generators: list[BaseGenerator] = [
    GraphColoringGenerator(GraphColoringBrute(), GraphColoringLoader()),
    KnapsackGenerator(KnapsackBrute(), KnapsackLoader()),
    TravelingSalesmanGenerator(TravelingSalesmanBrute(), TravelingSalesmanLoader()),
]

solvers: list[BaseSolver] = [
    GraphColoringILP(),
    KnapsackORTools(),
    TravelingSalesmanILP(),
]

scale_ranges = [(5, 12), (5, 20), (5, 11)]

for problem, generator, solver, scale_range in zip(
    problems, generators, solvers, scale_ranges
):
    for scale in range(*scale_range):
        print(f"Generating instances of scale {scale}...", end="\r")
        passed = True
        failures: list[BaseInstance] = []
        for i, inst in enumerate(
            generator.generate_multiple(
                scale, INSTANCES_PER_SCALE, "random", solve=True
            )
        ):
            print(f"Testing scale {scale}, instance {i + 1}...".ljust(40), end="\r")
            solution = solver.solve(inst)

            evaluation = inst.evaluate(solution)

            if not isinstance(evaluation, OptimalResult):
                passed = False
                failures.append(inst)
    print(
        f"{problem} checks: {'PASSED' if passed else 'FAILED ' + str([x.reasonable_encoding() for x in failures])}".ljust(
            40
        )
    )
