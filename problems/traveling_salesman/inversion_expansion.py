# This file goes through the existing TSP instances and expands them to include data for the shifted inversion variant

import random
import sys
from pathlib import Path

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from problems.traveling_salesman.model import (
    TravelingSalesmanInstance,
    TravelingSalesmanLoader,
)

loader = TravelingSalesmanLoader()

instance_paths: list[str] = []

random_dataset_directory = Path(
    "data/problem_instances/traveling_salesman/in_house/random_dataset"
)


def extract_instance(directory: Path) -> None:
    paths = list(directory.iterdir())
    if len(paths) == 2 and all(p.is_file() for p in paths):
        instance_paths.append(str(directory))
    else:
        for path in paths:
            if path.is_dir():
                extract_instance(path)


extract_instance(random_dataset_directory)

instance_paths.append("data/problem_instances/traveling_salesman/demo")

for instance_path in instance_paths:
    inst = loader.load(instance_path + "/problem.tsp", instance_path + "/solution.sol")

    if inst.minimum_ordering is None or inst.maximum_ordering is None:
        raise ValueError(
            "The instance is missing a value for at least one of the optimal orderings."
        )

    shift = random.randint(1, inst.graph.number_of_nodes())

    inst_with_shift = TravelingSalesmanInstance(
        graph=inst.graph.copy(),
        minimum_ordering=inst.minimum_ordering.copy(),
        maximum_ordering=inst.maximum_ordering.copy(),
        inversion_shift=shift,
    )

    loader.store(inst_with_shift, Path(instance_path))
