import importlib
import pkgutil
import random
import string
import sys
from pathlib import Path
from typing import Any, Iterable

import _jsonnet  # type: ignore
from funcy import omit
from pydantic import BaseModel, ConfigDict

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import BaseInstance, BaseLoader, BaseSolver

_class_registry = {}


def register(id):
    """Registers a class for future instantiation based on config input."""

    def decorator(clazz):
        _class_registry[id] = clazz
        return clazz

    return decorator


def instantiate_registered(data: dict):
    """Instantiates a registered class based on its id."""
    class_id = data.pop("id", None)

    if class_id is not None and class_id in _class_registry:
        return _class_registry[class_id](**omit(data, ["id"]))
    else:
        raise LookupError(f"Class id {class_id} not registered")


class ConfigProblemInstance(BaseModel):
    """A class to store the paths to a problem instance and its solution."""

    model_config = ConfigDict(extra="ignore")

    problem_path: str
    solution_path: str


def get_all_problem_instances(path: str) -> list[ConfigProblemInstance]:
    """Recursively extracts all problem instances from a directory."""
    problem_instances = []

    def extract_instance(directory: Path) -> None:
        paths = list(directory.iterdir())
        if len(paths) == 2 and all(p.is_file() for p in paths):
            problem_path = [f for f in paths if "problem" in f.name][0]
            solution_path = [f for f in paths if "solution" in f.name][0]
            problem_instances.append(
                ConfigProblemInstance(
                    problem_path=str(problem_path),
                    solution_path=str(solution_path),
                )
            )
        else:
            for path in paths:
                if path.is_dir():
                    extract_instance(path)

    extract_instance(Path(path))

    return problem_instances


class Config(BaseModel):
    """A class for the configuration being used to run an experiment."""

    model_config = ConfigDict(extra="ignore")

    id: str
    problem_type: str
    solver: dict[str, str]
    loader: dict[str, str]
    variants: list[str] = ["standard"]
    costumes: list[str] = ["textbook"]
    prompting_strategies: list[str] = ["zero_shot"]
    data: list[ConfigProblemInstance] | str

    @staticmethod
    def load(config_filename) -> "Config":
        """Construct a Config object from a Jsonnet file."""
        return Config.model_validate_json(_jsonnet.evaluate_file(config_filename))

    def get_solver(self) -> BaseSolver:
        solver = instantiate_registered(self.solver)
        if not isinstance(solver, BaseSolver):
            raise ValueError("Solver must be an instance of BaseSolver")
        return solver

    def get_loader(self) -> BaseLoader:
        loader = instantiate_registered(self.loader)
        if not isinstance(loader, BaseLoader):
            raise ValueError("Loader must be an instance of BaseLoader")
        return loader

    def get_data(
        self,
    ) -> Iterable[tuple[BaseInstance, str, str, str | None, str | None]]:
        """
        Returns a generator that yields instances, their problem names,
        and formats (for LLM solvers, and only when specified in the config).
        """
        if type(self.data) is str:
            self.data = get_all_problem_instances(self.data)
        if type(self.data) is not list:
            raise ValueError(
                f"Data should now be a list of ConfigProblemInstance objects (got {type(self.data)})"
            )

        loader = self.get_loader()

        return (
            (
                loader.load(
                    problem_path=inst.problem_path, solution_path=inst.solution_path
                ),
                Path(inst.problem_path).parent.stem,
                variant,
                costume,
                strategy,
            )
            for inst in self.data
            for variant in self.variants
            for costume in self.costumes
            for strategy in self.prompting_strategies
        )


def import_submodules(package_name):
    """Imports all submodules of a module, recursively."""  # Modified from https://stackoverflow.com/a/25083161/19048626
    package = sys.modules[package_name]
    for module_info in pkgutil.walk_packages(package.__path__):
        print(module_info)
        importlib.import_module(package_name + "." + module_info.name)


def csv_stringify(data: Any) -> str:
    """Converts data to a string that can be written as a single element/cell of a CSV file."""
    s = str(data).replace('"', '""').replace("\n", r"\n")
    return f'"{s}"' if "," in s else s


def random_id(size=8, chars=string.ascii_letters + string.digits):
    """Generates a random id string."""  # Modified from here: https://stackoverflow.com/a/2257449/19048626
    return "".join(random.choice(chars) for _ in range(size))


def csv_to_typst(csv: str, header_rows: int, row_label_counts: list[int]) -> str:
    """Converts a CSV string to a Typst table."""
    data: list[list[str]] = [row.split(",") for row in csv.split("\n")[:-1]]
    output = f"table(\n\ttable.header("

    hline, vline = "table.hline(stroke: 1.5pt),\n\t", "table.vline(stroke: 1.5pt),"

    # handle header row
    for row in data[:header_rows]:
        previous, count = None, 0
        for i, cell in enumerate(row):
            if cell == previous:
                count += 1
            else:
                if count > 1:
                    output += f"table.cell(colspan:{count}, [{previous}]),{vline}"
                elif previous is not None:
                    output += f"[{previous}],"
                previous, count = cell, 1
        if count > 1:
            output += f"table.cell(colspan:{count}, [{previous}]),"
        elif previous is not None:
            output += f"[{previous}],"
        previous, count = cell, 1
        output += "\n\t"
    output += "),"

    # handle data rows
    for i, row in enumerate(data[header_rows:]):
        if 0 in [i % x for x in row_label_counts]:
            output += hline
        for j, cell in enumerate(row):
            if j < len(row_label_counts):
                if i % row_label_counts[j] == 0:
                    output += f"table.cell(rowspan:{row_label_counts[j]}, [{cell}]),"
            else:
                output += f"[{cell}],"

        output += "\n\t"

    output += ")\n"

    return output
