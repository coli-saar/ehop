import csv
import importlib
import os
import sys
from pathlib import Path

from base.llm_solver import BaseLLMSolver
from base.results import (
    ErroneousResult,
    ILPException,
    IncompatibleFormatResult,
    OptimalResult,
    SuboptimalResult,
)

# must import these problems so the classes get registered
from problems import graph_coloring, knapsack, traveling_salesman
from utils.utils import Config, csv_stringify

# the following code further registers classes from the problem modules
problems = [graph_coloring, knapsack, traveling_salesman]
submodule_names = ["model", "llm", "symbolic", "alt"]
for problem in problems:
    for submodule_name in submodule_names:
        importlib.import_module(f"{problem.__name__}.{submodule_name}")


def already_tested(
    csv_filepath: Path,
    problem_name: str,
    costume: str | None,
    variant: str | None,
    prompting_strategy: str | None,
):
    """Checks whether a given problem has already been tested with the given costume, variant, and prompting strategy."""
    with open(csv_filepath, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] == problem_name and (
                (costume is None and line[1] == variant and prompting_strategy is None)
                or (
                    line[1] == costume
                    and line[2] == variant
                    and line[3] == prompting_strategy
                )
            ):
                return True
    return False


def main() -> None:
    """Runs experiments given a configuration file provided as a command line argument."""
    config = Config.load(sys.argv[1])

    here = Path(__file__).parent
    result_dir = here / f"data/results/{config.problem_type}/{config.solver['id']}/"

    # create the directory if it doesn't exist
    result_dir.mkdir(parents=True, exist_ok=True)

    solver_model = config.solver.get("model")
    result_path = result_dir / (
        (
            solver_model.split("/")[-1]
            if solver_model
            else config.solver.get("strategy") or "results"
        )
        + ".csv"
    )

    solver = config.get_solver()

    if not result_path.is_file() or os.stat(result_path).st_size == 0:
        # results csv file is new/empty, so add header
        columns = [
            "Problem Name",
            "Variant",
            "Size",
            "Optimal Value",
            "Time (s)",
            "Result Type",
            "Summary Value",
            "Solution/Error",
        ]
        if isinstance(solver, BaseLLMSolver):
            columns.insert(1, "Costume")
            columns.insert(3, "Prompting Strategy")
            columns += ["Prompt", "Response"]
        with open(result_path, "w") as f:
            f.write(f"{','.join(columns)}\n")

    for inst, problem_name, variant, costume, prompting_strategy in config.get_data():
        print(
            f"{problem_name.ljust(16)} | {variant} | {costume.ljust(20) + ' | ' if costume else ''}"
            + f"{prompting_strategy.ljust(16) + ' | ' if prompting_strategy else ''}",
            end="",
        )
        # # Uncomment the following code block to skip problems that have already been tested
        # if already_tested(
        #     result_path, problem_name, costume, variant, prompting_strategy
        # ):
        #     print("Skip (already tested)")
        #     continue

        size = inst.size()
        optimal_val = inst.optimal_value(variant)

        if isinstance(solver, BaseLLMSolver):
            if costume and variant:
                solver.set_template(costume, variant)
            if prompting_strategy:
                solver.set_prompting_strategy(prompting_strategy)
        elif variant:
            solver.set_variant(variant)

        try:
            solution, time = solver.timed_solve(inst)
        except ILPException as e:
            print(f"ILP Failure: {str(e)}")
            time = None
            result_type, summary_val, sol_err = "ILPFAILURE", None, str(e)
            if isinstance(solver, BaseLLMSolver):
                prompt, response = e.prompts, e.responses
        else:
            if isinstance(solver, BaseLLMSolver):
                prompt, response = solution.prompt, solution.response
            evaluation = inst.evaluate(solution, variant=variant, verbose=True)

            # classify result
            match evaluation:
                case OptimalResult(solution_string, summary_value):
                    result_type, summary_val, sol_err = (
                        "OPTIMAL",
                        summary_value,
                        solution_string,
                    )
                case SuboptimalResult(solution_string, summary_value):
                    result_type, summary_val, sol_err = (
                        "SUBOPTIMAL",
                        summary_value,
                        solution_string,
                    )
                case ErroneousResult(reason):
                    print(f"Erroneous: {reason}")
                    result_type, summary_val, sol_err = "ERRONEOUS", None, reason
                case IncompatibleFormatResult(reason):
                    print(f"Incompatible Format: {reason}")
                    result_type, summary_val, sol_err = "INCOMPATIBLE", None, reason
                case _:
                    raise ValueError("Unrecognized result type")

        values = [
            problem_name,
            variant,
            size,
            optimal_val,
            time,
            result_type,
            summary_val,
            sol_err,
        ]
        if isinstance(solver, BaseLLMSolver):
            values.insert(1, costume)
            values.insert(3, prompting_strategy)
            values += [prompt, response]

        with open(result_path, "a+", encoding="utf-8") as f:
            f.write(f"{','.join(csv_stringify(v) for v in values)}\n")


if __name__ == "__main__":
    main()
