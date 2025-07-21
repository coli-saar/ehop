import sys

import pandas as pd

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import (
    ABBREVS,
    COSTUME_LIST,
    GREEDY_STRAT_LIST,
    LLMS,
    PROBLEMS,
    PROMPTING_STRATEGIES,
    RESULT_TYPES,
    SCALE_WORDS,
    VARIANTS,
)
from base.llm_solver import BaseLLMSolver
from base.problem_structures import BaseInstance, BaseLoader, BaseSolver
from base.results import classify_result
from problems.graph_coloring.llm import GraphColoringLLM
from problems.graph_coloring.model import GraphColoringLoader
from problems.graph_coloring.symbolic import GraphColoringILP
from problems.knapsack.llm import KnapsackLLM
from problems.knapsack.model import KnapsackLoader
from problems.knapsack.symbolic import KnapsackORTools
from problems.traveling_salesman.llm import TravelingSalesmanLLM
from problems.traveling_salesman.model import TravelingSalesmanLoader
from problems.traveling_salesman.symbolic import TravelingSalesmanILP
from utils.llm_output_utils import extract_csloi
from utils.plotting_utils import load_df

recognition_terms = {
    "graph_coloring": ["graph coloring", "chromatic number"],
    "knapsack": ["knapsack"],
    "traveling_salesman": ["traveling salesman", "graph"],
}


def master_df(
    dataset: str,
    problem_inclusion_flags: list[bool] = [True, True, True],
    llms: list[str] = LLMS,
    strategies: list[str] = PROMPTING_STRATEGIES,
    filename: str = "consolidated.csv",
    var_dis_condition: bool = False,
    recognition_cols: list[str] = [],
    greedy_data: bool = False,
    inverted_solutions: bool = False,
) -> pd.DataFrame:
    if dataset == "both":
        kwargs = {k: v for k, v in locals().items() if k != "dataset"}
        random = master_df("random", **kwargs)
        hard = master_df("hard", **kwargs)

        random["Dataset"] = "Random"
        hard["Dataset"] = "Hard"

        return pd.concat([random, hard], ignore_index=True)
    elif dataset not in {"random", "hard"}:
        raise ValueError("Dataset must be either 'random' or 'hard'.")

    dfs = []

    for flag, problem, abbreviation, greedy_types in zip(
        problem_inclusion_flags, PROBLEMS, ABBREVS, GREEDY_STRAT_LIST
    ):
        if not flag:
            continue

        problem_dfs = []
        for llm in llms:
            df = load_df(
                f"data/results/{problem}/{problem.replace('_','-')}-llm/{llm.lower().replace(' ', '')}/{dataset}_dataset/{filename}",
                problem_prefix="in_house",
                categoricals={
                    "Costume": COSTUME_LIST,  # COSTUME_DICT[abbreviation],
                    "Variant": VARIANTS,
                    "Prompting Strategy": strategies,
                },
            )

            df["Problem"] = abbreviation.upper()
            df["LLM"] = llm
            df["Disguised"] = df["Costume"] != "textbook"

            for col in recognition_cols:
                df["Recognized_" + col.title()] = df[col].apply(
                    lambda resp: any(
                        term in str(resp).lower() for term in recognition_terms[problem]
                    )
                )

            if var_dis_condition:
                df["Condition"] = pd.Categorical(
                    df.apply(
                        lambda row: f"{'standard' if row['Variant'] == 'standard' else 'inverted'} {'costumed' if row['Disguised'] else 'textbook'}",
                        axis=1,
                    ),
                    categories=[
                        "standard textbook",
                        "inverted textbook",
                        "standard costumed",
                        "inverted costumed",
                    ],
                    ordered=True,
                )

            if greedy_data:
                for i, strat in enumerate(greedy_types):
                    greedy_df = (
                        load_df(
                            f"data/results/{problem}/{problem.replace('_','-')}-greedy/{dataset}_dataset/{strat}.csv",
                            problem_prefix="in_house",
                        )[["Problem Name", "Solution/Error"]]
                        .rename(columns={"Solution/Error": f"Greedy Solution {str(i)}"})
                        .set_index("Problem Name")
                    )

                    df = df.join(greedy_df, on="Problem Name")

                    df[f"Greedy Match {str(i)}"] = (
                        df["Solution/Error"] == df[f"Greedy Solution {str(i)}"]
                    )

                df["Greedy Match Any"] = pd.Categorical(
                    df[
                        [f"Greedy Match {str(i)}" for i in range(len(greedy_types))]
                    ].any(axis=1),
                    categories=[True, False],
                    ordered=True,
                )
            problem_dfs.append(df)

        if inverted_solutions:
            match abbreviation:
                case "gcp":
                    solver: BaseSolver = GraphColoringILP()
                    loader: BaseLoader = GraphColoringLoader()
                    llm_solver: BaseLLMSolver = GraphColoringLLM()
                    extension = "col"
                case "ksp":
                    solver = KnapsackORTools()
                    loader = KnapsackLoader()
                    llm_solver = KnapsackLLM()
                    extension = "in"
                case "tsp":
                    solver = TravelingSalesmanILP()
                    loader = TravelingSalesmanLoader()
                    llm_solver = TravelingSalesmanLLM()
                    extension = "tsp"
                case _:
                    raise ValueError(f"Unknown problem abbreviation: {abbreviation}")
            scale_word = SCALE_WORDS[abbreviation]

            def get_inverted_inst(problem_name: str) -> BaseInstance:
                scale_num = int(problem_name.split("_")[2])
                folder = f"data/problem_instances/{problem}/in_house/{dataset}_dataset/{scale_num}_{scale_word}/{problem_name}/"
                inst = loader.load(
                    folder + f"problem.{extension}", folder + "solution.sol"
                )
                return inst.inverted_inst(solver)  # type: ignore

            inv_insts_df = {
                name: get_inverted_inst(name)
                for name in df["Problem Name"].unique().tolist()
            }

            def get_inverted_result(row: pd.Series) -> tuple[str, int | None, str]:
                if row["Result Type"] == "ILPFAILURE":
                    return "ILPFAILURE", None, row["Solution/Error"]
                problem_name = row["Problem Name"]
                inv_inst = inv_insts_df[problem_name]
                response = row["Response"]
                if response.startswith("("):
                    response = response.split("', '")[-1][:-2]
                loi = extract_csloi(
                    response if isinstance(response, str) else response[-1]
                )
                solution = llm_solver.extract_solution(inv_inst, ("",), "", None, loi)
                result = inv_inst.evaluate(solution)
                return classify_result(result)

            for df in problem_dfs:
                df[
                    [
                        "Inverted Result Type",
                        "Inverted Solution/Error",
                        "Inverted Summary Value",
                    ]
                ] = df.apply(get_inverted_result, axis=1, result_type="expand")
                df["Inverted Result Type"] = pd.Categorical(
                    df["Inverted Result Type"],
                    categories=RESULT_TYPES,
                    ordered=True,
                )

        dfs += problem_dfs

    full_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    return full_df


if __name__ == "__main__":
    df = master_df("both", inverted_solutions=True)
    print(len(df))
    print(df.columns)
