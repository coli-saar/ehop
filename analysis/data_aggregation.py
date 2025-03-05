import sys

import pandas as pd

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import (
    ABBREVS,
    COSTUME_LIST,
    GREEDY_STRAT_LIST,
    LLM_FULL_NAMES,
    LLMS,
    PROBLEMS,
    PROMPTING_STRATEGIES,
    VARIANTS,
)
from utils.plotting_utils import load_df


def master_df(dataset: str, var_dis_condition: bool = False) -> pd.DataFrame:
    if dataset not in {"random", "hard"}:
        raise ValueError("Dataset must be either 'random' or 'hard'.")

    dfs = []

    for problem, abbreviation, greedy_types in zip(
        PROBLEMS, ABBREVS, GREEDY_STRAT_LIST
    ):
        for llm, llm_name in zip(LLMS, LLM_FULL_NAMES):
            df = load_df(
                f"data/results/{problem}/{problem.replace('_','-')}-llm/{llm.lower()}/{dataset}_dataset/consolidated.csv",
                problem_prefix="in_house",
                categoricals={
                    "Costume": COSTUME_LIST,  # COSTUME_DICT[abbreviation],
                    "Variant": VARIANTS,
                    "Prompting Strategy": PROMPTING_STRATEGIES,
                },
            )

            df["Problem"] = abbreviation.upper()
            df["LLM"] = llm
            df["Disguised"] = df["Costume"] != "textbook"

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

            # greedy_dfs = [
            #     load_df(
            #         f"data/results/{problem}/{problem.replace('_','-')}-greedy/random_dataset/{strat}.csv",
            #         problem_prefix="in_house",
            #     )
            #     for strat in greedy_types
            # ]

            dfs.append(df)

    full_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    return full_df


if __name__ == "__main__":
    hard_df = master_df("hard")
    print(len(hard_df))
