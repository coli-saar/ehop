# This file checks that the consolidated_results files have the correct content by comparing them to the original files.
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import (
    ABBREVS,
    COSTUME_DICT,
    DATASETS,
    LLMS,
    PROBLEMS,
    PROMPTING_STRATEGIES,
    VARIANTS,
)

KEY_COLUMNS = ["Problem Name", "Costume", "Variant", "Prompting Strategy"]

for problem, abbrev in zip(PROBLEMS, ABBREVS):
    print(abbrev.upper())
    for dataset in DATASETS:
        print(dataset)
        for llm in LLMS:
            print(llm)

            results_folder = f"data/results/{problem}/{problem.replace('_', '-')}-llm/{llm.lower()}/{dataset}_dataset/"

            original_files = [
                results_folder + f"{name}.csv"
                for name in [
                    "",
                    "only_shots",
                    "no_shots",
                    "no_shots_part_1",
                    "no_shots_part_2",
                    "no_shots_part_3",
                    "redos",
                    "shifted",
                    "new_ilp_lp",
                    "new_ilp_python",
                    "no_ilp",
                    "ilp_lp",
                    "ilp_python",
                ]
                if Path(results_folder + f"{name}.csv").exists()
            ]

            df = (
                pd.concat(
                    (pd.read_csv(f, float_precision="high") for f in original_files),
                    ignore_index=True,
                )
                .drop_duplicates(
                    subset=KEY_COLUMNS,
                    keep="last",
                )
                .sort_values(by=KEY_COLUMNS)
                .reset_index(drop=True)
            )

            if abbrev == "tsp":
                df = df[df["Variant"] != "maximize"]

            df.reset_index(drop=True, inplace=True)

            df["Summary Value"] = df["Summary Value"].astype(pd.Int32Dtype())
            df["Costume"] = pd.Categorical(
                df["Costume"], categories=COSTUME_DICT[abbrev], ordered=True
            )
            df["Variant"] = pd.Categorical(
                df["Variant"],
                categories=VARIANTS,
                ordered=True,
            )
            df["Prompting Strategy"] = pd.Categorical(
                df["Prompting Strategy"], categories=PROMPTING_STRATEGIES, ordered=True
            )

            print(
                "Are you sure you want to create new consolidated.csv files? If so, comment out this line and the exit below it."
            )
            exit()

            df.to_csv(results_folder + "consolidated.csv", index=False, na_rep="None")

            print("Saved to", results_folder + "consolidated.csv")
    print()
