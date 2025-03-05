import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import (
    ABBREVS,
    DATASETS,
    GREEDY_STRAT_LIST,
    LLMS,
    PROBLEMS,
)
from analysis.data_aggregation import master_df
from utils.plotting_utils import get_line_plot_data, load_df

comparison_sets: dict[str, list[str]] = {
    "all": [
        "standard textbook",
        "inverted textbook",
        "standard costumed",
        "inverted costumed",
    ],
    "isolated_deviations": [
        "standard textbook",
        "inverted textbook",
        "standard costumed",
    ],
    "textbook_inversion": ["standard textbook", "inverted textbook"],
    "costumed_inversion": ["standard costumed", "inverted costumed"],
    "standard_costuming": ["standard textbook", "standard costumed"],
    "inverted_costuming": ["inverted textbook", "inverted costumed"],
}

for dataset in DATASETS:
    df = master_df(dataset, var_dis_condition=True)

    df = df[df["Prompting Strategy"] != "ilp_lp"]

    # Greedy results
    greedy_results: dict[str, dict[str, float]] = {problem: {} for problem in PROBLEMS}

    for problem in PROBLEMS:
        result_folder = Path(
            f"data/results/{problem}/{problem.replace('_', '-')}-greedy/{dataset}_dataset"
        )
        for result_file in result_folder.glob("*.csv"):
            greedy_results[problem][result_file.stem] = (  # type: ignore
                load_df(result_file)
                .groupby("Result Type", observed=False)
                .size()
                .fillna(0)
                / (1 if dataset == "hard" and problem == "graph_coloring" else 1.5)
            ).round(1)["OPTIMAL"]

    greedy_strats = (
        GREEDY_STRAT_LIST
        if dataset == "random"
        else [["random_sequential"], ["value"], []]
    )

    greedy_accs = [
        greedy_results[problem][greedy_strats[i][0]]
        for i, problem in enumerate(PROBLEMS)
        if greedy_strats[i]
    ]

    for llm in LLMS:
        llm_df = df[df["LLM"] == llm]

        for comparison_name, comparison_set in comparison_sets.items():
            comparison_df = llm_df[llm_df["Condition"].isin(comparison_set)]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.subplots(nrows=3, ncols=1)

            num_conditions = len(comparison_set)

            for i, abbrev in enumerate(ABBREVS):
                problem_df = comparison_df[comparison_df["Problem"] == abbrev.upper()]

                grouped_df: pd.DataFrame = (
                    (
                        problem_df.groupby(
                            ["Condition", "Prompting Strategy", "Result Type"],
                            observed=True,
                        )
                        .size()
                        .unstack()
                        .fillna(0)
                    )
                    .apply(lambda row: row / row.sum() * 100, axis=1)
                    .round(1)["OPTIMAL"]
                    .unstack("Prompting Strategy")
                ).reset_index()

                width = 1 / (num_conditions + 1)
                multiplier = 0

                x = np.arange(4)

                for _, row in grouped_df.iterrows():
                    offset = width * multiplier
                    rects = ax[i].bar(
                        x + offset,
                        (
                            row["one_shot"],
                            row["zero_shot_cot"],
                            row["one_shot_cot"],
                            row["ilp_python"],
                        ),
                        width,
                        label=row["Condition"],
                    )
                    ax[i].bar_label(rects, padding=0.2)
                    multiplier += 1

                if len(greedy_accs) > i:
                    ax[i].axhline(
                        y=greedy_accs[i],
                        color="red",
                        linestyle="--",
                        label="Greedy",
                    )

                ax[i].set_ylim((0, 100))
                ax[i].set_ylabel(abbrev.upper())

                if i < 2:
                    ax[i].set_xticks([])
                    ax[i].set_xticklabels([])

            ax[0].legend()

            ax[2].set_xticks(x + (width * ((num_conditions - 1) / 2)))
            ax[2].set_xticklabels(
                ["One-Shot", "Zero-Shot COT", "One-Shot COT", "ILP Python"]
            )

            fig.suptitle(f"{llm} ({dataset.title()} Dataset)")

            fig.tight_layout()
            Path(f"analysis/plots/condition_comparison/").mkdir(
                parents=True, exist_ok=True
            )
            fig.savefig(
                f"analysis/plots/condition_comparison/{dataset}_{llm}_{comparison_name}.png"
            )
            plt.close(fig)
