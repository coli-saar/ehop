import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import (
    ABBREVS,
    DATASETS,
    GREEDY_STRAT_LIST,
    LLM_FULL_NAMES,
    LLMS,
    PROBLEM_SCALES,
    PROBLEMS,
    SCALE_WORDS,
)
from analysis.data_aggregation import master_df
from utils.plotting_utils import get_line_plot_data, load_df

DETAILED_FLAG = False

Path("analysis/plots/scale_effect").mkdir(parents=True, exist_ok=True)

for dataset in DATASETS:

    # Greedy results
    greedy_results: dict[str, dict[str, tuple[list[float], list[float]]]] = {
        problem: {} for problem in PROBLEMS
    }

    for problem in PROBLEMS:
        result_folder = Path(
            f"data/results/{problem}/{problem.replace('_', '-')}-greedy/{dataset}_dataset"
        )
        for result_file in result_folder.glob("*.csv"):
            greedy_results[problem][result_file.stem] = get_line_plot_data(
                load_df(result_file), ["Scale", "Result Type"]
            )

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.subplots(nrows=1, ncols=3)

    full_df = master_df(dataset)

    greedy_strats = (
        GREEDY_STRAT_LIST
        if dataset == "random"
        else [["random_sequential"], ["value"], []]
    )

    for i, (problem, abbreviation, greedy_types) in enumerate(
        zip(PROBLEMS, ABBREVS, greedy_strats)
    ):
        problem_df = full_df[full_df["Problem"] == abbreviation.upper()]

        ax[i].grid(which="both", linewidth=2)

        for j, (llm, llm_name) in enumerate(zip(LLMS, LLM_FULL_NAMES)):
            df = problem_df[problem_df["LLM"] == llm]

            if DETAILED_FLAG:
                prompt_strats = ["one_shot_cot", "ilp_python"]
                sns.set_palette("tab10", n_colors=len(LLMS))  # + 1)  # +1 for llama 3.3
                for k, prompt_strat in enumerate(prompt_strats):
                    line = get_line_plot_data(
                        df[
                            (df["Prompting Strategy"] == prompt_strat)
                            & (df["Costume"] == "textbook")
                            & (df["Variant"] == "standard")
                        ],
                        groupby=["Scale", "Result Type"],
                    )
                    ax[i].plot(
                        line[0],
                        line[1],
                        "s-" if llm == "GPT" else "v-",
                        linewidth=3,
                        markersize=8,
                        color=sns.color_palette()[j],
                        label=llm
                        + " "
                        + prompt_strat.replace("_", " ")
                        .title()
                        .replace(" Shot", "-Shot")
                        .replace("Cot", "CoT")
                        .replace("Ilp", "ILP"),
                        dashes=[2] if k else (None, None),
                    )
            else:
                sns.set_palette("tab10", n_colors=len(LLMS))
                line = get_line_plot_data(
                    df[(df["Costume"] == "textbook") & (df["Variant"] == "standard")],
                    groupby=["Scale", "Result Type"],
                )
                ax[i].plot(
                    line[0],
                    line[1],
                    "o-",
                    linewidth=3,
                    markersize=8,
                    color=sns.color_palette()[j],
                    label=llm,
                )

        # # llama 3.3 section
        # df = load_df(
        #     f"data/results/{problem}/{problem.replace('_','-')}-llm/llama3.3/{dataset}_dataset/oscot_python.csv"
        # )
        # for j, prompt_strat in enumerate(prompt_strats):
        #     line = get_line_plot_data(
        #         df[
        #             (df["Prompting Strategy"] == prompt_strat)
        #             & (df["Costume"] == "textbook")
        #             & (df["Variant"] == "standard")
        #         ],
        #         groupby=["Scale", "Result Type"],
        #     )
        #     ax[i].plot(
        #         line[0],
        #         line[1],
        #         "d-" if llm == "Llama" else "o-",
        #         color=sns.color_palette()[2],
        #         label="Llama 3.3"
        #         + " | "
        #         + prompt_strat.replace("_", " ")
        #         .title()
        #         .replace(" Shot", "-Shot")
        #         .replace("Cot", "CoT")
        #         .replace("Ilp", "ILP"),
        #         dashes=[2] if j else (None, None),
        #     )

        if greedy_types:
            line = greedy_results[problem][greedy_types[0]]
            ax[i].plot(
                line[0],
                line[1],
                "o-",
                linewidth=3,
                markersize=8,
                color="black",
                label="Greedy" + ("" if dataset == "random" else " (weak)"),
            )

        ax[i].set_xlim(PROBLEM_SCALES[abbreviation]["limits"])
        ax[i].set_xticks(PROBLEM_SCALES[abbreviation]["ticks"])
        ax[i].set_xlabel(SCALE_WORDS[abbreviation].title(), fontsize=22)

        ax[i].set_ylim((-2.5, 102.5))
        if i != 0:
            ax[i].set_yticklabels([])
        ax[i].set_title(problem.replace("_", " ").title(), fontsize=24)

        ax[i].tick_params(axis="both", which="major", labelsize=22)

    ax[0].set_ylabel("Optimality Rate", fontsize=24)

    if DETAILED_FLAG:
        ax[1 if dataset == "random" else 0].legend(
            fontsize=20,
            loc="lower left" if dataset == "random" else "upper left",
            # bbox_to_anchor=(-0.02, -0.01) if dataset == "random" else None,
            fancybox=True,
            # shadow=True,
        )
    else:
        ax[2].legend(
            fontsize=20,
            loc="upper right",
            fancybox=True,
            shadow=True,
        )

    fig.tight_layout()

    fig.savefig(
        f"analysis/plots/scale_effect/{dataset.title()}_Dataset_Scale_Effect{'_Detailed' if DETAILED_FLAG else ''}.png"
    )
    fig.savefig(
        f"analysis/plots/scale_effect/{dataset.title()}_Dataset_Scale_Effect{'_Detailed' if DETAILED_FLAG else ''}.svg"
    )
    plt.close()
