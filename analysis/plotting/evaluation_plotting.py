import sys
from pathlib import Path

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import (
    ABBREVS,
    COSTUME_DICT,
    DATASETS,
    GREEDY_STRAT_LIST,
    LLM_FULL_NAMES,
    LLMS,
    PROBLEM_SCALES,
    PROBLEMS,
    PROMPTING_STRATEGIES,
    VARIANTS,
)
from analysis.data_aggregation import master_df
from utils.plotting_utils import (
    Plotter,
    accuracy_df,
    get_diff_line_plot_data,
    get_line_plot_data,
    load_df,
)

for dataset in DATASETS:
    print(dataset.title())

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

    full_df = master_df(dataset)

    greedy_strats = (
        GREEDY_STRAT_LIST
        if dataset == "random"
        else [["random_sequential"], ["value"], []]
    )

    for problem, abbreviation, greedy_types in zip(PROBLEMS, ABBREVS, greedy_strats):
        print(f"  {abbreviation.upper()}")
        problem_df = full_df[full_df["Problem"] == abbreviation.upper()]
        plotter = Plotter(problem)
        for llm, llm_name in zip(LLMS, LLM_FULL_NAMES):
            print(f"    {llm}...", end="\r")
            Path(f"analysis/plots/{problem}/{dataset}").mkdir(
                parents=True, exist_ok=True
            )

            df = problem_df[problem_df["LLM"] == llm]

            variants = VARIANTS

            # print(accuracy_df(df))
            # print(
            #     accuracy_df(
            #         df, groupby=["Prompting Strategy", "Scale"], denominator=200
            #     )
            # )

            xticks = PROBLEM_SCALES[abbreviation]["ticks"]

            for prompting_strategy in PROMPTING_STRATEGIES:
                ps_df = df[df["Prompting Strategy"] == prompting_strategy]

                plotter.costume_variant_stack_plot(
                    ps_df,
                    COSTUME_DICT[abbreviation],
                    title_prefix=f"{abbreviation.upper()} {llm_name} {prompting_strategy.replace('_', ' ').title()}",
                    xlim=(xticks[0], xticks[-1]),
                    ylim=(0, 100),
                    xticks=xticks,
                    variants=variants,
                    filename=f"{dataset}/{abbreviation.upper()}_{llm}_{prompting_strategy.title()}_Stacked.png",
                )

                plotter.costume_variant_line_plot(
                    ps_df,
                    COSTUME_DICT[abbreviation],
                    title_prefix=f"{abbreviation.upper()} {llm_name} {prompting_strategy.replace('_', ' ').title()}",
                    extra_lines=[
                        greedy_results[problem][greedy_type]
                        for greedy_type in greedy_types
                    ],
                    extra_labels=[
                        (
                            "Greedy" + " " + greedy_type.replace("_", " ").title()
                            if greedy_type != "results"
                            else "Greedy"
                        )
                        for greedy_type in greedy_types
                    ],
                    xlim=(xticks[0] - 1, xticks[-1] + 1),
                    xticks=xticks,
                    variants=variants,
                    filename=f"{dataset}/{abbreviation.upper()}_{llm}_{prompting_strategy.title()}_Optimal.png",
                )

                plotter.costume_variant_line_plot(
                    ps_df,
                    COSTUME_DICT[abbreviation],
                    title_prefix=f"{abbreviation.upper()} {llm_name} {prompting_strategy.replace('_', ' ').title()}",
                    groupby=["Scale", "Validity"],
                    last_group_val="VALID",
                    xlim=(xticks[0] - 1, xticks[-1] + 1),
                    xticks=xticks,
                    variants=variants,
                    filename=f"{dataset}/{abbreviation.upper()}_{llm}_{prompting_strategy.title()}_Valid.png",
                )

                plotter.line_plot(
                    [
                        get_diff_line_plot_data(
                            ps_df[ps_df["Costume"] == costume],
                            ["Scale", "Result Type"],
                            value_1="standard",
                            value_2="inverted",
                        )
                        for costume in COSTUME_DICT[abbreviation]
                    ],
                    labels=[
                        costume.replace("_", " ").title()
                        for costume in COSTUME_DICT[abbreviation]
                    ],
                    title=f"{abbreviation.upper()} {llm_name} {prompting_strategy.replace('_', ' ').title()} Variant Differences (Standard - Inverted)",
                    bold_y=[0],
                    xlim=(xticks[0] - 1, xticks[-1] + 1),
                    ylim=(-100, 100),
                    xticks=xticks,
                    filename=f"{dataset}/{abbreviation.upper()}_{llm}_{prompting_strategy.title()}_Differences.png",
                )

                plotter.line_plot(
                    [
                        get_line_plot_data(
                            ps_df[
                                (
                                    (ps_df["Costume"] == "textbook")
                                    if textbook
                                    else (ps_df["Costume"] != "textbook")
                                )
                                & (ps_df["Variant"] == variant)
                            ],
                            ["Scale", "Result Type"],
                        )
                        for variant in variants
                        for textbook in [True, False]
                    ],
                    [
                        ("Textbook" if textbook else "Costume")
                        + " | "
                        + variant.title()
                        for variant in variants
                        for textbook in [True, False]
                    ],
                    extra_lines=[
                        greedy_results[problem][greedy_type]
                        for greedy_type in greedy_types
                    ],
                    extra_labels=[
                        (
                            "Greedy" + " " + greedy_type.replace("_", " ").title()
                            if greedy_type != "results"
                            else "Greedy"
                        )
                        for greedy_type in greedy_types
                    ],
                    half_dashed=True,
                    title=f"{abbreviation.upper()} {llm_name} {prompting_strategy.replace('_', ' ').title()} Textbook vs Costumes",
                    xlim=(xticks[0] - 1, xticks[-1] + 1),
                    xticks=xticks,
                    ylim=(-2.5, 102.5),
                    filename=f"{dataset}/{abbreviation.upper()}_{llm}_{prompting_strategy.title()}_Textbook_vs_Costumes.png",
                )

            plotter.line_plot(
                [
                    get_line_plot_data(
                        df[
                            (df["Prompting Strategy"] == prompting_strategy)
                            & (df["Costume"] == "textbook")
                            & (df["Variant"] == "standard")
                        ],
                        ["Scale", "Result Type"],
                    )
                    for prompting_strategy in PROMPTING_STRATEGIES
                ],
                [
                    prompting_strategy.replace("_", " ").title()
                    for prompting_strategy in PROMPTING_STRATEGIES
                ],
                extra_lines=[
                    greedy_results[problem][greedy_type] for greedy_type in greedy_types
                ],
                extra_labels=[
                    (
                        "Greedy" + " " + greedy_type.replace("_", " ").title()
                        if greedy_type != "results"
                        else "Greedy"
                    )
                    for greedy_type in greedy_types
                ],
                title=f"{abbreviation.upper()} {llm_name} Textbook Standard",
                xlim=(xticks[0] - 1, xticks[-1] + 1),
                xticks=xticks,
                ylim=(-2.5, 102.5),
                filename=f"{dataset}/{abbreviation.upper()}_{llm}_Textbook_Standard.png",
            )

            print(f"    ✅ {llm}")
