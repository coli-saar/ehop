import sys

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import DATASETS
from utils.analysis.data_aggregation import master_df

for dataset in DATASETS:
    print(dataset.title() + " Dataset:")
    # df = master_df(
    #     dataset, llms=["gpt", "llama3.1", "qwen_no_think"], var_dis_condition=True
    # )
    df = master_df(
        dataset,
        # llms=["deepseek", "qwen_think"],
        llms=["qwen_think"],
        strategies=["zero_shot", "ilp_python"],
        var_dis_condition=True,
    )

    df = df[
        # (df["Condition"] != "inverted costumed") &
        (df["Prompting Strategy"] != "ilp_lp")
    ]

    print(
        (
            df.groupby(
                [
                    "Problem",
                    "Condition",
                    "LLM",
                    "Prompting Strategy",
                    "Result Type",
                ],
                observed=True,
            )
            .size()
            .unstack()
            .fillna(0)
        )
        .apply(lambda row: row / row.sum() * 100, axis=1)
        .round(1)["OPTIMAL"]
        .unstack()
        .unstack()
        .to_csv(lineterminator="\n")
        # .replace(",", "],[")
        # .replace("\n", "],\n[")
    )
