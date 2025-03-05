# This file runs some thorough sanity checks on the consolidated/aggregated data to check whether it seems to contain what it is expected to.

import sys

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import (
    ABBREVS,
    COSTUME_DICT,
    DATASETS,
    LLMS,
    PROBLEM_SCALES,
    PROMPTING_STRATEGIES,
    VARIANTS,
)
from analysis.data_aggregation import master_df

for dataset in DATASETS:
    print(f"Generating {dataset} dataset...", end="\r")
    df = master_df(dataset)
    print(f"Checking whether {dataset} dataset has all expected rows...", end="\r")
    for abbrev in ABBREVS:
        scales = PROBLEM_SCALES[abbrev]["ticks"]
        if abbrev == "gcp" and dataset == "hard":
            scales = (6, 7, 8, 9)
        costumes = COSTUME_DICT[abbrev]
        variants = VARIANTS
        abbrev_df = df[df["Problem"] == abbrev.upper()]
        for scale in scales:
            scale_df = abbrev_df[abbrev_df["Scale"] == scale]
            for costume in costumes:
                costume_df = scale_df[scale_df["Costume"] == costume]
                for variant in variants:
                    variant_df = costume_df[costume_df["Variant"] == variant]
                    for ps in PROMPTING_STRATEGIES:
                        ps_df = variant_df[variant_df["Prompting Strategy"] == ps]
                        for llm in LLMS:
                            llm_df = ps_df[ps_df["LLM"] == llm]
                            for i in range(25):
                                if (
                                    len(
                                        llm_df[
                                            llm_df["Problem Name"]
                                            == f"in_house_{scale}_{i}"
                                        ]
                                    )
                                    != 1
                                ):
                                    print(
                                        f"{dataset} {abbrev} {scale} {costume} {variant} {ps} {llm} {len(llm_df[llm_df['Problem Name']== f'in_house_{scale}_{i}'])}"
                                    )
    print(
        f"Checking that {dataset} has current version of ILPFAILURE mechanics...",
        end="\r",
    )
    for _, row in df.iterrows():
        if "An error occurred" in row["Prompt"]:
            print(
                f"{dataset} {row['Problem']} {row['Problem Name']} {row['Costume']} {row['Variant']} {row['Prompting Strategy']} {row['LLM']}"
            )

    print(f"Finished checking {dataset} dataset.", end=" " * 40 + "\n")
