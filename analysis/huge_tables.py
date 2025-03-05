import sys

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from analysis.common_analysis_constants import DATASETS
from analysis.data_aggregation import master_df
from utils.utils import csv_to_typst

for dataset in DATASETS:
    print(dataset.title() + " Dataset:")
    df = master_df(dataset)

    print(
        csv_to_typst(
            (
                df.groupby(
                    [
                        "Problem",
                        "LLM",
                        "Variant",
                        "Costume",
                        "Result Type",
                        "Prompting Strategy",
                    ],
                    observed=True,
                )
                .size()
                .unstack()
            )
            .apply(
                lambda row: (
                    row if dataset == "hard" and row.name[0] == "GCP" else row / 1.5
                ),
                axis=1,
            )
            .round(1)
            .unstack()
            .fillna(0)
            .drop(
                columns=[
                    ("one_shot", "ILPFAILURE"),
                    ("zero_shot_cot", "ILPFAILURE"),
                    ("one_shot_cot", "ILPFAILURE"),
                ]
            )
            .to_csv(lineterminator="\n"),
            3,
            [16, 8, 4],
        )
        .replace("[GCP]", "[#gcp-short]")
        .replace("[KSP]", "[#ksp-short]")
        .replace("[TSP]", "[#tsp-short]")
        .replace("[one_shot]", "[One-Shot]")
        .replace("[zero_shot_cot]", "[Zero-Shot CoT]")
        .replace("[one_shot_cot]", "[One-Shot CoT]")
        .replace("[ilp_lp]", "[ILP LP]")
        .replace("[ilp_python]", "[ILP Python]")
        .replace("[OPTIMAL]", "[O]")
        .replace("[SUBOPTIMAL]", "[S]")
        .replace("[ERRONEOUS]", "[E]")
        .replace("[INCOMPATIBLE]", "[I]")
        .replace("[ILPFAILURE]", "[F]")
        .replace("[standard]", "[#variant-emojis.standard]")
        .replace("[inverted]", "[#variant-emojis.inverted]")
        .replace("[parties_with_exes]", "[#costume-emojis.gcp.parties]")
        .replace("[student_groups]", "[#costume-emojis.gcp.students]")
        .replace("[taekwondo_tournament]", "[#costume-emojis.gcp.taekwondo]")
        .replace("[lemonade_stand]", "[#costume-emojis.ksp.lemonade]")
        .replace("[sightseeing]", "[#costume-emojis.ksp.sightseeing]")
        .replace("[party_planning]", "[#costume-emojis.ksp.planning]")
        .replace("[task_schedule]", "[#costume-emojis.tsp.tasks]")
        .replace("[exercise_schedule]", "[#costume-emojis.tsp.exercises]")
        .replace("[un_seating]", "[#costume-emojis.tsp.seating]")
    )
