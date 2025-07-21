# A file with functions for formatting/summarizing data from a DataFrame created with data_aggregation.master_df

import pandas as pd


def full_table(df: pd.DataFrame) -> str:
    """
    Formats the DataFrame into a CSV table with prompting strategies and result types as stacked column headers.
    """
    if df["Dataset"].nunique() != 1:
        raise ValueError(
            "DataFrame must contain only one dataset type (Random or Hard)."
        )
    hard = df.iloc[0]["Dataset"] == "Hard"
    return (
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
            lambda row: (row if hard and row.name[0] == "GCP" else row / 1.5),
            axis=1,
        )
        .round(1)
        .unstack()
        .fillna(0)
        .to_csv()
    ) or "Tabulation failed: DataFrame is empty or not properly formatted."


def paper_table(df: pd.DataFrame) -> str:
    """
    Formats the DataFrame into a CSV table matching the format used in the main paper (without calculating differences across rows).
    """
    return (
        df[df["Condition"] != "inverted costumed"]
        .groupby(
            ["Problem", "Condition", "Prompting Strategy", "Result Type"], observed=True
        )
        .size()
        .unstack()
        .fillna(0)
        .apply(lambda row: row / row.sum() * 100, axis=1)
        .round(1)["OPTIMAL"]
        .unstack()
        .to_csv()
        .replace("standard textbook", "Textbook")
        .replace("inverted textbook", "Inverted")
        .replace("standard costumed", "Costumed")
    ) or "Tabulation failed: DataFrame is empty or not properly formatted."


def classic_breakdown_table(df: pd.DataFrame) -> str:
    """
    Formats the DataFrame into a CSV table with a classic breakdown of results by problem, condition, and prompting strategy.
    """
    return (
        df.groupby(
            ["Problem", "Prompting Strategy", "Condition", "Result Type"], observed=True
        )
        .size()
        .unstack()
        .apply(lambda row: row / row.sum() * 100, axis=1)
        .round(1)
        .fillna(0)
        .to_csv()
    ) or "Tabulation failed: DataFrame is empty or not properly formatted."


def recognition_table(df: pd.DataFrame) -> str:
    """
    Formats the DataFrame into a CSV table showing the recognition of reasoning by problem, condition, and prompting strategy.
    """
    return (
        df.groupby(
            [
                "Problem",
                "Prompting Strategy",
                "Condition",
                "Recognized_Reasoning",
            ]
        )
        .size()
        .unstack()
        .fillna(0)
        .apply(lambda row: row / row.sum() * 100, axis=1)
        .round(1)
        .rename(columns={True: "Recognized", False: "Not Recognized"})  # type: ignore
        .to_csv()
    ) or "Tabulation failed: DataFrame is empty or not properly formatted."
