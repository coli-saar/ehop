from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULT_TYPES = ["OPTIMAL", "SUBOPTIMAL", "ERRONEOUS", "INCOMPATIBLE", "ILPFAILURE"]
VARIANTS = ["standard", "inverted"]


def load_df(
    path: str | Path,
    categoricals: dict[str, list] | None = None,
    problem_prefix: str | None = "in_house",
    extract_scale: bool = True,
    extract_validity: bool = True,
):
    """Loads a CSV file of experiment results into a pandas DataFrame."""
    # Load the data (keep only problems with the specified prefix)
    df = pd.read_csv(path)
    if problem_prefix is not None:
        df = df[df["Problem Name"].str.startswith(problem_prefix)].reset_index(
            drop=True
        )

    # Add Scale column as another proxy for problem difficulty
    if extract_scale:
        df["Scale"] = (
            df["Problem Name"].apply(lambda x: x.split("_")[2]).astype(pd.Int32Dtype())
        )

    # Tell pandas that the following are categorical variables with an ordering
    df["Result Type"] = pd.Categorical(
        df["Result Type"], categories=RESULT_TYPES, ordered=True
    )
    if extract_validity:
        df["Validity"] = pd.Categorical(
            df["Result Type"].apply(
                lambda x: "VALID" if x in ["OPTIMAL", "SUBOPTIMAL"] else "INVALID"
            ),
            categories=["VALID", "INVALID"],
            ordered=True,
        )
    if categoricals is not None:
        for k, v in categoricals.items():
            df[k] = pd.Categorical(df[k], categories=v, ordered=True)
    return df


def accuracy_df(
    df: pd.DataFrame,
    groupby: list[str] = [
        "Prompting Strategy",
        "Variant",
        "Costume",
        "Scale",
    ],
    unstack_levels: list[str] = ["Scale"],
    denominator: int = 25,
) -> pd.DataFrame | pd.Series:
    """Turns a DataFrame of experiment results into a DataFrame of accuracy percentages."""
    raw_vals_df = (
        df.groupby(groupby + ["Result Type"], observed=False)
        .count()
        .iloc[:, 0]
        .unstack(level="Result Type")["OPTIMAL"]
    )
    unstacked_df: pd.DataFrame | pd.Series = raw_vals_df
    for level in unstack_levels:
        unstacked_df = unstacked_df.unstack(level=level)
    return (unstacked_df * (100 / denominator)).round(3)


def get_line_plot_data(
    df: pd.DataFrame, groupby: list[str], last_group_val: str = "OPTIMAL"
) -> tuple[list[float], list[float]]:
    """
    Groups a DataFrame by a list of columns and returns the accuracy percentages for the last group value.
    Returns a tuple of values for the last group value and corresponding accuracy percentages.
    """
    percentages = (
        df.groupby(groupby, observed=False)
        .count()
        .iloc[:, 0]
        .unstack(level=-1, fill_value=0)
        .apply(lambda row: row / row.sum() * 100, axis=1)
    )

    acc = percentages[last_group_val]
    acc.index = acc.index.astype(int)

    return (acc.index.to_list(), acc.to_list())


def get_diff_line_plot_data(
    df: pd.DataFrame,
    groupby: list[str],
    column: str = "Variant",
    value_1: str = "standard",
    value_2: str = "inverted",
    last_group_val: str = "OPTIMAL",
) -> tuple[list[float], list[float]]:
    """Gets line plot data for the difference between two values of a column."""
    standard_line = get_line_plot_data(
        df[df[column] == value_1], groupby, last_group_val
    )
    inverted_line = get_line_plot_data(
        df[df[column] == value_2], groupby, last_group_val
    )

    assert standard_line[0] == inverted_line[0]
    return (
        standard_line[0],
        [s - i for s, i in zip(standard_line[1], inverted_line[1])],
    )


class Plotter:
    """A class with a variety of plotting methods for visualizing experiment data."""

    def __init__(self, problem_name: str = "miscellaneous"):
        self.problem_name = problem_name
        sns.set_theme(style="darkgrid")
        sns.set_palette("viridis", n_colors=len(RESULT_TYPES))
        Path(f"analysis/plots/{self.problem_name}/").mkdir(parents=True, exist_ok=True)

    def line_plot(
        self,
        lines: list[tuple[list[float], list[float]]],
        labels: list[str],
        extra_lines: list[tuple[list[float], list[float]]] = [],
        extra_labels: list[str] = [],
        bold_x: list[int] | None = None,
        bold_y: list[float] | None = None,
        figsize: tuple[int, int] = (8, 5),
        title: str | None = None,
        filename: str | None = None,
        half_dashed: bool = False,
        **kwargs,
    ):
        sns.set_theme(style="whitegrid")
        sns.set_palette(
            "tab10", n_colors=len(lines) // 2 if half_dashed else len(lines)
        )
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if bold_x is not None:
            for x in bold_x:
                ax.axvline(x=x, color="k")
        if bold_y is not None:
            for y in bold_y:
                ax.axhline(y=y, color="k")
        for i, (line, name) in enumerate(zip(lines, labels)):
            ax.plot(
                line[0],
                line[1],
                "d-" if half_dashed and i % 2 == 1 else "o-",
                color=sns.color_palette()[i // 2 if half_dashed else i],
                label=name,
                dashes=[2] if half_dashed and i % 2 == 1 else (None, None),
            )
        if extra_lines and extra_labels:
            sns.set_palette("gist_gray", n_colors=len(extra_lines))
            for i, (line, name) in enumerate(zip(extra_lines, extra_labels)):
                ax.plot(
                    line[0],
                    line[1],
                    "X-",
                    color=sns.color_palette()[i],
                    label=name,
                )
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fancybox=True, shadow=True
        )
        fig.suptitle(
            title or f"{self.problem_name.replace('_', ' ').title()} Results",
            fontweight="bold",
        )
        ax.set(**kwargs)
        plt.tight_layout()
        plt.savefig(f"analysis/plots/{self.problem_name}/{filename or 'Lineplot.png'}")
        plt.close()
        sns.set_palette("viridis", n_colors=len(RESULT_TYPES))

    def costume_variant_line_plot(
        self,
        df: pd.DataFrame,
        costumes: list[str],
        title_prefix: str = "",
        groupby: list[str] = ["Scale", "Result Type"],
        variants: list[str] = VARIANTS,
        last_group_val: str = "OPTIMAL",
        filename: str = "Costume_Variant_Lineplot.png",
        **kwargs,
    ):
        self.line_plot(
            lines=[
                get_line_plot_data(
                    df[(df["Costume"] == costume) & (df["Variant"] == variant)],
                    groupby=groupby,
                    last_group_val=last_group_val,
                )
                for costume in costumes
                for variant in variants
            ],
            labels=[
                f"{costume.replace('_', ' ').title().replace('Un','UN')} | {variant.title()}"
                for costume in costumes
                for variant in variants
            ],
            title=(
                (title_prefix + " " if title_prefix else "")
                + "Costume/Variant Performances"
            ),
            ylim=(-2.5, 102.5),
            xlabel="Scale",
            ylabel=f"Accuracy (% {last_group_val})",
            half_dashed=True,
            filename=filename,
            **kwargs,
        )

    def costume_variant_stack_plot(
        self,
        df: pd.DataFrame,
        costumes: list[str],
        variants: list[str] = VARIANTS,
        title_prefix: str = "",
        figsize: tuple[int, int] = (12, 8),
        filename: str | None = None,
        **kwargs,
    ):
        fig, ax = plt.subplots(len(variants), len(costumes), figsize=figsize)
        for i, variant in enumerate(variants):
            for j, costume in enumerate(costumes):
                y_vals = []
                for result_type in RESULT_TYPES:
                    x, y = get_line_plot_data(
                        df[(df["Costume"] == costume) & (df["Variant"] == variant)],
                        ["Scale", "Result Type"],
                        last_group_val=result_type,
                    )
                    y_vals.append(y)
                ax[i, j].stackplot(x, y_vals, alpha=0.8, labels=RESULT_TYPES)
                ax[i, j].set(**kwargs)
                ax[i, j].set_title(
                    f"{costume.replace('_', ' ').title().replace('Un','UN')} | {variant.title()}"
                )
                if i < len(variants) - 1:
                    ax[i, j].tick_params(axis="x", labelbottom=False)
                if j > 0:
                    ax[i, j].tick_params(axis="y", labelleft=False)
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncols=len(RESULT_TYPES))
        fig.suptitle(
            (title_prefix + " " if title_prefix else "") + "Costume/Variant Results"
        )
        # fig.tight_layout()
        plt.savefig(
            f"analysis/plots/{self.problem_name}/{filename or 'Costume_Variant_Stackplot.png'}"
        )
        plt.close()

    def template_comparison(self, llm_df: pd.DataFrame, filename: str | None = None):
        ax = (
            llm_df.groupby(["Template", "Result Type"], observed=False)
            .size()
            .unstack()
            .plot(kind="bar", stacked=True, figsize=(10, 5), rot=0, colormap="viridis")
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(RESULT_TYPES),
            fancybox=True,
            shadow=True,
        )
        plt.savefig(
            f"analysis/plots/{self.problem_name}/{filename or 'Template_Comparison.png'}"
        )
        plt.close()

    def result_by_x_and_y(
        self,
        llm_df: pd.DataFrame,
        x: str = "Size",
        y: str = "Template",
        title: str | None = None,
        filename: str | None = None,
    ):
        plot = sns.displot(
            data=llm_df,
            x=x,
            hue="Result Type",
            col=y,
            multiple="stack",
            bins=10,
        )
        plot.figure.suptitle(
            title or f"{self.problem_name.replace('_', ' ').title()} Result Types",
            fontweight="bold",
        )
        plot.figure.subplots_adjust(top=0.85)
        plt.savefig(
            "analysis/plots/"
            + (f"{self.problem_name}/" if self.problem_name is not None else "")
            + (
                filename
                or f"Result_by_{x.replace(' ', '_')}_and_{y.replace(' ', '_')}.pdf"
            )
        )
        plt.close()

    def distance_distributions(
        self,
        dfs: list[pd.DataFrame],
        names: list[str],
        minimization: bool,
        filename: str | None = None,
    ):
        model_list = []
        distance_list = []

        for df, name in zip(dfs, names):
            for _, row in df.iterrows():
                if pd.isna(row["Summary Value"]) or pd.isna(row["Optimal Value"]):
                    continue
                model_list.append(name)
                distance_list.append(row["Summary Value"] - row["Optimal Value"])

        if not minimization:
            distance_list = [-d for d in distance_list]

        distance_df = pd.DataFrame({"Model": model_list, "Distance": distance_list})
        distance_df["Distance"] = distance_df["Distance"].astype(int)

        bins = max(distance_df["Distance"]) - min(distance_df["Distance"]) + 1

        plot = sns.histplot(
            # {"GPT-4o": llm_distances, "Random": random_distances},
            distance_df,
            x="Distance",
            hue="Model",
            bins=bins,
            discrete=True,
            # multiple="dodge",
            shrink=0.8,
            element="poly",
        )
        plot.set(xlabel="Distance from Optimal", ylabel="Number of Valid Solutions")
        if bins <= 10:
            plot.set_xticks(
                range(min(distance_df["Distance"]), max(distance_df["Distance"]) + 1)
            )
        plt.savefig(
            f"analysis/plots/{self.problem_name}/{filename or 'Distances_from_Optimal.png'}"
        )
        plt.close()

    def result_type_pies(
        self,
        dfs: list[pd.DataFrame],
        names: list[str],
        title: str | None = None,
        filename: str | None = None,
    ):
        fig, ax = plt.subplots(ncols=len(dfs), figsize=(12, 5))

        for df, name, subplot in zip(dfs, names, ax):
            subplot.set_title(name, fontweight="bold")
            df.groupby("Result Type", observed=False).size().plot(
                kind="pie",
                autopct="%.1f",
                colormap="viridis",
                pctdistance=0.7,
                ax=subplot,
            )

        for subplot in ax:
            for text in subplot.texts:
                if text._text in RESULT_TYPES or text._text == "0.0":
                    text.set_visible(False)
                else:
                    text.set_color("white")

        fig.suptitle(
            title or f"{self.problem_name.replace('_', ' ').title()} Result Types",
            fontweight="bold",
        )
        fig.legend(RESULT_TYPES, ncol=len(RESULT_TYPES), loc="lower center")
        plt.tight_layout()
        plt.savefig(
            f"analysis/plots/{self.problem_name}/{filename or 'Result_Type_Pies.png'}"
        )
        plt.close()

    def result_type_table(
        self,
        dfs: list[pd.DataFrame],
        names: list[str],
        total: int,
        title: str | None = None,
        filename: str | None = None,
    ):
        fig, ax = plt.subplots()
        table_rows = []

        for df, name in zip(dfs, names):
            grouped_df = (
                df.groupby("Result Type", observed=False).size().reset_index().T
            )
            new_header = grouped_df.iloc[0]
            grouped_df = grouped_df[1:]
            grouped_df.columns = pd.Index([str(col) for col in new_header])
            row_dict = (grouped_df.iloc[0] / total * 100).to_dict() | {"Model": name}
            table_rows.append(row_dict)

        full_df = pd.DataFrame(table_rows, columns=["Model"] + RESULT_TYPES)

        sns.heatmap(
            full_df.set_index("Model"),
            annot=True,
            fmt=".1f",
            cmap="vlag",
            linewidths=0.2,
            ax=ax,
        )

        ax.set(xlabel="", ylabel="")
        ax.xaxis.set_tick_params(rotation=30)
        ax.xaxis.tick_top()

        fig.suptitle(
            title or f"{self.problem_name.replace('_', ' ').title()} Result Types",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"analysis/plots/{self.problem_name}/{filename or 'Result_Type_Table.png'}"
        )
        plt.close()

    def parameter_based_result_type_table(
        self,
        llm_df: pd.DataFrame,
        total: int,
        columns: list[str] = ["Prompting Strategy", "Result Type"],
        size: tuple[int, int] = (6, 4),
        title: str | None = None,
        filename: str | None = None,
    ):
        fig, ax = plt.subplots()

        grouped = llm_df.groupby(columns, observed=False).size().unstack() / total * 100
        # grouped.index = grouped.index.map(
        #     lambda x: " ".join(str(elem) for elem in x)
        #     .replace("_", " ")
        #     .title()
        #     .replace("Cot", "CoT")
        # )

        print(grouped.to_csv())

        fig.set_size_inches(size)
        sns.heatmap(
            grouped,
            annot=True,
            fmt=".1f",
            cmap="vlag",
            linewidths=0.2,
            ax=ax,
            vmin=0,
            vmax=100,
        )

        ax.set(xlabel="", ylabel="")
        ax.yaxis.set_tick_params(rotation=0)
        ax.xaxis.set_tick_params(rotation=30)
        ax.xaxis.tick_top()

        fig.suptitle(
            title or f"{self.problem_name.replace('_', ' ').title()} Result Types",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"analysis/plots/{self.problem_name}/{filename or 'Prompt_Based_Result_Type_Table.png'}"
        )
        plt.close()

    def prompt_template_breakdown_table(
        self,
        llm_df: pd.DataFrame,
        total: int,
        title: str | None = None,
        filename: str | None = None,
    ):
        fig, ax = plt.subplots()

        grouped = (
            llm_df.groupby(
                ["Prompting Strategy", "Template", "Result Type"], observed=False
            )
            .size()
            .unstack()
            / total
            * 100
        )
        grouped.index = grouped.index.map(
            lambda x: " ".join(str(elem) for elem in x)
            .replace("_", " ")
            .title()
            .replace("Cot", "CoT")
        )

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)
        sns.heatmap(
            grouped,
            annot=True,
            fmt=".1f",
            cmap="vlag",
            linewidths=0.2,
            ax=ax,
            vmin=0,
            vmax=100,
        )
        ax.set(xlabel="", ylabel="")
        ax.yaxis.set_tick_params(rotation=0)
        ax.xaxis.set_tick_params(rotation=30)
        ax.xaxis.tick_top()

        ax.yaxis.set_ticklabels(
            [
                label.get_text().replace("-", " | ").replace(" Shot", "-Shot")
                for label in ax.yaxis.get_ticklabels()
            ]
        )

        fig.suptitle(
            title or f"{self.problem_name.replace('_', ' ').title()} Result Types",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"analysis/plots/{self.problem_name}/{filename or 'Prompt_Template_Breakdown_Table.png'}"
        )
        plt.close()
