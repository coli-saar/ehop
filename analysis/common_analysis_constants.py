# A collection of constants used in the analysis scripts

PROBLEMS: list[str] = ["graph_coloring", "knapsack", "traveling_salesman"]
ABBREVS: list[str] = ["gcp", "ksp", "tsp"]
SCALE_WORDS: dict[str, str] = {"gcp": "nodes", "ksp": "items", "tsp": "cities"}

COSTUME_DICT: dict[str, list[str]] = {
    "gcp": [
        "textbook",
        "student_groups",
        "parties_with_exes",
        "taekwondo_tournament",
    ],
    "ksp": [
        "textbook",
        "lemonade_stand",
        "sightseeing",
        "party_planning",
    ],
    "tsp": [
        "textbook",
        "task_schedule",
        "exercise_schedule",
        "un_seating",
    ],
}

COSTUME_LIST: list[str] = [
    "textbook",
    "student_groups",
    "parties_with_exes",
    "taekwondo_tournament",
    "lemonade_stand",
    "sightseeing",
    "party_planning",
    "task_schedule",
    "exercise_schedule",
    "un_seating",
]

VARIANTS: list[str] = ["standard", "inverted"]

PROMPTING_STRATEGIES: list[str] = [
    "one_shot",
    "zero_shot_cot",
    "one_shot_cot",
    "ilp_lp",
    "ilp_python",
]

DATASETS: list[str] = ["random", "hard"]

GREEDY_STRAT_LIST: list[list[str]] = [
    ["largest_first", "random_sequential"],
    ["density", "value"],
    ["results"],
]

LLMS: list[str] = ["GPT", "Llama"]
LLM_FULL_NAMES: list[str] = ["GPT-4o (2024-08-06)", "Llama 3.1 70B Instruct"]

PROBLEM_SCALES: dict[str, dict[str, tuple[float, ...]]] = {
    "gcp": {"ticks": (4, 5, 6, 7, 8, 9), "limits": (3.875, 9.125)},
    "ksp": {"ticks": (4, 8, 12, 16, 20, 24), "limits": (3.5, 24.5)},
    "tsp": {"ticks": (4, 5, 6, 7, 8, 9), "limits": (3.875, 9.125)},
}
