{
    id: "knapsack-llama-random",
    problem_type: "knapsack",
    solver: {
        id: "knapsack-llm",
        model: "PATH_TO_LLAMA_MODEL_HERE",
    },
    loader: {
        id: "knapsack-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "party_planning", "lemonade_stand", "sightseeing"],
    prompting_strategies: ["one_shot", "zero_shot_cot", "one_shot_cot", "ilp_python"],
    data: "data/problem_instances/knapsack/in_house/random_dataset"
}
