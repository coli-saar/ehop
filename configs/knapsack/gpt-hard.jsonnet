{
    id: "knapsack-gpt-hard",
    problem_type: "knapsack",
    solver: {
        id: "knapsack-llm",
        model: "gpt-4o-2024-08-06",
    },
    loader: {
        id: "knapsack-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "party_planning", "lemonade_stand", "sightseeing"],
    prompting_strategies: ["one_shot", "zero_shot_cot", "one_shot_cot", "ilp_python"],
    data: "data/problem_instances/knapsack/in_house/hard_dataset"
}
