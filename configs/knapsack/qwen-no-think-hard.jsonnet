{
    id: "knapsack-qwen-no-think-hard",
    problem_type: "knapsack",
    solver: {
        id: "knapsack-llm",
        model: "PATH_TO_QWEN_3_32B_HERE",
        thinking: "False",
    },
    loader: {
        id: "knapsack-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "party_planning", "lemonade_stand", "sightseeing"],
    prompting_strategies: ["one_shot", "zero_shot_cot", "one_shot_cot", "ilp_lp", "ilp_python"],
    data: "data/problem_instances/knapsack/in_house/hard_dataset"
    output_filename: "qwen_no_think_hard",
}
