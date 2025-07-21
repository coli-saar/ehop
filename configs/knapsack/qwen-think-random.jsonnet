{
    id: "knapsack-qwen-think-random",
    problem_type: "knapsack",
    solver: {
        id: "knapsack-llm",
        model: "PATH_TO_QWEN_3_32B_HERE",
        thinking: "True",
    },
    loader: {
        id: "knapsack-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "party_planning", "lemonade_stand", "sightseeing"],
    prompting_strategies: ["zero_shot", "ilp_python"],
    data: "data/problem_instances/knapsack/in_house/random_dataset"
    output_filename: "qwen_think_random",
}
