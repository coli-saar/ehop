{
    id: "knapsack-qwen-think-hard",
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
    data: "data/problem_instances/knapsack/in_house/hard_dataset"
    output_filename: "qwen_think_hard",
}
