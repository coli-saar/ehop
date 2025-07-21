{
    id: "traveling-salesman-qwen-no-think-hard",
    problem_type: "traveling_salesman",
    solver: {
        id: "traveling-salesman-llm",
        model: "PATH_TO_QWEN_3_32B_HERE",
        thinking: "False",
    },
    loader: {
        id: "traveling-salesman-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "task_schedule", "exercise_schedule", "un_seating"],
    prompting_strategies: ["one_shot", "zero_shot_cot", "one_shot_cot", "ilp_lp", "ilp_python"],
    data: "data/problem_instances/traveling_salesman/in_house/hard_dataset"
    output_filename: "qwen_no_think_hard",
}
