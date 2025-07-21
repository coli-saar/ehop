{
    id: "traveling-salesman-qwen-hard",
    problem_type: "traveling_salesman",
    solver: {
        id: "traveling-salesman-llm",
        model: "PATH_TO_QWEN_3_32B_HERE",
        thinking: "True",
    },
    loader: {
        id: "traveling-salesman-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "task_schedule", "exercise_schedule", "un_seating"],
    prompting_strategies: ["zero_shot", "ilp_python"],
    data: "data/problem_instances/traveling_salesman/in_house/hard_dataset"
    output_filename: "qwen_think_hard",
}
