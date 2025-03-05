{
    id: "traveling-salesman-gpt-hard",
    problem_type: "traveling_salesman",
    solver: {
        id: "traveling-salesman-llm",
        model: "gpt-4o-2024-08-06",
    },
    loader: {
        id: "traveling-salesman-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "task_schedule", "exercise_schedule", "un_seating"],
    prompting_strategies: ["one_shot", "zero_shot_cot", "one_shot_cot", "ilp_python"],
    data: "data/problem_instances/traveling_salesman/in_house/hard_dataset"
}
