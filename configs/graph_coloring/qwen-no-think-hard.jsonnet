{
    id: "graph-coloring-qwen-no-think-hard",
    problem_type: "graph_coloring",
    solver: {
        id: "graph-coloring-llm",
        model: "PATH_TO_QWEN_3_32B_HERE",
        thinking: "False",
    },
    loader: {
        id: "graph-coloring-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "student_groups", "taekwondo_tournament", "parties_with_exes"],
    prompting_strategies: ["one_shot", "zero_shot_cot", "one_shot_cot", "ilp_lp", "ilp_python"],
    data: "data/problem_instances/graph_coloring/in_house/hard_dataset"
    output_filename: "qwen_no_think_hard",
}
