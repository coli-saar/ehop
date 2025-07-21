{
    id: "graph-coloring-qwen-think-random",
    problem_type: "graph_coloring",
    solver: {
        id: "graph-coloring-llm",
        model: "PATH_TO_QWEN_3_32B_HERE",
        thinking: "True",
    },
    loader: {
        id: "graph-coloring-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "student_groups", "taekwondo_tournament", "parties_with_exes"],
    prompting_strategies: ["zero_shot", "ilp_python"],
    data: "data/problem_instances/graph_coloring/in_house/random_dataset"
    output_filename: "qwen_think_random",
}
