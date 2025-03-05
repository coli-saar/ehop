{
    id: "graph-coloring-gpt-random",
    problem_type: "graph_coloring",
    solver: {
        id: "graph-coloring-llm",
        model: "gpt-4o-2024-08-06",
    },
    loader: {
        id: "graph-coloring-loader"
    },
    variants: ["standard", "inverted"],
    costumes: ["textbook", "student_groups", "taekwondo_tournament", "parties_with_exes"],
    prompting_strategies: ["one_shot", "zero_shot_cot", "one_shot_cot", "ilp_python"],
    data: "data/problem_instances/graph_coloring/in_house/random_dataset"
}
