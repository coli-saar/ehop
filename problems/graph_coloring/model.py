import re
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from utils.llm_output_utils import extract_csloi

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import (
    BaseInstance,
    BaseLLMSolution,
    BaseLoader,
    BaseSolution,
)
from base.results import (
    ErroneousResult,
    IncompatibleFormatResult,
    OptimalResult,
    Result,
    SuboptimalResult,
)
from utils.utils import register


@dataclass
class GraphColoringSolution(BaseSolution):
    coloring: list[int]  # e.g., [1, 2, 1, 3, 2, 1]

    def __str__(self) -> str:
        return str(self.coloring)

    def get_node_color(self, node: int) -> int:
        return self.coloring[node - 1]


class GraphColoringLLMSolution(BaseLLMSolution, GraphColoringSolution): ...


@dataclass
class GraphColoringInstance(BaseInstance[GraphColoringSolution]):
    graph: nx.Graph  # vertices are 1-indexed
    complement_graph: nx.Graph
    chromatic_number: int | None
    optimal_coloring: list[int] | None  # colors are 1-indexed

    def __init__(
        self,
        graph: nx.Graph,
        chromatic_number: int | None = None,
        optimal_coloring: list[int] | None = None,
    ):
        if set(graph.nodes) != set(range(1, len(graph) + 1)):
            raise ValueError("Graph nodes must be 1-indexed integers.")

        self.graph = graph
        self.complement_graph = nx.complement(graph)

        self.optimal_coloring = optimal_coloring
        if chromatic_number is None and optimal_coloring is not None:
            self.chromatic_number = len(set(optimal_coloring))
        else:
            self.chromatic_number = chromatic_number

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphColoringInstance):
            return NotImplemented

        node_num_match = self.graph.number_of_nodes() == other.graph.number_of_nodes()
        edge_match: bool = self.graph.edges() == other.graph.edges()

        return node_num_match and edge_match

    def evaluate(
        self,
        solution: GraphColoringSolution,
        variant: str = "standard",
        verbose: bool = False,
    ) -> Result:
        if variant not in {"standard", "inverted"}:
            raise ValueError('Variant must be either "standard" or "inverted"')

        # check for valid values
        if len(solution.coloring) != self.graph.number_of_nodes():
            return IncompatibleFormatResult(
                f"Solution has colors for the wrong number of nodes ({len(solution.coloring)} instead of {self.graph.number_of_nodes()})."
            )
        if not set(solution.coloring).issubset(range(1, len(self.graph) + 1)):
            return IncompatibleFormatResult(
                "Solution uses a bad set of colors (non-integers or integers outside the expected range)."
            )

        # check for valid coloring
        for u, v in self.graph.edges:
            if solution.get_node_color(u) == solution.get_node_color(v):
                return ErroneousResult(
                    f"Solution has {'non-' if variant == 'inverted' else ''}adjacent nodes with the same color. Nodes {u} and {v} are both colored {solution.get_node_color(u)}."
                )

        num_colors = len(set(solution.coloring))

        optimum = self.optimal_value(variant)

        # result is valid, compare it to optimum
        if verbose:
            print(f"Used {num_colors} colors, vs. optimum {optimum}")

        if num_colors == optimum:
            return OptimalResult(
                solution_string=str(solution), summary_value=num_colors
            )
        else:
            return SuboptimalResult(
                solution_string=str(solution), summary_value=num_colors
            )

    def optimal_value(self, variant: str = "standard") -> int:
        match variant:
            case "standard" | "inverted":
                if self.chromatic_number is None:
                    raise RuntimeError("Optimal value not known/provided")
                return self.chromatic_number
            case _:
                raise ValueError('Variant must be either "standard" or "inverted"')

    def reasonable_encoding(self) -> str:
        num_nodes = self.graph.number_of_nodes()
        edges = [str(edge).replace(" ", "") for edge in self.graph.edges]
        return f"{num_nodes}{''.join(edges)}"


@register("graph-coloring-loader")
class GraphColoringLoader(BaseLoader[GraphColoringInstance]):
    @staticmethod
    def load(
        problem_path: str, solution_path: str | None = None
    ) -> GraphColoringInstance:
        g = nx.Graph()
        expected_edges = None
        num_edges = 0

        with open(problem_path, "r") as f:
            for line in f:
                if line.startswith("c") or not line.strip():
                    continue  # skip comments and empty lines
                elif not g.number_of_nodes():
                    if not re.match(r"p \w*edge\w* \d+ \d+", line):
                        raise ValueError(
                            f'Expected a problem line (formatted as "p edge NODES EDGES"), got: {line}'
                        )
                    else:
                        num_nodes, expected_edges = map(int, line.split()[2:])
                        g.add_nodes_from(range(1, num_nodes + 1))
                elif re.match(r"e \d+ \d+", line):
                    u, v = map(int, line.split()[1:])
                    if u == v:
                        raise ValueError(
                            f"Self-loop detected: edge ({u}, {v}) is not allowed"
                        )
                    elif u > g.number_of_nodes() or v > g.number_of_nodes():
                        raise ValueError(
                            f"Edge ({u}, {v}) has a node index that is out of bounds for the number of nodes ({g.number_of_nodes()})"
                        )
                    else:
                        g.add_edge(u, v)
                        num_edges += 1
                else:
                    raise ValueError(f"Unexpected line: {line}")

        if num_edges != expected_edges:
            raise ValueError(
                f"Number of edges does not match expected value: expected {expected_edges}, got {len(g.edges)}"
            )

        inst = GraphColoringInstance(graph=g)

        if solution_path is not None:
            with open(solution_path, "r") as f:
                for line in f:
                    if inst.chromatic_number is None:
                        inst.chromatic_number = int(line)
                    elif inst.optimal_coloring is None:
                        inst.optimal_coloring = extract_csloi(line)
                    else:
                        raise ValueError("Unexpected line in solution file")

        return inst

    @staticmethod
    def store(inst: GraphColoringInstance, folder: Path, header: str = "") -> None:
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / "problem.col", "w") as f:
            if header:
                f.write(header + "\n")
            f.write(
                f"p edge {inst.graph.number_of_nodes()} {inst.graph.number_of_edges()}\n"
            )
            for u, v in inst.graph.edges:
                f.write(f"e {u} {v}\n")

        if inst.chromatic_number is not None and inst.optimal_coloring is not None:
            with open(folder / "solution.sol", "w") as f:
                f.write(str(inst.chromatic_number) + "\n")
                f.write(", ".join(map(str, inst.optimal_coloring)) + "\n")
