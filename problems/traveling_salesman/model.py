import re
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import networkx as nx

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
from utils.llm_output_utils import extract_csloi
from utils.utils import register


@dataclass
class TravelingSalesmanSolution(BaseSolution):
    ordering: list[int]  # e.g., [1, 3, 2, 5, 4]

    def __str__(self) -> str:
        return str(self.ordering)


class TravelingSalesmanLLMSolution(BaseLLMSolution, TravelingSalesmanSolution): ...


def invert_tsp_graph(g: nx.Graph, shift: int = 1) -> nx.Graph:
    m = max(map(lambda x: x[2]["weight"], edge_data := g.edges(data=True)))
    inverted_data = [
        (u, v, {"weight": m - attr["weight"] + shift}) for u, v, attr in edge_data
    ]
    return nx.Graph(inverted_data)


@dataclass
class TravelingSalesmanInstance(BaseInstance[TravelingSalesmanSolution]):
    graph: nx.Graph  # vertices are 1-indexed
    minimum_distance: int | None
    minimum_ordering: list[int] | None
    maximum_distance: int | None
    maximum_ordering: list[int] | None
    inversion_shift: int | None = None
    shifted_graph: nx.Graph | None = None
    shifted_maximum_distance: int | None = None

    def __init__(
        self,
        graph: nx.Graph,
        minimum_distance: int | None = None,
        minimum_ordering: list[int] | None = None,
        maximum_distance: int | None = None,
        maximum_ordering: list[int] | None = None,
        inversion_shift: int | None = None,
        shifted_maximum_distance: int | None = None,
    ):
        if set(graph.nodes) != set(range(1, len(graph) + 1)):
            raise ValueError("Graph nodes must be 1-indexed integers.")

        self.graph = graph

        self.minimum_ordering = minimum_ordering
        if minimum_distance is None and minimum_ordering is not None:
            self.minimum_distance = self.compute_distance(minimum_ordering)
        else:
            self.minimum_distance = minimum_distance

        self.maximum_ordering = maximum_ordering
        if maximum_distance is None and maximum_ordering is not None:
            self.maximum_distance = self.compute_distance(maximum_ordering)
        else:
            self.maximum_distance = maximum_distance

        self.inversion_shift = inversion_shift
        self.shifted_graph = (
            invert_tsp_graph(graph, shift=inversion_shift)
            if inversion_shift is not None
            else None
        )
        if shifted_maximum_distance is None and (
            inversion_shift is not None and minimum_ordering is not None
        ):
            self.shifted_maximum_distance = TravelingSalesmanInstance(
                graph=invert_tsp_graph(graph, shift=inversion_shift)
            ).compute_distance(minimum_ordering)
        else:
            self.shifted_maximum_distance = shifted_maximum_distance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TravelingSalesmanInstance):
            return NotImplemented

        node_num_match = self.graph.number_of_nodes() == other.graph.number_of_nodes()
        edge_match: bool = all(
            x == y for x, y in zip(self.graph.edges.data(), other.graph.edges.data())
        )

        return node_num_match and edge_match

    def compute_distance(self, ordering: list[int], variant: str = "standard") -> int:
        """
        A helper function to compute the total distance of a given ordering.
        It is assumed that the given ordering is a 1-indexed tour of the graph that starts at node 1.
        """
        if variant == "inverted":
            if self.inversion_shift is None:
                raise ValueError(
                    "Cannot compute shifted distance without an inversion shift."
                )
            g = invert_tsp_graph(self.graph, shift=self.inversion_shift)
        else:
            g = self.graph

        return sum(
            [g.edges[u, v]["weight"] for u, v in zip(ordering, ordering[1:] + [1])]
        )

    def evaluate(
        self,
        solution: TravelingSalesmanSolution,
        variant: str = "standard",
        verbose: bool = False,
    ) -> Result:
        # check for valid values
        if len(solution.ordering) != self.graph.number_of_nodes():
            return IncompatibleFormatResult(
                f"Solution has ordering with the wrong number of locations ({len(solution.ordering)} instead of {self.graph.number_of_nodes()})."
            )
        if not set(solution.ordering).issubset(range(1, len(self.graph) + 1)):
            return IncompatibleFormatResult(
                "Solution uses a bad set of locations (non-integers or integers outside the expected range)."
            )

        # check for valid ordering
        for location in range(1, self.graph.number_of_nodes() + 1):
            if location not in solution.ordering:
                return ErroneousResult(
                    f"Solution does not contain location {location}."
                )

        if solution.ordering[0] != 1:
            return ErroneousResult("Solution does not start at location 1.")

        # at this point, the solution should be valid (i.e., it's a permutation of 1 through n that starts with 1)

        distance = self.compute_distance(solution.ordering, variant)

        optimum = self.optimal_value(variant)

        # result is valid, compare it to optimum
        if verbose:
            print(f"Traveled distance {distance}, vs. optimum {optimum}")

        if distance == optimum:
            return OptimalResult(solution_string=str(solution), summary_value=distance)
        else:
            return SuboptimalResult(
                solution_string=str(solution), summary_value=distance
            )

    def optimal_value(self, variant: str = "standard") -> int:
        match variant:
            case "standard":
                if self.minimum_distance is None:
                    raise RuntimeError("Optimal value not known/provided")
                return self.minimum_distance
            case "maximize":
                if self.maximum_distance is None:
                    raise RuntimeError("Optimal value not known/provided")
                return self.maximum_distance
            case "inverted":
                if self.shifted_maximum_distance is None:
                    raise RuntimeError("Optimal value not known/provided")
                return self.shifted_maximum_distance
            case _:
                raise ValueError(
                    'Variant must be one of "standard", "maximize", "inverted"'
                )

    def reasonable_encoding(self) -> str:
        num_nodes = self.graph.number_of_nodes()
        edge_weights = [
            str(val)
            for i, row in enumerate(nx.adjacency_matrix(self.graph).todense())
            for j, val in enumerate(row)
            if j > i
        ]
        return f"{num_nodes}|{','.join(edge_weights)}"


@register("traveling-salesman-loader")
class TravelingSalesmanLoader(BaseLoader[TravelingSalesmanInstance]):
    @staticmethod
    def load(
        problem_path: str, solution_path: str | None = None
    ) -> TravelingSalesmanInstance:
        g = nx.Graph()
        edge_weight_section = False
        from_vertex, to_vertex = 1, 2
        inversion_shift = None

        with open(problem_path, "r") as f:
            for line in f:
                if edge_weight_section and re.match(r"\d+", line):
                    g.add_edge(from_vertex, to_vertex, weight=int(line))
                    to_vertex += 1
                    if to_vertex > g.number_of_nodes():
                        from_vertex += 1
                        to_vertex = from_vertex + 1
                elif line.startswith("COMMENT : INVERSION_SHIFT="):
                    shift_string = line.split("=")[1].strip()
                    inversion_shift = (
                        None if shift_string == "None" else int(shift_string)
                    )
                elif (
                    line.startswith("NAME")
                    or line.startswith("COMMENT")
                    or not line.strip()
                ):
                    continue  # skip name, comments, and empty lines
                elif line.startswith("TYPE"):
                    if line.split(":")[1].strip() != "TSP":
                        raise ValueError("Problem type is not TSP")
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    if line.split(":")[1].strip() != "EXPLICIT":
                        raise ValueError("Edge weight type is not EXPLICIT")
                elif line.startswith("EDGE_WEIGHT_FORMAT"):
                    if line.split(":")[1].strip() != "UPPER_ROW":
                        raise ValueError("Edge weight format is not UPPER_ROW")
                elif line.startswith("DIMENSION"):
                    num_nodes = int(line.split(":")[1])
                    g.add_nodes_from(range(1, num_nodes + 1))
                elif line.startswith("EDGE_WEIGHT_SECTION"):
                    edge_weight_section = True
                else:
                    raise ValueError(f"Unexpected line: {line}")

        if list(nx.non_edges(g)):
            raise ValueError("TSP graph is not complete (missing edges)")

        inst = TravelingSalesmanInstance(graph=g, inversion_shift=inversion_shift)

        if solution_path:
            with open(solution_path, "r") as f:
                for line in f:
                    if inst.minimum_distance is None:
                        inst.minimum_distance = int(line)
                    elif inst.minimum_ordering is None:
                        inst.minimum_ordering = extract_csloi(line)
                    elif inst.maximum_distance is None:
                        inst.maximum_distance = int(line)
                    elif inst.maximum_ordering is None:
                        inst.maximum_ordering = extract_csloi(line)
                    elif inst.shifted_maximum_distance is None:
                        inst.shifted_maximum_distance = int(line)
                    else:
                        raise ValueError("Unexpected line in solution file")

        return inst

    @staticmethod
    def store(inst: TravelingSalesmanInstance, folder: Path, header: str = "") -> None:
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / "problem.tsp", "w") as f:
            n = inst.graph.number_of_nodes()
            shift = inst.inversion_shift

            if header:
                f.write(header + "\n")
            else:
                f.write(f"NAME : {folder.stem}\n")
            f.write(
                dedent(
                    f"""\
                    TYPE : TSP
                    DIMENSION : {n}
                    EDGE_WEIGHT_TYPE : EXPLICIT
                    EDGE_WEIGHT_FORMAT : UPPER_ROW
                    COMMENT : INVERSION_SHIFT={shift}
                    EDGE_WEIGHT_SECTION
                    """
                )
            )

            adj_mat = nx.adjacency_matrix(inst.graph).todense()
            for i in range(n):
                for j in range(i + 1, n):
                    f.write(f"{adj_mat[i][j]}\n")

        if (
            inst.minimum_distance is not None
            and inst.minimum_ordering is not None
            and inst.maximum_distance is not None
            and inst.maximum_ordering is not None
            and inst.shifted_maximum_distance is not None
        ):
            with open(folder / "solution.sol", "w") as f:
                f.write(str(inst.minimum_distance) + "\n")
                f.write(", ".join(map(str, inst.minimum_ordering)) + "\n")
                f.write(str(inst.maximum_distance) + "\n")
                f.write(", ".join(map(str, inst.maximum_ordering)) + "\n")
                f.write(str(inst.shifted_maximum_distance) + "\n")
