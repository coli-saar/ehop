import subprocess
import sys
from itertools import product
from pathlib import Path

import networkx as nx
from gurobipy import GRB, Env, Model
from numpy import argmax

sys.path.insert(1, "../ehop")

from base.problem_structures import BaseSolver
from utils.utils import register

from .model import GraphColoringInstance, GraphColoringLoader, GraphColoringSolution


@register("graph-coloring-symbolic-brute")
class GraphColoringBrute(BaseSolver[GraphColoringSolution, GraphColoringInstance]):
    """A brute-force solver for finding a valid coloring with the least colors possible."""

    def solve(
        self, inst: GraphColoringInstance, num_colors: int | None = None
    ) -> GraphColoringSolution:
        num_colors = num_colors or 1

        while num_colors < inst.graph.number_of_nodes():
            # iterate through every coloring with the current number of colors
            for coloring in product(
                range(1, num_colors + 1), repeat=inst.graph.number_of_nodes()
            ):
                valid = True

                # check the coloring for validity
                for u, v in inst.graph.edges:
                    if coloring[u - 1] == coloring[v - 1]:
                        valid = False
                        break

                if valid:
                    return GraphColoringSolution(coloring=list(coloring))
            num_colors += 1  # too few colors, check next smallest value

        # there was nothing better than giving each vertex a unique color
        return GraphColoringSolution(
            coloring=list(range(1, inst.graph.number_of_nodes() + 1))
        )


@register("graph-coloring-symbolic-ilp")
class GraphColoringILP(BaseSolver[GraphColoringSolution, GraphColoringInstance]):
    """A solver that uses an Integer Linear Program to find a valid coloring with the least colors possible."""

    def solve(self, inst: GraphColoringInstance) -> GraphColoringSolution:
        n = inst.graph.number_of_nodes()
        edges = inst.graph.edges

        with Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            model = Model("Graph Coloring", env=env)

            # Create chromatic number variable and set objective (minimizing # of colors)
            X = model.addVar(vtype=GRB.INTEGER, name="X")
            model.setObjective(X, GRB.MINIMIZE)

            # Create color variables for each node
            colors = model.addMVar((n, n), vtype=GRB.BINARY, name="colors")

            # Each node must be assigned exactly one color
            model.addConstr(colors.sum(axis=1) == 1, name="one_color_per_node")

            # Nodes connected by an edge must have different colors
            model.addConstrs(
                (
                    colors[i - 1, color] + colors[j - 1, color] <= 1
                    for i, j in edges
                    for color in range(n)
                ),
                name="different_colors",
            )

            # Chromatic number must reflect largest color used
            model.addConstrs(
                (
                    sum([colors[node, i] * (i + 1) for i in range(n)]) <= X  # type: ignore
                    for node in range(n)
                ),
                name="count_colors",
            )

            model.optimize()

            # # Code below writes model info to files
            # model.write("gcp.lp")  # the variables, bounds, constraints, and objective
            # model.write("gcp.sol") # the solution (values of all variables)

            coloring = list(argmax(colors.getAttr("X"), axis=1) + 1)

            return GraphColoringSolution(coloring=coloring)


def get_coloring(g: nx.Graph, chromatic_number: int) -> list[int]:
    """A function for finding a coloring of a graph that satisfies the given chromatic number. Raises an error if no such coloring exists."""
    n = g.number_of_nodes()
    strategies = [
        "largest_first",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
        "saturation_largest_first",
    ]
    solved = False

    # try heuristics
    for strat in strategies:
        d = nx.greedy_color(g, strategy=strat)
        if len(set(d.values())) == chromatic_number:
            solved = True
            break

    # resort to brute force
    if not solved:
        brute = GraphColoringBrute()
        brute.solve(GraphColoringInstance(graph=g), num_colors=chromatic_number)

    if not solved:
        raise ValueError("No valid coloring found")

    return [d[v] + 1 for v in range(1, n + 1)]


@register("graph-coloring-symbolic-fast")
class GraphColoringFast(BaseSolver[GraphColoringSolution, GraphColoringInstance]):
    def solve(self, inst: GraphColoringInstance) -> GraphColoringSolution:
        """
        A solver that uses the fastColor algorithm for finding a graph's chromatic number.
        **NOTE**: This solver is only compatible with Linux systems!
        """
        folder = Path("./temp_graph")
        folder.mkdir(exist_ok=True)
        GraphColoringLoader.store(inst, folder)

        result = subprocess.run(
            "./problems/graph_coloring/fastColor/fastColor -f temp_graph/problem.col -t 5".split(),
            capture_output=True,
        )

        print(result.stdout.decode("utf-8"))

        chromatic_number = int(
            result.stdout.decode("utf-8").split("color num =")[1].split()[0]
        )

        coloring = get_coloring(inst.graph, chromatic_number)

        return GraphColoringSolution(coloring=coloring)
