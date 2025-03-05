import sys
from itertools import permutations

import networkx as nx
import numpy as np
from gurobipy import GRB, Env, Model
from numpy import argmax

sys.path.insert(1, "../ehop")

from base.problem_structures import BaseSolver
from utils.utils import register

from .model import TravelingSalesmanInstance, TravelingSalesmanSolution


@register("traveling-salesman-symbolic-brute")
class TravelingSalesmanBrute(
    BaseSolver[TravelingSalesmanSolution, TravelingSalesmanInstance]
):
    """A brute-force solver for the Traveling Salesman Problem."""

    def solve(self, inst: TravelingSalesmanInstance) -> TravelingSalesmanSolution:
        n = inst.graph.number_of_nodes()
        weight_mat = nx.adjacency_matrix(inst.graph).todense()

        best_tour = None
        best_distance = np.inf if self.variant == "standard" else -np.inf
        comparison = (
            (lambda x, y: x < y) if self.variant == "standard" else (lambda x, y: x > y)
        )

        for perm in permutations(range(1, n)):
            ordering = [0] + list(perm) + [0]
            distance = sum([weight_mat[ordering[i], ordering[i + 1]] for i in range(n)])

            if comparison(distance, best_distance):
                best_distance = distance
                best_tour = ordering[:-1]

        if best_tour is None:
            raise ValueError("No valid tour found")

        ordering = [x + 1 for x in best_tour]

        return TravelingSalesmanSolution(ordering=ordering)


@register("traveling-salesman-symbolic-ilp")
class TravelingSalesmanILP(
    BaseSolver[TravelingSalesmanSolution, TravelingSalesmanInstance]
):
    """A solver that uses an Integer Linear Program to find an optimal solution."""

    def solve(self, inst: TravelingSalesmanInstance) -> TravelingSalesmanSolution:
        n = inst.graph.number_of_nodes()
        weight_mat = nx.adjacency_matrix(inst.graph).todense()

        with Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            model = Model("Traveling Salesman", env=env)

            # Create distance variable and set objective (minimizing distance)
            D = model.addVar(vtype=GRB.INTEGER, name="D")

            # Create variables for each edge (both directions)
            edges = model.addMVar((n, n), vtype=GRB.BINARY, name="edges")

            # No self-loops
            model.addConstr(edges.diagonal().sum() == 0, name="no_self_loops")

            # Each node must be arrived at and left from exactly once
            model.addConstr(edges.sum(axis=0) == 1, name="one_arrival_per_node")
            model.addConstr(edges.sum(axis=1) == 1, name="one_departure_per_node")

            # Subtour elimination constraints (Miller-Tucker-Zemlin formulation: https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation)
            counters = model.addMVar(n, vtype=GRB.INTEGER, name="counters")
            model.addConstrs(
                counters[i] - counters[j] + 1 <= (n - 1) * (1 - edges[i, j])
                for i in range(1, n)
                for j in range(1, n)
                if i != j
            )
            model.addConstrs(2 <= counters[i] for i in range(1, n))
            model.addConstrs(counters[i] <= n for i in range(1, n))

            # Distance variable must equal the sum of the weights of the edges
            model.addConstr(
                (
                    sum(  # type: ignore
                        [
                            edges[source, target] * weight_mat[source, target]
                            for source in range(n)
                            for target in range(n)
                        ]
                    )
                    == D
                ),
                name="calculate_distance",
            )

            model.setObjective(
                D, GRB.MINIMIZE if self.variant == "standard" else GRB.MAXIMIZE
            )

            model.optimize()

            # Code below writes model info to files
            # model.write("tsp.lp")  # the variables, bounds, constraints, and objective
            # model.write("tsp.sol")  # the solution (values of all variables)

            tour_matrix = edges.getAttr("X")

            node_sequence = [0]
            while len(node_sequence) < n:
                # print(tour_matrix)
                last_node = node_sequence[-1]
                next_node = int(argmax(tour_matrix[last_node]))
                node_sequence.append(next_node)
                tour_matrix[last_node][next_node] = 0
                # tour_matrix[next_node][last_node] = 0

            ordering = [x + 1 for x in node_sequence]

            return TravelingSalesmanSolution(ordering=ordering)
