import sys
import unittest

import networkx as nx

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from llm_clients import DummyLLMClient
from problems.graph_coloring.llm import GraphColoringLLM
from problems.graph_coloring.model import GraphColoringInstance

gcp_inst = GraphColoringInstance(nx.Graph([(1, 2), (2, 3), (3, 4)]), chromatic_number=2)


class PromptingStrategies(unittest.TestCase):
    def test_zero_shot(self):
        client = DummyLLMClient()
        solver = GraphColoringLLM(model=client)

        solver.solve(gcp_inst)
        self.assertEqual(
            client.get_history(),
            [
                "I have a network of 4 nodes, numbered 1 to 4, with various nodes being connected to one another. I want to color the nodes such that no two connected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 2 are connected.\nNode 2 and node 3 are connected.\nNode 3 and node 4 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
                "Dummy Response",
            ],
        )

    def test_one_shot(self):
        client = DummyLLMClient()
        solver = GraphColoringLLM(model=client, prompting_strategy="one_shot")

        solver.solve(gcp_inst)
        self.assertEqual(
            client.get_history(),
            [
                "I have a network of 9 nodes, numbered 1 to 9, with various nodes being connected to one another. I want to color the nodes such that no two connected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 2 are connected.\nNode 1 and node 3 are connected.\nNode 1 and node 4 are connected.\nNode 1 and node 5 are connected.\nNode 1 and node 6 are connected.\nNode 1 and node 7 are connected.\nNode 2 and node 6 are connected.\nNode 2 and node 7 are connected.\nNode 2 and node 8 are connected.\nNode 2 and node 9 are connected.\nNode 3 and node 4 are connected.\nNode 3 and node 6 are connected.\nNode 3 and node 7 are connected.\nNode 4 and node 5 are connected.\nNode 4 and node 6 are connected.\nNode 4 and node 9 are connected.\nNode 5 and node 7 are connected.\nNode 5 and node 9 are connected.\nNode 6 and node 9 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
                "4,2,2,3,2,1,1,1,4",
                "I have a network of 4 nodes, numbered 1 to 4, with various nodes being connected to one another. I want to color the nodes such that no two connected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 2 are connected.\nNode 2 and node 3 are connected.\nNode 3 and node 4 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
                "Dummy Response",
            ],
        )


class HistoryState(unittest.TestCase):
    def test_zero_shot(self):
        client = DummyLLMClient()
        solver = GraphColoringLLM(model=client)

        for _ in range(8):
            solver.solve(gcp_inst)
            self.assertEqual(
                client.get_history(),
                [
                    "I have a network of 4 nodes, numbered 1 to 4, with various nodes being connected to one another. I want to color the nodes such that no two connected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 2 are connected.\nNode 2 and node 3 are connected.\nNode 3 and node 4 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
                    "Dummy Response",
                ],
            )

    def test_one_shot(self):
        client = DummyLLMClient()
        solver = GraphColoringLLM(model=client, prompting_strategy="one_shot")

        for _ in range(8):
            solver.solve(gcp_inst)
            self.assertEqual(
                client.get_history(),
                [
                    "I have a network of 9 nodes, numbered 1 to 9, with various nodes being connected to one another. I want to color the nodes such that no two connected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 2 are connected.\nNode 1 and node 3 are connected.\nNode 1 and node 4 are connected.\nNode 1 and node 5 are connected.\nNode 1 and node 6 are connected.\nNode 1 and node 7 are connected.\nNode 2 and node 6 are connected.\nNode 2 and node 7 are connected.\nNode 2 and node 8 are connected.\nNode 2 and node 9 are connected.\nNode 3 and node 4 are connected.\nNode 3 and node 6 are connected.\nNode 3 and node 7 are connected.\nNode 4 and node 5 are connected.\nNode 4 and node 6 are connected.\nNode 4 and node 9 are connected.\nNode 5 and node 7 are connected.\nNode 5 and node 9 are connected.\nNode 6 and node 9 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
                    "4,2,2,3,2,1,1,1,4",
                    "I have a network of 4 nodes, numbered 1 to 4, with various nodes being connected to one another. I want to color the nodes such that no two connected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 2 are connected.\nNode 2 and node 3 are connected.\nNode 3 and node 4 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
                    "Dummy Response",
                ],
            )

    def test_ilp_python(self):
        client = DummyLLMClient(
            "from gurobipy import Model\ndef f():\n\treturn Model('GCP')"
        )
        solver = GraphColoringLLM(model=client, prompting_strategy="ilp_python")

        for _ in range(8):
            solver.solve(gcp_inst)
            self.assertEqual(len(client.get_history()), 4)

    def test_ilp_python_error(self):
        client = DummyLLMClient()
        solver = GraphColoringLLM(model=client, prompting_strategy="ilp_python")

        for _ in range(8):
            try:
                solver.solve(gcp_inst)
            except:
                pass
            self.assertEqual(len(client.get_history()), 2)

    def test_zs_python_alternation(self):
        client = DummyLLMClient(
            "from gurobipy import Model\ndef f():\n\treturn Model('GCP')"
        )
        solver = GraphColoringLLM(model=client)

        for _ in range(8):
            solver.set_prompting_strategy("zero_shot")
            solver.solve(gcp_inst)
            self.assertEqual(len(client.get_history()), 2)
            solver.set_prompting_strategy("ilp_python")
            solver.solve(gcp_inst)
            self.assertEqual(len(client.get_history()), 4)


if __name__ == "__main__":
    unittest.main()
