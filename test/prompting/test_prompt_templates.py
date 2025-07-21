import sys
import unittest

import networkx as nx

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from llm_clients import DummyLLMClient
from problems.graph_coloring.llm import GraphColoringLLM
from problems.graph_coloring.model import GraphColoringInstance
from problems.knapsack.llm import KnapsackLLM
from problems.knapsack.model import KnapsackInstance
from problems.traveling_salesman.llm import TravelingSalesmanLLM
from problems.traveling_salesman.model import TravelingSalesmanInstance

gcp_inst = GraphColoringInstance(
    nx.Graph([(1, 2), (2, 3), (3, 4)]),
    chromatic_number=2,
)

ksp_inst = KnapsackInstance(
    num_items=3,
    profits=[1, 2, 3],
    weights=[4, 5, 6],
    capacity=9,
    optimal_items=[0, 1],
)

tsp_inst = TravelingSalesmanInstance(
    graph=nx.Graph(
        [
            (1, 2, {"weight": 3}),
            (1, 3, {"weight": 8}),
            (1, 4, {"weight": 4}),
            (2, 3, {"weight": 5}),
            (2, 4, {"weight": 9}),
            (3, 4, {"weight": 6}),
        ]
    ),
    minimum_ordering=[1, 2, 3, 4],
    inversion_shift=1,
)


class GraphColoring(unittest.TestCase):
    def test_prompt_response_reasoning(self):
        client = DummyLLMClient(reasoning="I can think!")
        solver = GraphColoringLLM(model=client)

        solution = solver.solve(gcp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I have a network of 4 nodes, numbered 1 to 4, with various nodes being connected to one another. I want to color the nodes such that no two connected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 2 are connected.\nNode 2 and node 3 are connected.\nNode 3 and node 4 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
            ),
        )
        self.assertEqual(solution.response, "Dummy Response")
        self.assertEqual(solution.reasoning, "I can think!")

    def test_inverted(self):
        client = DummyLLMClient()
        solver = GraphColoringLLM(model=client, variant="inverted")

        solution = solver.solve(gcp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I have a network of 4 nodes, numbered 1 to 4, with various nodes being connected to one another. I want to color the nodes such that no two unconnected nodes have the same color.\n\nThe connections are as follows:\nNode 1 and node 3 are connected.\nNode 1 and node 4 are connected.\nNode 2 and node 4 are connected.\n\nHow can I color the nodes using the fewest colors possible? Generate a comma-separated list of the colors for each node, where the colors are represented by integers ranging from 1 to the number of colors used. The colors should be in the order of the vertices, so the first color will correspond to node 1, the second color will correspond to node 2, and so on.\nPlease add no formatting and no explanations.",
            ),
        )

    def test_costumed(self):
        client = DummyLLMClient()
        solver = GraphColoringLLM(model=client, costume="student_groups")

        solution = solver.solve(gcp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I am a teacher, and I want to assign my 4 students to different groups. I need the groups to focus, so I need to make sure that no two students who are friends with one another are in the same group, otherwise they may get distracted. I don't need the groups to all be the same size, but I want to minimize the total number of groups.\n\nThe friendships are as follows:\nStudent 1 and student 2 are friends.\nStudent 2 and student 3 are friends.\nStudent 3 and student 4 are friends.\n\nWhich group should each student be assigned to? Generate a comma-separated list with each student's group, where the groups are represented by integers ranging from 1 to the total number of groups. The groups should be in the order of the students' numbers, so the first group in the list will correspond to student 1, the second group will correspond to student 2, and so on.\nPlease add no formatting and no explanations.",
            ),
        )


class Knapsack(unittest.TestCase):
    def test_prompt_response_reasoning(self):
        client = DummyLLMClient()
        solver = KnapsackLLM(model=client)

        solution = solver.solve(ksp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I am trying to fill a bag with valuable items. Each item has a weight and a value.\n\nHere are the items I have:\nItem 1 has a weight of 4 kg and a value of 1 €.\nItem 2 has a weight of 5 kg and a value of 2 €.\nItem 3 has a weight of 6 kg and a value of 3 €.\n\nWhich items should I pack to get the most value possible while also making sure the total weight of the items does not exceed the bag's capacity of 9 kg? Generate a comma-separated list of the items I should put in the bag, where each item is represented by its number.\nPlease add no formatting and no explanations.",
            ),
        )

    def test_inverted(self):
        client = DummyLLMClient()
        solver = KnapsackLLM(model=client, variant="inverted")

        solution = solver.solve(ksp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I am trying to fill a bag with worthless items. Each item has a weight and a value.\n\nHere are the items I have:\nItem 1 has a weight of 4 kg and a value of 1 €.\nItem 2 has a weight of 5 kg and a value of 2 €.\nItem 3 has a weight of 6 kg and a value of 3 €.\n\nWhich items should I pack to get the least value possible while also making sure the total weight of the items is at least 6 kg? Generate a comma-separated list of the items I should put in the bag, where each item is represented by its number.\nPlease add no formatting and no explanations.",
            ),
        )

    def test_costumed(self):
        client = DummyLLMClient()
        solver = KnapsackLLM(model=client, costume="lemonade_stand")

        solution = solver.solve(ksp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I am running a lemonade stand where I don't set a single price but rather let the customers make custom offers. Each customer is offering a specific amount of money for a specific amount of lemonade. Each offer is rigid, so I can only fulfill it exactly as stated or not fulfill it at all.\n\nI have the following offers:\nCustomer 1 is offering $1 for 4 gallons of lemonade.\nCustomer 2 is offering $2 for 5 gallons of lemonade.\nCustomer 3 is offering $3 for 6 gallons of lemonade.\n\nWhich customers' offers should I take up to make my revenue as large as possible given that I can't sell more than 9 total gallons of lemonade? Generate a comma-separated list of the customers whose offers I should take up, where each customer is represented by their number.\nPlease add no formatting and no explanations.",
            ),
        )


class TravelingSalesman(unittest.TestCase):
    def test_prompt_response_reasoning(self):
        client = DummyLLMClient()
        solver = TravelingSalesmanLLM(model=client)

        solution = solver.solve(tsp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I am planning a trip to visit several cities. Here are the distances between each pair of cities:\n\nCity 1 and city 2 are 3 miles apart.\nCity 1 and city 3 are 8 miles apart.\nCity 1 and city 4 are 4 miles apart.\nCity 2 and city 3 are 5 miles apart.\nCity 2 and city 4 are 9 miles apart.\nCity 3 and city 4 are 6 miles apart.\n\nWhat is the shortest possible route that starts at city 1, visits each city exactly once, and returns to city 1? Please generate a comma-separated list of the cities in the order I should visit them, where the cities are represented by their respective numbers.\nPlease add no formatting and no explanations.",
            ),
        )

    def test_inverted(self):
        client = DummyLLMClient()
        solver = TravelingSalesmanLLM(model=client, variant="inverted")

        solution = solver.solve(tsp_inst)
        self.assertEqual(
            solution.prompting,
            (
                "I am planning a trip to visit several cities. Here are the distances between each pair of cities:\n\nCity 1 and city 2 are 7 miles apart.\nCity 1 and city 3 are 2 miles apart.\nCity 1 and city 4 are 6 miles apart.\nCity 2 and city 3 are 5 miles apart.\nCity 2 and city 4 are 1 miles apart.\nCity 3 and city 4 are 4 miles apart.\n\nWhat is the longest possible route that starts at city 1, visits each city exactly once, and returns to city 1? Please generate a comma-separated list of the cities in the order I should visit them, where the cities are represented by their respective numbers.\nPlease add no formatting and no explanations.",
            ),
        )

    def test_costumed(self):
        client = DummyLLMClient()
        solver = TravelingSalesmanLLM(model=client, costume="un_seating")

        solution = solver.solve(tsp_inst)
        self.assertEqual(
            solution.prompting,
            (
                'I am responsible for the seating assignments at an upcoming UN meeting. There will be representatives from 4 nations sitting at a round table. The representative from nation 1 will be leading the discussion, so they will be sitting in the designated "Director Seat," but nothing else is decided yet. There is some amount of political tension between each pair of nations, and I\'ve been given a list of tension scores for each pair of representatives, with higher scores indicating higher tension. Here are the tension levels between each pair of representatives:\n\nRepresentative 1 and representative 2 have tension score 3.\nRepresentative 1 and representative 3 have tension score 8.\nRepresentative 1 and representative 4 have tension score 4.\nRepresentative 2 and representative 3 have tension score 5.\nRepresentative 2 and representative 4 have tension score 9.\nRepresentative 3 and representative 4 have tension score 6.\n\nI want to minimize the total tension between adjacent pairs of representatives to prevent the discussion from getting heated. What should the seating order be, starting at the Director Seat and continuing clockwise? Note that the last person in the ordering will also be sitting next to the Director Seat. Please generate a comma-separated list of the representatives in the order they should be seated, where the representatives are represented by their respective numbers.\nPlease add no formatting and no explanations.',
            ),
        )


if __name__ == "__main__":
    unittest.main()
