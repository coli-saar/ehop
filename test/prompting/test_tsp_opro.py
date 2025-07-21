import sys
import unittest

import networkx as nx

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import BaseSolution
from base.results import classify_result
from bots import OPROBot
from llm_clients import DummyLLMClient
from problems.traveling_salesman.llm import TravelingSalesmanLLM
from problems.traveling_salesman.model import (
    TravelingSalesmanInstance,
    TravelingSalesmanSolution,
)
from problems.traveling_salesman.symbolic import TravelingSalesmanILP
from utils.llm_output_utils import extract_csloi

g = nx.Graph(
    [
        (1, 2, {"weight": 1}),
        (1, 3, {"weight": 7}),
        (1, 4, {"weight": 6}),
        (1, 5, {"weight": 2}),
        (2, 3, {"weight": 2}),
        (2, 4, {"weight": 8}),
        (2, 5, {"weight": 5}),
        (3, 4, {"weight": 1}),
        (3, 5, {"weight": 9}),
        (4, 5, {"weight": 4}),
    ]
)

solver = TravelingSalesmanILP()

tsp_inst = TravelingSalesmanInstance(
    graph=g,
    minimum_ordering=solver.solve(TravelingSalesmanInstance(g)).get_list(),
    inversion_shift=1,
)

sol_1 = TravelingSalesmanSolution(ordering=[1, 2, 3, 5, 4])
sol_2 = TravelingSalesmanSolution(ordering=[1, 2, 4, 3, 5])

half_prompt_1 = "BASE\n\nPRETRAJECTORY\n"
half_prompt_2 = "\nPOSTTRAJECTORY"


class TSPOPRO(unittest.TestCase):
    def test_optimal_first_answer(self):
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )
        bot = OPROBot(
            inst=tsp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertIsNone(bot.get_message("1,2,3,4,5")[0])

    def test_optimal_second_answer(self):
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )
        bot = OPROBot(
            inst=tsp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertTrue(bot.get_message("1,3,4,2,5")[0])
        self.assertIsNone(bot.get_message("1,2,3,4,5")[0])

    def test_repeated_response(self):
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )
        bot = OPROBot(
            inst=tsp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=1,
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        for _ in range(8):
            self.assertEqual(
                bot.get_message("1,3,4,2,5")[0],
                half_prompt_1
                + "\nSolution: 1, 3, 4, 2, 5\nScore: 23\n"
                + half_prompt_2,
            )

    def test_invalid_responses(self):
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )
        bot = OPROBot(
            inst=tsp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=1,
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertEqual(bot.get_message("1,2")[0], "BASE")
        self.assertEqual(bot.get_message("3,1,2,4,5")[0], "BASE")
        self.assertEqual(bot.get_message("BLAH BLAH")[0], "BASE")
        self.assertEqual(bot.get_message("")[0], "BASE")

    def test_step_limit(self):
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )
        bot = OPROBot(
            inst=tsp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        for _ in range(79):
            self.assertTrue(bot.get_message("1,2,4,3,5")[0])
        self.assertIsNone(bot.get_message("1,2,4,3,5")[0])

    def test_response_in_trajectory(self):
        variant = "standard"
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [sol_1, sol_2]:
            result, summary, _ = classify_result(tsp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=tsp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=1,
        )

        message = (
            half_prompt_1
            + "\nSolution: 1, 2, 3, 5, 4\nScore: 22\n\nSolution: 1, 2, 4, 3, 5\nScore: 21\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        for _ in range(8):
            self.assertEqual(bot.get_message("1,2,3,5,4")[0], message)

    def test_trajectory_growth(self):
        variant = "standard"
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [sol_1, sol_2]:
            result, summary, _ = classify_result(tsp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=tsp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=2,
        )

        message = (
            half_prompt_1
            + "\nSolution: 1, 2, 3, 5, 4\nScore: 22\n\nSolution: 1, 2, 4, 3, 5\nScore: 21\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        self.assertEqual(bot.get_message("1,3,2,4,5")[0], message)
        self.assertEqual(
            bot.get_message("1,2,5,4,3")[0],
            half_prompt_1
            + "\nSolution: 1, 3, 2, 4, 5\nScore: 23\n\nSolution: 1, 2, 3, 5, 4\nScore: 22\n\nSolution: 1, 2, 4, 3, 5\nScore: 21\n\nSolution: 1, 2, 5, 4, 3\nScore: 18\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,5,2,3,4")[0],
            half_prompt_1
            + "\nSolution: 1, 3, 2, 4, 5\nScore: 23\n\nSolution: 1, 2, 3, 5, 4\nScore: 22\n\nSolution: 1, 2, 4, 3, 5\nScore: 21\n\nSolution: 1, 2, 5, 4, 3\nScore: 18\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,5,2,3,4")[0],
            half_prompt_1
            + "\nSolution: 1, 3, 2, 4, 5\nScore: 23\n\nSolution: 1, 2, 3, 5, 4\nScore: 22\n\nSolution: 1, 2, 4, 3, 5\nScore: 21\n\nSolution: 1, 2, 5, 4, 3\nScore: 18\n\nSolution: 1, 5, 2, 3, 4\nScore: 16\n"
            + half_prompt_2,
        )

    def test_inverted(self):
        variant = "inverted"
        llm_solver = TravelingSalesmanLLM(
            model=DummyLLMClient(),
            inst=tsp_inst,
        )

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [sol_1, sol_2]:
            result, summary, _ = classify_result(tsp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=tsp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                tsp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=True,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=2,
        )

        message = (
            half_prompt_1
            + "\nSolution: 1, 2, 3, 5, 4\nScore: 28\n\nSolution: 1, 2, 4, 3, 5\nScore: 29\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        self.assertEqual(bot.get_message("1,3,2,4,5")[0], message)
        self.assertEqual(
            bot.get_message("1,2,5,4,3")[0],
            half_prompt_1
            + "\nSolution: 1, 3, 2, 4, 5\nScore: 27\n\nSolution: 1, 2, 3, 5, 4\nScore: 28\n\nSolution: 1, 2, 4, 3, 5\nScore: 29\n\nSolution: 1, 2, 5, 4, 3\nScore: 32\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,5,2,3,4")[0],
            half_prompt_1
            + "\nSolution: 1, 3, 2, 4, 5\nScore: 27\n\nSolution: 1, 2, 3, 5, 4\nScore: 28\n\nSolution: 1, 2, 4, 3, 5\nScore: 29\n\nSolution: 1, 2, 5, 4, 3\nScore: 32\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,5,2,3,4")[0],
            half_prompt_1
            + "\nSolution: 1, 3, 2, 4, 5\nScore: 27\n\nSolution: 1, 2, 3, 5, 4\nScore: 28\n\nSolution: 1, 2, 4, 3, 5\nScore: 29\n\nSolution: 1, 2, 5, 4, 3\nScore: 32\n\nSolution: 1, 5, 2, 3, 4\nScore: 34\n"
            + half_prompt_2,
        )


if __name__ == "__main__":
    unittest.main()
