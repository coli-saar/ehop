import sys
import unittest

import networkx as nx

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.problem_structures import BaseSolution
from base.results import classify_result
from bots import OPROBot
from llm_clients import DummyLLMClient
from problems.graph_coloring.llm import GraphColoringLLM
from problems.graph_coloring.model import GraphColoringInstance, GraphColoringSolution
from problems.knapsack.llm import KnapsackLLM
from problems.knapsack.model import KnapsackInstance, KnapsackSolution
from utils.llm_output_utils import extract_csloi

g = nx.empty_graph(5)
graph = nx.relabel_nodes(g, {i: i + 1 for i in range(5)})
graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

gcp_inst = GraphColoringInstance(graph=graph, optimal_coloring=[1, 2, 1, 2, 1])

gcp_sol_1 = GraphColoringSolution([1, 2, 3, 4, 5])
gcp_sol_2 = GraphColoringSolution([1, 3, 1, 2, 1])


half_prompt_1 = "BASE\n\nPRETRAJECTORY\n"
half_prompt_2 = "\nPOSTTRAJECTORY"


class GCPOPRO(unittest.TestCase):
    def test_optimal_first_answer(self):
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)
        bot = OPROBot(
            inst=gcp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertIsNone(bot.get_message("1,2,1,2,1")[0])
        self.assertEqual(bot.best_solution, [1, 2, 1, 2, 1])

    def test_optimal_second_answer(self):
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)
        bot = OPROBot(
            inst=gcp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=False,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertTrue(bot.get_message("1,3,2,4,1")[0])
        self.assertIsNone(bot.get_message("1,2,1,2,1")[0])
        self.assertEqual(bot.best_solution, [1, 2, 1, 2, 1])

    def test_repeated_response(self):
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)
        bot = OPROBot(
            inst=gcp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
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
                bot.get_message("1,3,4,2,1")[0],
                half_prompt_1 + "\nSolution: 1, 3, 4, 2, 1\nScore: 4\n" + half_prompt_2,
            )
        self.assertEqual(bot.best_solution, [1, 3, 4, 2, 1])

    def test_invalid_responses(self):
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)
        bot = OPROBot(
            inst=gcp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
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
        self.assertEqual(bot.get_message("3,1,2,4,5,6,7")[0], "BASE")
        self.assertEqual(bot.get_message("BLAH BLAH")[0], "BASE")
        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertEqual(bot.best_solution, [])
        self.assertIsNone(bot.best_score)

    def test_step_limit(self):
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)
        bot = OPROBot(
            inst=gcp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
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
        self.assertEqual(bot.best_solution, [1, 2, 4, 3, 5])

    def test_response_in_trajectory(self):
        variant = "standard"
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [gcp_sol_1, gcp_sol_2]:
            result, summary, _ = classify_result(gcp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=gcp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
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
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        for _ in range(8):
            self.assertEqual(bot.get_message("1,3,1,2")[0], message)

    def test_trajectory_growth(self):
        variant = "standard"
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [gcp_sol_1, gcp_sol_2]:
            result, summary, _ = classify_result(gcp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=gcp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
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
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        self.assertEqual(bot.get_message("1,3,2,4,1")[0], message)
        self.assertEqual(
            bot.get_message("1,2,3,2,1")[0],
            half_prompt_1
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 2, 4, 1\nScore: 4\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n\nSolution: 1, 2, 3, 2, 1\nScore: 3\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,3,2,4,1")[0],
            half_prompt_1
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 2, 4, 1\nScore: 4\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n\nSolution: 1, 2, 3, 2, 1\nScore: 3\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,2,3,4,1")[0],
            half_prompt_1
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 2, 4, 1\nScore: 4\n\nSolution: 1, 2, 3, 4, 1\nScore: 4\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n\nSolution: 1, 2, 3, 2, 1\nScore: 3\n"
            + half_prompt_2,
        )

    def test_inverted(self):
        variant = "inverted"
        llm_solver = GraphColoringLLM(model=DummyLLMClient(), inst=gcp_inst)

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [gcp_sol_1, gcp_sol_2]:
            result, summary, _ = classify_result(gcp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=gcp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                gcp_inst, ("",), "", None, extract_csloi(s)
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
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        self.assertEqual(bot.get_message("1,3,2,4,1")[0], message)
        self.assertEqual(
            bot.get_message("1,2,3,2,1")[0],
            half_prompt_1
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 2, 4, 1\nScore: 4\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n\nSolution: 1, 2, 3, 2, 1\nScore: 3\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,3,2,4,1")[0],
            half_prompt_1
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 2, 4, 1\nScore: 4\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n\nSolution: 1, 2, 3, 2, 1\nScore: 3\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,2,3,4,1")[0],
            half_prompt_1
            + "\nSolution: 1, 2, 3, 4, 5\nScore: 5\n\nSolution: 1, 3, 2, 4, 1\nScore: 4\n\nSolution: 1, 2, 3, 4, 1\nScore: 4\n\nSolution: 1, 3, 1, 2, 1\nScore: 3\n\nSolution: 1, 2, 3, 2, 1\nScore: 3\n"
            + half_prompt_2,
        )


ksp_inst = KnapsackInstance(
    num_items=5,
    profits=[1, 2, 3, 4, 5],
    weights=[1, 2, 3, 4, 5],
    capacity=8,
    optimal_items=[2, 4],
)

ksp_sol_1 = KnapsackSolution([0, 1])
ksp_sol_2 = KnapsackSolution([1, 2])


inv_ksp_sol_1 = KnapsackSolution([2, 3, 4])
inv_ksp_sol_2 = KnapsackSolution([0, 3, 4])


class KSPOPRO(unittest.TestCase):
    # TODO: Implement the Knapsack OPRO tests
    def test_optimal_first_answer(self):
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)
        bot = OPROBot(
            inst=ksp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=True,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertIsNone(bot.get_message("3,5")[0])

    def test_optimal_second_answer(self):
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)
        bot = OPROBot(
            inst=ksp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=True,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertTrue(bot.get_message("1,2")[0])
        self.assertIsNone(bot.get_message("3,5")[0])

    def test_repeated_response(self):
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)
        bot = OPROBot(
            inst=ksp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=True,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=1,
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        for _ in range(8):
            self.assertEqual(
                bot.get_message("1,3")[0],
                half_prompt_1 + "\nSolution: 1, 3\nScore: 4\n" + half_prompt_2,
            )

    def test_invalid_responses(self):
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)
        bot = OPROBot(
            inst=ksp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=True,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=1,
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        self.assertEqual(bot.get_message("1,7")[0], "BASE")
        self.assertEqual(bot.get_message("1,2,3,4")[0], "BASE")
        self.assertEqual(bot.get_message("BLAH BLAH")[0], "BASE")
        self.assertEqual(bot.get_message("")[0], "BASE")

    def test_step_limit(self):
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)
        bot = OPROBot(
            inst=ksp_inst,
            variant="standard",
            trajectory=[],
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=True,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
        )

        self.assertEqual(bot.get_message("")[0], "BASE")
        for _ in range(79):
            self.assertTrue(bot.get_message("1,2")[0])
        self.assertIsNone(bot.get_message("1,2")[0])

    def test_response_in_trajectory(self):
        variant = "standard"
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [ksp_sol_1, ksp_sol_2]:
            result, summary, _ = classify_result(ksp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=ksp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
            ),
            bigger_is_better=True,
            base_prompt="BASE",
            trajectory_prefix="PRETRAJECTORY",
            trajectory_suffix="POSTTRAJECTORY",
            max_steps=10,
            prompts_per_step=1,
        )

        message = (
            half_prompt_1
            + "\nSolution: 1, 2\nScore: 3\n\nSolution: 2, 3\nScore: 5\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        for _ in range(8):
            self.assertEqual(bot.get_message("1,2")[0], message)

    def test_trajectory_growth(self):
        variant = "standard"
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [ksp_sol_1, ksp_sol_2]:
            result, summary, _ = classify_result(ksp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=ksp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
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
            + "\nSolution: 1, 2\nScore: 3\n\nSolution: 2, 3\nScore: 5\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        self.assertEqual(bot.get_message("1,5")[0], message)
        self.assertEqual(
            bot.get_message("2,5")[0],
            half_prompt_1
            + "\nSolution: 1, 2\nScore: 3\n\nSolution: 2, 3\nScore: 5\n\nSolution: 1, 5\nScore: 6\n\nSolution: 2, 5\nScore: 7\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("2,4")[0],
            half_prompt_1
            + "\nSolution: 1, 2\nScore: 3\n\nSolution: 2, 3\nScore: 5\n\nSolution: 1, 5\nScore: 6\n\nSolution: 2, 5\nScore: 7\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("2")[0],
            half_prompt_1
            + "\nSolution: 2\nScore: 2\n\nSolution: 1, 2\nScore: 3\n\nSolution: 2, 3\nScore: 5\n\nSolution: 1, 5\nScore: 6\n\nSolution: 2, 4\nScore: 6\n\nSolution: 2, 5\nScore: 7\n"
            + half_prompt_2,
        )
        self.assertEqual(bot.best_solution, [2, 5])

    def test_inverted(self):
        variant = "inverted"
        llm_solver = KnapsackLLM(model=DummyLLMClient(), inst=ksp_inst)

        trajectory: list[tuple[BaseSolution, int]] = []
        for sol in [inv_ksp_sol_1, inv_ksp_sol_2]:
            result, summary, _ = classify_result(ksp_inst.evaluate(sol, variant))
            if result != "SUBOPTIMAL" or summary is None:
                continue
            trajectory.append((sol, summary))

        bot = OPROBot(
            inst=ksp_inst,
            variant=variant,
            trajectory=trajectory,
            solution_extraction=lambda s: llm_solver.extract_solution(
                ksp_inst, ("",), "", None, extract_csloi(s)
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
            + "\nSolution: 3, 4, 5\nScore: 12\n\nSolution: 1, 4, 5\nScore: 10\n"
            + half_prompt_2
        )

        self.assertEqual(bot.get_message("")[0], message)
        self.assertEqual(bot.get_message("2,3,4")[0], message)
        self.assertEqual(
            bot.get_message("1,3,4")[0],
            half_prompt_1
            + "\nSolution: 3, 4, 5\nScore: 12\n\nSolution: 1, 4, 5\nScore: 10\n\nSolution: 2, 3, 4\nScore: 9\n\nSolution: 1, 3, 4\nScore: 8\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,3,5")[0],
            half_prompt_1
            + "\nSolution: 3, 4, 5\nScore: 12\n\nSolution: 1, 4, 5\nScore: 10\n\nSolution: 2, 3, 4\nScore: 9\n\nSolution: 1, 3, 4\nScore: 8\n"
            + half_prompt_2,
        )
        self.assertEqual(
            bot.get_message("1,3,4,5")[0],
            half_prompt_1
            + "\nSolution: 1, 3, 4, 5\nScore: 13\n\nSolution: 3, 4, 5\nScore: 12\n\nSolution: 1, 4, 5\nScore: 10\n\nSolution: 2, 3, 4\nScore: 9\n\nSolution: 1, 3, 5\nScore: 9\n\nSolution: 1, 3, 4\nScore: 8\n"
            + half_prompt_2,
        )

        self.assertEqual(bot.best_solution, [1, 3, 4])


if __name__ == "__main__":
    unittest.main()
