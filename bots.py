from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Any, Callable

import gurobipy

from base.bot_and_client import BaseBot
from base.problem_structures import BaseInstance, BaseSolution
from base.results import ILPException, classify_result
from utils.ilp import solve_ilp_file
from utils.llm_output_utils import (
    execute_generated_code,
    extract_lp_code,
    extract_python_code,
)


class AutoBot(BaseBot):
    """
    A bot that automatically sends a list of messages in order, regardless of the messages it receives.
    """

    def __init__(
        self, messages: list[str], conversational: bool = False, **kwargs
    ) -> None:
        self.messages_sent = 0
        self.messages = messages
        self.conversational = conversational
        self.kwargs = kwargs or {}

    def is_conversational(self) -> bool:
        return self.conversational

    def get_message(self, llm_response: str) -> tuple[str | None, dict[str, Any]]:
        try:
            response = self.messages[self.messages_sent]
        except IndexError:
            response = None
        else:
            self.messages_sent += 1
        return response, self.kwargs


class OPROBot(BaseBot):
    """
    A bot inspired by the OPRO paper (https://doi.org/10.48550/arXiv.2309.03409).
    """

    def __init__(
        self,
        inst: BaseInstance,
        variant: str,
        trajectory: list[tuple[BaseSolution, int]],
        solution_extraction: Callable[[str], BaseSolution],
        bigger_is_better: bool,
        base_prompt: str,
        trajectory_prefix: str,
        trajectory_suffix: str,
        max_steps: int = 10,
        prompts_per_step: int = 8,
    ) -> None:
        self.inst = inst
        self.variant = variant
        self.trajectory = trajectory
        self.base_prompt = base_prompt
        self.trajectory_prefix = trajectory_prefix
        self.trajectory_suffix = trajectory_suffix
        self.solution_extraction = solution_extraction

        self.messages_sent = 0
        self.max_steps = max_steps
        self.prompts_per_step = prompts_per_step
        self.done = False

        self.curr_solution_batch: list[tuple[BaseSolution, int]] = []
        self.best_solution: list[int] = []
        self.best_score: int | None = None
        self.bigger_is_better = bigger_is_better

        self.trajectory.sort(key=lambda x: x[1], reverse=not self.bigger_is_better)

    def is_conversational(self) -> bool:
        return False

    def get_message(self, llm_response: str) -> tuple[str | None, dict[str, Any]]:
        solution = self.solution_extraction(llm_response)
        # print()
        # print(
        #     self.messages_sent,
        #     llm_response,
        #     [str(pair[0]) for pair in self.trajectory]
        #     + [str(pair[0]) for pair in self.curr_solution_batch],
        #     solution,
        #     solution
        #     in (
        #         [pair[0] for pair in self.trajectory]
        #         + [pair[0] for pair in self.curr_solution_batch]
        #     ),
        # )

        # print(
        #     [solution == pair[0] for pair in self.trajectory + self.curr_solution_batch]
        # )

        if (
            solution
            not in (
                [pair[0] for pair in self.trajectory]
                + [pair[0] for pair in self.curr_solution_batch]
            )
            and solution.get_list() != []
        ):
            # print("Classifying solution")
            result, summary, _ = classify_result(
                self.inst.evaluate(solution, self.variant)
            )

            if result in ["OPTIMAL", "SUBOPTIMAL"]:
                assert (
                    summary is not None
                ), "Summary should not be None for OPTIMAL or SUBOPTIMAL results"

                if self.best_score is None or (
                    (self.bigger_is_better and summary > self.best_score)
                    or (not self.bigger_is_better and summary < self.best_score)
                ):
                    self.best_score = summary
                    self.best_solution = solution.get_list()

                if result == "OPTIMAL":
                    self.done = True
                elif result == "SUBOPTIMAL":
                    self.curr_solution_batch.append((solution, summary))

        if self.done or self.messages_sent == self.max_steps * self.prompts_per_step:
            return None, {}

        # print([str(pair[0]) for pair in self.curr_solution_batch])

        if self.messages_sent % self.prompts_per_step == 0:
            # print("Merging batch")
            self.trajectory.extend(self.curr_solution_batch)
            self.curr_solution_batch = []
            self.trajectory.sort(key=lambda x: x[1], reverse=not self.bigger_is_better)
            # print(
            #     "New Trajectory:",
            #     [str(pair[0]) for pair in self.trajectory],
            # )

        if self.trajectory:
            message = self.base_prompt + "\n\n" + self.trajectory_prefix + "\n\n"
            for solution, score in self.trajectory:
                message += f"Solution: {str(solution)[1:-1]}\nScore: {score}\n\n"
            message += self.trajectory_suffix
        else:
            message = self.base_prompt

        self.messages_sent += 1
        return message, {"temperature": 1.0}


class ILPBot(BaseBot):
    """
    A bot that implements an LLM-based ILP translation pipeline,
    using either LP files or gurobipy-based Python code depending on the mode.
    """

    MAX_LP_TOKENS = 6000
    MAX_PYTHON_TOKENS = 3072

    def __init__(
        self,
        mode: str,
        problem_message: str,
        solution_message: str,
        error_message: str,
        final_error_message: str,
        max_ilp_attempts: int = 1,  # default to no retries
        raise_ilp_exception: bool = True,
    ) -> None:
        if max_ilp_attempts < 1:
            raise ValueError("max_ilp_attempts must be at least 1")
        self.messages_sent = 0
        self.done = False
        if mode not in {"lp", "python"}:
            raise ValueError("mode must be 'lp' or 'python'")
        self.mode = mode
        self.problem_message = problem_message
        self.solution_message = solution_message
        self.error_message = error_message
        self.final_error_message = final_error_message
        self.max_ilp_attempts = max_ilp_attempts
        self.raise_ilp_exception = raise_ilp_exception

    def is_conversational(self) -> bool:
        return True

    def get_message(self, llm_response: str) -> tuple[str | None, dict[str, Any]]:
        if self.done or self.messages_sent > self.max_ilp_attempts:
            message: tuple[str | None, dict[str, Any]] = None, {}
        elif self.messages_sent == 0:
            message = self.problem_message, {
                "max_tokens": (
                    self.MAX_LP_TOKENS if self.mode == "lp" else self.MAX_PYTHON_TOKENS
                )
            }
        else:
            message = (
                self.handle_lp_response(llm_response)
                if self.mode == "lp"
                else self.handle_python_response(llm_response)
            )

        self.messages_sent += 1
        return message

    def handle_lp_response(self, response: str) -> tuple[str | None, dict[str, Any]]:
        # create a unique temporary file (in case of experiments running in parallel)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        f = NamedTemporaryFile(
            mode="w", suffix=".lp", dir=".", prefix=timestamp, delete=False
        )
        f.write(extract_lp_code(response))
        f.flush()
        try:
            obj_val, names, values = solve_ilp_file(f.name)
        except ILPException as e:
            f.close()
            Path(f.name).unlink()
            raise e
        except gurobipy.GurobiError as e:
            f.close()
            Path(f.name).unlink()
            if self.raise_ilp_exception:
                raise ILPException(repr(e))
            message = (
                (self.error_message.format(e), {"max_tokens": self.MAX_LP_TOKENS})
                if self.messages_sent < self.max_ilp_attempts
                else (self.final_error_message.format(e), {})
            )
        else:
            f.close()
            Path(f.name).unlink()
            assignments = "Value Assignments:\n" + "\n".join(
                [f"{name}: {val}" for name, val in zip(names, values)]
            )
            solution_text = f"Objective value: {obj_val}\n" + assignments
            self.done = True
            message = self.solution_message.format(solution_text), {}
        return message

    def handle_python_response(
        self, response: str
    ) -> tuple[str | None, dict[str, Any]]:
        output, error = None, None
        try:
            code = extract_python_code(response)
            extended_code = code + dedent(
                """

                import sys, io

                old_out = sys.stdout
                sys.stdout = io.StringIO()

                model = f()

                sys.stdout = old_out

                for v in model.getVars():
                    print(f"{v.VarName} = {v.X}")

                del model
                """
            )
            output, error = execute_generated_code(extended_code)
        except ValueError as e:
            error = repr(e)

        if error:
            full_output = f"{output + chr(10) if output else ''}{error}"  # chr(10) is a newline character
            if self.raise_ilp_exception:
                raise ILPException(full_output)
            message = (
                (
                    self.error_message.format(full_output),
                    {"max_tokens": self.MAX_PYTHON_TOKENS},
                )
                if self.messages_sent < self.max_ilp_attempts
                else (self.final_error_message.format(full_output), {})
            )
        else:
            self.done = True
            message = self.solution_message.format(output), {}
        return message
