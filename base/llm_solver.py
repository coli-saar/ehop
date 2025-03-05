import sys
from abc import abstractmethod
from pathlib import Path
from textwrap import dedent

from jinja2 import Environment, FileSystemLoader

from bots import AutoBot, ILPBot

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.bot_and_client import BaseBot, BaseLLMClient
from base.problem_structures import BaseSolver, T_Instance, T_LLMSolution
from llm_clients import load_client


class BaseLLMSolver(BaseSolver[T_LLMSolution, T_Instance]):
    """
    The BaseLLMSolver class provides a structure for solving a problem instance using an LLMClient.
    """

    here: Path  # to be defined by subclasses
    default_demo: T_Instance  # to be defined by subclasses

    def __init__(
        self,
        model: BaseLLMClient | str | None = None,
        costume: str = "textbook",
        variant: str = "standard",
        prompting_strategy: str = "zero_shot",
        demo_inst: T_Instance | None = None,
    ) -> None:
        self.environment = Environment(loader=FileSystemLoader(self.here / "costumes"))
        if model is not None:
            self.set_model(model)
        self.costume = costume
        self.variant = variant
        self.set_template(costume, variant)
        self.set_prompting_strategy(prompting_strategy)
        self.set_demo_inst(demo_inst if demo_inst is not None else self.default_demo)

    def set_model(self, model: BaseLLMClient | str) -> None:
        self.model = model if isinstance(model, BaseLLMClient) else load_client(model)

    def set_template(self, costume: str, variant: str) -> None:
        self.costume = costume
        self.variant = variant
        self.template = self.environment.get_template(
            f"{self.costume}/{self.variant}/template.txt"
        )

    def set_costume(self, costume: str) -> None:
        self.costume = costume
        self.set_template(costume, self.variant)

    def set_variant(self, variant: str) -> None:
        super().set_variant(variant)
        self.set_template(self.costume, variant)

    def set_prompting_strategy(self, prompting_strategy: str) -> None:
        self.prompting_strategy = prompting_strategy

    def set_demo_inst(self, demo_inst: T_Instance) -> None:
        self.demo_inst = demo_inst

    def generate_prompt(
        self, inst: T_Instance, prompting_strategy: str
    ) -> tuple[str, ...] | BaseBot:
        """
        Generates a prompt for the given instance and prompting strategy.
        For prompting strategies that require multiple prompts, returns an object of a subclass of BaseBot.
        """
        prompt = self.template.render(inst=inst)

        match prompting_strategy:
            case "zero_shot":
                return (prompt + "\nPlease add no formatting and no explanations.",)
            case "zero_shot_cot":
                return (
                    prompt
                    + "\nYou may explain your reasoning, but do not add any more explanations once you have produced the comma-separated list.\n\nLet's think step by step.",
                )
            case "zero_shot_one_thought":
                prompt += "\n\nBefore you solve this problem, I'd like you to think about it for a moment. Please express your thoughts about the problem in a few sentences. Do NOT solve the problem yet. Just explain what you're thinking about the problem and how you might go about solving it."
                return AutoBot(
                    [
                        prompt,
                        "Now that you've thought about the problem, provide a solution without any formatting or explanations.",
                    ],
                )
            case "one_shot":
                shot_question = self.generate_prompt(self.demo_inst, "zero_shot")
                shot_answer = (
                    (
                        self.here
                        / f"costumes/{self.costume}/{self.variant}/one_shot_answer.txt"
                    )
                    .read_text()
                    .strip(),
                )
                real_question = self.generate_prompt(inst, "zero_shot")
                if isinstance(shot_question, BaseBot) or isinstance(
                    real_question, BaseBot
                ):
                    raise ValueError("Questions should be of type tuple[str, ...] here")
                return shot_question + shot_answer + real_question
            case "one_shot_cot":
                shot_question = self.generate_prompt(self.demo_inst, "zero_shot_cot")
                shot_answer = (
                    (
                        self.here
                        / f"costumes/{self.costume}/{self.variant}/cot_answer.txt"
                    )
                    .read_text()
                    .strip(),
                )
                if isinstance(shot_question, BaseBot):
                    raise ValueError("Question should be of type tuple[str, ...] here")
                return (
                    shot_question
                    + shot_answer
                    + (
                        f"{prompt}\nYou may explain your reasoning, but do not add any more explanations once you have produced the comma-separated list.",
                    )
                )
            case "ilp_lp":
                prompt += dedent(
                    """

                    Instead of solving the problem, please express it as an Integer Linear Programming (ILP) problem in the LP file format.
                    Here is an example of the LP file format:
                    ```
                    \\ This is a comment

                    Maximize
                      3 a + 2 b + c - 2 d
                    Subject To
                      c0: a + b = 1
                      c1: b - c >= 0
                      c2: a + 2 c <= 4
                    Bounds
                      1 <= c <= 5
                      2 <= d
                    Binaries \\ these are binary variables that can only be 0 or 1
                      a b
                    Generals \\ these are integer variables
                      c d
                    End
                    ```

                    Start by thinking step by step about the variables and constraints you'll need in order to express the problem fully, and then create the specification in the LP format.

                    Do not use constraints with strict inequalities (like `<` or `>`) or non-equalities (like `!=`). Instead, use the operators `<=`, `>=`, and `=`.

                    All constraints must have all terms with variables on the left and all constant terms on the right (note that `a <= b` can be instead written as `a - b <= 0`).

                    Also, remember that coefficients are expressed as integers with a space between the coefficient and the corresponding variable (e.g., `12 x`).

                    Each bound should be specified in terms of a single variable. If a bound involves multiple variables, it should be specified as a constraint in the Subject To section.

                    Finally, note that binary variables are constrained to be 0 or 1, so there is no need to put anything in the "Bounds" section for binary variables. If there are no bounds to provide, the bounds section can simply be omitted.

                    Please provide the ILP problem in the LP format and do not solve the problem yourself.
                    """
                )
                solution_message = "Your ILP problem was successfully solved. Here is the solution:\n\n{}\n\nTranslate this solution back to the original problem and provide it as originally specified.\nDo not add any more explanation once you've provided the solution."
                error_message = 'An error occurred while Gurobi was solving the ILP problem:\n\n{}\n\nPlease try again, prioritizing a complete LP specification over explanations. Keep in mind that all constraints must have only terms with variables on the left and only constant terms on the right. Note that "a <= b" can be instead written as "a - b <= 0".'
                final_error_message = "An error occurred while Gurobi was solving the ILP problem:\n\n{}\n\nInstead of fixing the ILP specification, please provide a final answer for the problem. Do not add any more explanation once you've provided the solution."
                return ILPBot(
                    "lp", prompt, solution_message, error_message, final_error_message
                )
            case "ilp_python":
                prompt += dedent(
                    """

                    Please express this as an Integer Linear Programming (ILP) problem using Python with the gurobipy library.
                    Specifically, define a function named f that returns an optimized `gurobipy.Model` object which represents the problem.
                    Here is an example of the format you should use for your answer:
                    ```
                    from gurobipy import GRB, Model

                    def f():
                        # Create the model
                        model = Model("Example")

                        # Create helper variables
                        n = 5

                        # Add variables
                        vars = model.addMVar((n), vtype=GRB.BINARY, name="vars")

                        # Add constraints
                        model.addConstrs(vars[i] + vars[i + 1] <= 1 for i in range(n - 1))
                        model.addConstr(vars[0] + vars[n - 1] <= 1)

                        # Set objective
                        model.setObjective(vars.sum(), GRB.MAXIMIZE)

                        # Optimize/solve the model
                        model.optimize()

                        # Return the optimized model
                        return model
                    ```

                    Start by thinking step by step about the variables and constraints you'll need in order to express the problem fully, and then define the Python function f.

                    Make sure you import any gurobipy functions you need at the beginning of your code.

                    Note that Gurobi does not support constraints with strict inequalities (like < or >) or non-equalities (like !=).

                    When adding constraints, make sure you use `addConstr` for individual constraints and `addConstrs` for groups of constraints (including ones generated by a loop).

                    Also, note that the expression for the model's objective should not contain a call to a function like `max` or `min`. Instead, you should create a single variable to represent the value you want as your objective and add constraints to the model to ensure that it is the maximum/minimum value (e.g., adding constraints like `model.addConstrs(vars[i] <= my_max for i in range(vars.size))`).

                    Finally, remember that the `@` operator cannot be used for matrix multiplication involving a list, so in such cases you should instead use the `sum` function combined with list comprehensions (e.g., to calculate the dot product of two sequences `vars` and `coefs`, use `sum([vars[i] * coefs[i] for i in range(vars.size)])`).

                    Make sure you only use the gurobipy library for this problem. Do not import any other libraries, and do not provide any code other than what is required to define the function f. Do not call the function or predict its output.
                    """
                )
                solution_message = "Your code was executed successfully. Here are all the variables of the model and their optimal values:\n\n{}\n\nTranslate this solution back to the original problem and provide it as originally specified.\nDo not add any more explanation once you've provided the solution."
                error_message = "An error occurred while executing your code. Here is the output:\n\n{}\n\nPlease try again."
                final_error_message = "An error occurred while executing your code. Here is the output:\n\n{}\n\nInstead of fixing the code, please provide a final answer for the problem. Do not add any more explanation once you've provided the solution."
                return ILPBot(
                    "python",
                    prompt,
                    solution_message,
                    error_message,
                    final_error_message,
                )
            case _:
                raise ValueError(
                    f'Prompting strategy "{self.prompting_strategy}" not recognized'
                )

    def prompt_response(
        self, inst: T_Instance
    ) -> tuple[tuple[str, ...], str] | tuple[tuple[str, ...], tuple[str, ...]]:
        if self.model is None:
            raise RuntimeError(
                "LLMClient is not defined. To set it, use the set_model method or provide it when initializing the solver."
            )

        prompt = self.generate_prompt(inst, self.prompting_strategy)

        if isinstance(prompt, tuple):
            return prompt, self.model.prompt(prompt)
        else:
            return self.model.bot_prompt(prompt)

    @abstractmethod
    def solve(self, inst: T_Instance) -> T_LLMSolution: ...
