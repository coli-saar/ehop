import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Generic, TypeVar

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from base.bot_and_client import BaseLLMClient
from base.results import Result

# generic type variables
T_Solution = TypeVar("T_Solution", bound="BaseSolution")
T_LLMSolution = TypeVar("T_LLMSolution", bound="BaseLLMSolution")
T_Instance = TypeVar("T_Instance", bound="BaseInstance")
T_Solver = TypeVar("T_Solver", bound="BaseSolver")
T_Loader = TypeVar("T_Loader", bound="BaseLoader")
T_LLMClient = TypeVar("T_LLMClient", bound="BaseLLMClient")


@dataclass
class BaseSolution(ABC):
    """
    The BaseSolution class is a dataclass that represents a solution to a problem instance.
    """

    @abstractmethod
    def __str__(self) -> str: ...


@dataclass
class BaseLLMSolution(BaseSolution):
    """
    The BaseLLMSolution class is a dataclass that represents an LLM's solution to a problem instance.
    """

    def __init__(
        self, prompt: tuple[str, ...], response: str | tuple[str, ...], **kwargs
    ) -> None:
        self.prompt = prompt
        self.response = response
        super().__init__(**kwargs)


class BaseInstance(ABC, Generic[T_Solution]):
    """
    The BaseInstance class is a dataclass that represents an instance of a problem.
    It both stores information about the problem instance and provides a method to evaluate a solution.
    """

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def evaluate(
        self, solution: T_Solution, variant: str = "standard", verbose: bool = False
    ) -> Result: ...

    @abstractmethod
    def reasonable_encoding(self) -> str:
        """
        Provides a string that encodes all of the information about the instance in a "reasonable" way.
        This can be used to measure the scale of the instance, and was inspired by the book Computers and Intractability (Garey & Johnson).
        """
        ...

    @abstractmethod
    def optimal_value(self, variant: str = "standard") -> int:
        """
        Returns the optimal value for the instance, whatever that means for the given problem type.
        """
        ...

    def size(self) -> int:
        """
        Returns an integer that represents the size of the instance based on a reasonable encoding scheme
        to be implemented in the reasonable_encoding method.
        """
        return len(self.reasonable_encoding())


class BaseLoader(ABC, Generic[T_Instance]):
    """
    The BaseLoader class provides a method for loading problem instances from input files
    as well as a method for storing instances as files.
    """

    @staticmethod
    @abstractmethod
    def load(problem_path: str, solution_path: str | None) -> T_Instance: ...

    @staticmethod
    @abstractmethod
    def store(inst: T_Instance, folder: Path, header: str = "") -> None: ...


class BaseSolver(ABC, Generic[T_Solution, T_Instance]):
    """
    The BaseSolver class provides a method for solving a problem instance.
    """

    def __init__(self, variant: str = "standard") -> None:
        self.set_variant(variant)

    def set_variant(self, variant: str) -> None:
        if variant not in {"standard", "inverted"}:
            raise ValueError("Variant must be either 'standard' or 'inverted'.")
        self.variant = variant

    @abstractmethod
    def solve(self, inst: T_Instance) -> T_Solution: ...

    def timed_solve(self, inst: T_Instance) -> tuple[T_Solution, float]:
        """
        Wraps the solving process with a timer to measure the time taken to solve the instance.
        """
        start = perf_counter()
        solution = self.solve(inst)
        time = perf_counter() - start
        return solution, time


class BaseGenerator(ABC, Generic[T_Instance, T_Solver, T_Loader]):
    """
    The BaseGenerator class provides a method for generating problem instances.
    """

    problem_folder: str  # to be defined by subclasses
    scale_descriptor: str  # to be defined by subclasses

    def __init__(
        self, solver: T_Solver, loader: T_Loader, greedy_solver: T_Solver | None = None
    ) -> None:
        self.solver = solver
        self.loader = loader
        self.greedy_solver = greedy_solver

    @abstractmethod
    def generate(self, scale: int, solve: bool = True) -> T_Instance: ...

    @abstractmethod
    def generate_multiple(
        self,
        scale: int,
        num_instances: int,
        distribution: str = "random",
        solve: bool = True,
        verbose: bool = False,
    ) -> list[T_Instance]: ...

    def generate_and_store(
        self,
        scales: list[int],
        instances_per_scale: int,
        distribution: str = "random",
        solve: bool = True,
        to_exclude: list[T_Instance] = [],
        verbose: bool = False,
    ) -> None:
        for scale in scales:
            if verbose:
                print(f"Generating instances of scale {scale}...")
            instances = self.generate_multiple(
                scale, instances_per_scale, distribution, solve, verbose
            )
            if to_exclude:
                if verbose:
                    print(f"Removing any excluded instances in scale {scale}...")
                contains_excluded = any([inst in instances for inst in to_exclude])
                while contains_excluded:
                    for inst in to_exclude:
                        try:
                            instances.remove(inst)
                        except ValueError:
                            continue
                        else:
                            instances += self.generate_multiple(
                                scale, 1, distribution, solve, verbose
                            )
                    contains_excluded = any([inst in instances for inst in to_exclude])
            for i, inst in enumerate(instances):
                dir = Path(
                    f"./data/problem_instances/{self.problem_folder}/in_house/{scale}_{self.scale_descriptor}/in_house_{scale}_{i}/"
                )
                dir.mkdir(parents=True, exist_ok=True)
                self.loader.store(inst, dir)
            if verbose:
                print(f"Generation complete for instances of scale {scale}.{' '*10}")
