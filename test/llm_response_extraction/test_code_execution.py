import sys
import time
import unittest
from textwrap import dedent

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from utils.llm_output_utils import execute_generated_code

execution_test_variable = ""


class CodeExecutionOutput(unittest.TestCase):
    def test_no_output(self):
        input_string = dedent(
            """\
            x = 5
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(output, ("", None))

    def test_simple_output(self):
        input_string = dedent(
            """\
            print("Hello, world!")
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(output, ("Hello, world!", None))

    def test_multiple_outputs(self):
        input_string = dedent(
            """\
            x = 5
            print("Hello, world!")
            x += 1
            print(x)
            print("Goodbye, world!")
            del x
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(output, ("Hello, world!\n6\nGoodbye, world!", None))


class CodeExecutionException(unittest.TestCase):
    def test_simple_exception(self):
        input_string = dedent(
            """\
            raise Exception("This is an exception")
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(output, ("", "Exception at line 1: This is an exception"))

    def test_specific_exception(self):
        input_string = dedent(
            """\
            raise RuntimeError("This is a runtime error")
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(
            output, ("", "RuntimeError at line 1: This is a runtime error")
        )

    def test_multiple_exceptions(self):
        input_string = dedent(
            """\
            raise RuntimeError("This is a runtime error")
            raise ValueError("This is a value error")
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(
            output, ("", "RuntimeError at line 1: This is a runtime error")
        )

    def test_output_and_exception(self):
        input_string = dedent(
            """\
            print("Hello, world!")
            raise Exception("This is an exception")
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(
            output, ("Hello, world!", "Exception at line 2: This is an exception")
        )

    def test_variable_access(self):
        execution_test_variable = "Hello, world!"
        input_string = dedent(
            """\
            print(execution_test_variable)
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(
            output,
            ("", "NameError at line 1: name 'execution_test_variable' is not defined"),
        )

    def test_function_access(self):
        f = lambda x: x + 1
        input_string = dedent(
            """\
            print(f(5))
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(output, ("", "NameError at line 1: name 'f' is not defined"))

    def test_global_variable_access(self):
        global execution_test_variable
        execution_test_variable = "test"
        input_string = dedent(
            """\
            print(execution_test_variable)
            """
        )
        output = execute_generated_code(input_string)
        self.assertEqual(
            output,
            ("", "NameError at line 1: name 'execution_test_variable' is not defined"),
        )


class CodeExecutionTime(unittest.TestCase):
    def test_quick(self):
        input_string = dedent(
            """\
            import time

            print("Sleeping...")
            time.sleep(0.5)
            print("Awake!")
            """
        )
        output = execute_generated_code(input_string, time_limit=5)
        self.assertEqual(output, ("Sleeping...\nAwake!", None))

    def test_just_made_it(self):
        input_string = dedent(
            """\
            import time

            print("Sleeping...")
            time.sleep(0.5)
            print("Awake!")
            """
        )
        output = execute_generated_code(input_string, time_limit=0.6)
        self.assertEqual(output, ("Sleeping...\nAwake!", None))

    def test_too_slow(self):
        input_string = dedent(
            """\
            import time

            print("Sleeping...")
            time.sleep(1)
            print("Awake!")
            """
        )
        output = execute_generated_code(input_string, time_limit=0.5)
        time.sleep(0.75)
        self.assertEqual(
            output,
            (
                "Sleeping...",
                "TimeoutError: Code execution took too long (more than 0.5 second).",
            ),
        )


if __name__ == "__main__":
    unittest.main()
