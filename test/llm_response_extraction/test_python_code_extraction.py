import sys
import unittest
from textwrap import dedent

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from utils.llm_output_utils import extract_python_code


class PythonCodeExtractionValid(unittest.TestCase):
    def test_sample_code(self):
        input_string = dedent(
            """\
            ```
            x = 5
            print(x)
            ```"""
        )
        code = extract_python_code(input_string)
        self.assertEqual(code, "x = 5\nprint(x)")

    def test_import(self):
        input_string = dedent(
            """\
            ```
            import gurobipy

            model = gurobipy.Model()
            ```"""
        )
        code = extract_python_code(input_string)
        self.assertEqual(code, "import gurobipy\n\nmodel = gurobipy.Model()")

    def test_from_import(self):
        input_string = dedent(
            """\
            ```
            from gurobipy import Model, GRB

            model = Model()
            ```"""
        )
        code = extract_python_code(input_string)
        self.assertEqual(code, "from gurobipy import Model, GRB\n\nmodel = Model()")


class PythonCodeExtractionInvalid(unittest.TestCase):
    def test_disallowed_import(self):
        input_string = dedent(
            """\
            ```
            import os

            os.clear()
            ```"""
        )
        self.assertRaises(ValueError, extract_python_code, input_string)

    def test_good_and_bad_imports(self):
        input_string = dedent(
            """\
            ```
            import gurobipy
            import os

            os.clear()
            ```"""
        )
        self.assertRaises(ValueError, extract_python_code, input_string)

    def test_tricky_good_and_bad_imports(self):
        input_string = dedent(
            """\
            ```
            import gurobipy, os

            os.clear()
            ```"""
        )
        self.assertRaises(ValueError, extract_python_code, input_string)


if __name__ == "__main__":
    unittest.main()
