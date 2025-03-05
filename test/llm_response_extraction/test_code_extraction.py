import sys
import unittest
from textwrap import dedent

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from utils.llm_output_utils import extract_code


class CodeExtractionGood(unittest.TestCase):
    def test_just_code(self):
        input_string = dedent(
            """\
            ```
            # This is a comment
            x = 5
            ```"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "# This is a comment\nx = 5")

    def test_code_with_lang(self):
        input_string = dedent(
            """\
            ```python
            # This is a comment
            x = 5
            ```"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "# This is a comment\nx = 5")

    def test_code_with_intro(self):
        input_string = dedent(
            """\
            Here is some Python code that assigns the value 5 to the variable x:
            ```
            # This is a comment
            x = 5
            ```"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "# This is a comment\nx = 5")

    def test_code_with_outro(self):
        input_string = dedent(
            """\
            ```
            # This is a comment
            x = 5
            ```
            Hope that helps!"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "# This is a comment\nx = 5")

    def test_code_with_intro_andoutro(self):
        input_string = dedent(
            """\
            Here is some Python code that assigns the value 5 to the variable x:
            ```
            # This is a comment
            x = 5
            ```
            Hope that helps!"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "# This is a comment\nx = 5")


class CodeExtractionTricky(unittest.TestCase):
    def test_no_formatting(self):
        input_string = dedent(
            """\
            Here is some Python code that assigns the value 5 to the variable x:
            x = 5"""
        )
        code = extract_code(input_string)
        self.assertEqual(
            code,
            "Here is some Python code that assigns the value 5 to the variable x:\nx = 5",
        )

    def test_cutoff_response(self):
        input_string = dedent(
            """\
            Here is some Python code that assigns the value 5 to the variable x:
            ```
            # This is a comment"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "# This is a comment")

    def test_multiple_blocks(self):
        input_string = dedent(
            """\
            Here is some Python code that assigns the value 5 to the variable x:
            ```
            # This is a comment
            x = 5
            ```
            Now here is the actual answer:
            ```
            def f(x):
                return x + 1
            ```
            Hope that helps!"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "def f(x):\n    return x + 1")

    def test_multiple_blocks_no_signoff(self):
        input_string = dedent(
            """\
            Here is some Python code that assigns the value 5 to the variable x:
            ```
            # This is a comment
            x = 5
            ```
            Now here is the actual answer:
            ```
            def f(x):
                return x + 1
            ```"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "def f(x):\n    return x + 1")

    def test_multiple_blocks_cut_off(self):
        input_string = dedent(
            """\
            Here is some Python code that assigns the value 5 to the variable x:
            ```
            # This is a comment
            x = 5
            ```
            Now here is the actual answer:
            ```
            def f(x):
                return x + 1"""
        )
        code = extract_code(input_string)
        self.assertEqual(code, "def f(x):\n    return x + 1")


if __name__ == "__main__":
    unittest.main()
