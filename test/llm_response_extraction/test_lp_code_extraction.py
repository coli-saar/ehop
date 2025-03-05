import sys
import unittest
from textwrap import dedent

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from utils.llm_output_utils import extract_lp_code


class LPCodeExtractionValid(unittest.TestCase):
    def test_sample_code(self):
        input_string = dedent(
            """\
            ```
            \\ LP format example

            Maximize
              x + y + z
            Subject To
              c0: x + y = 1
              c1: x + 5 y + 2 z <= 10
              qc0: x + y + [ x ^ 2 - 2 x * y + 3 y ^ 2 ] <= 5
            Bounds
              0 <= x <= 5
              z >= 2
            Generals
              x y z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "\\ LP format example\n\nMaximize\n  x + y + z\nSubject To\n  c0: x + y = 1\n  c1: x + 5 y + 2 z <= 10\n  qc0: x + y + [ x ^ 2 - 2 x * y + 3 y ^ 2 ] <= 5\nBounds\n  0 <= x <= 5\n  z >= 2\nGenerals\n  x y z\nEnd\n",
            code,
        )

    def test_just_code(self):
        input_string = dedent(
            """\
            ```lp
            Maximize
              x + y + z
            Subject To
              c0: x + y = 1
            Bounds
              0 <= x <= 5
              z <= 2
            Generals
              x y z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y = 1\nBounds\n  0 <= x <= 5\n  z <= 2\nGenerals\n  x y z\nEnd\n",
            code,
        )

    def test_multiple_constraints(self):
        input_string = dedent(
            """\
            ```plaintext
            Maximize
              x + y + z
            Subject To
              c0: x + y = 1
              c1: x + z = 2
              c2: x - y <= 2
            Bounds
              0 <= x <= 5
            Generals
              x y z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y = 1\n  c1: x + z = 2\n  c2: x - y <= 2\nBounds\n  0 <= x <= 5\nGenerals\n  x y z\nEnd\n",
            code,
        )

    def test_generals_and_binaries(self):
        input_string = dedent(
            """\
            ```lp
            Maximize
              x + y + z
            Subject To
              c0: x + y = 1
            Bounds
              0 <= x <= 5
              z <= 2
            Generals
              x y
            Binaries
              z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y = 1\nBounds\n  0 <= x <= 5\n  z <= 2\nGenerals\n  x y\nBinaries\n  z\nEnd\n",
            code,
        )


class LPCodeExtractionInvalid(unittest.TestCase):
    def test_variable_on_right(self):
        input_string = dedent(
            """\
            ```
            Maximize
              x + y + z
            Subject To
              c0: x + y = 1
              c1: x + z = y
            Bounds
              0 <= x <= 5
            Generals
              x y z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y = 1\n  c1: x + z - y = 0\nBounds\n  0 <= x <= 5\nGenerals\n  x y z\nEnd\n",
            code,
        )

    def test_multiplied_variable_on_right(self):
        input_string = dedent(
            """\
            ```
            Maximize
              x + y + z
            Subject To
              c0: x + y = 1
              c1: x + z = 12 y
            Bounds
              0 <= x <= 5
            Generals
              x y z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y = 1\n  c1: x + z - 12 y = 0\nBounds\n  0 <= x <= 5\nGenerals\n  x y z\nEnd\n",
            code,
        )

    def test_variables_on_right(self):
        input_string = dedent(
            """\
            ```
            Maximize
              x + y + z
            Subject To
              c0: x + y = 3 z
              c1: x + z = y
            Bounds
              0 <= x <= 5
            Generals
              x y z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y - 3 z = 0\n  c1: x + z - y = 0\nBounds\n  0 <= x <= 5\nGenerals\n  x y z\nEnd\n",
            code,
        )

    def test_list_bound(self):
        input_string = dedent(
            """\
            ```
            Maximize
              x + y + z
            Subject To
              c0: x + y = 4
              c1: x + z = 6
            Bounds
              x, y <= 5
            Generals
              x y z
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y = 4\n  c1: x + z = 6\nBounds\n  x <= 5\n  y <= 5\nGenerals\n  x y z\nEnd\n",
            code,
        )

    def test_longer_list_bound(self):
        input_string = dedent(
            """\
            ```
            Maximize
              x + y + z
            Subject To
              c0: x + y = 4
              c1: x + z = 6
            Bounds
              x, y, z, a, b, c <= 5
            Generals
              x y z a b c
            End
            ```"""
        )
        code = extract_lp_code(input_string)
        self.assertEqual(
            "Maximize\n  x + y + z\nSubject To\n  c0: x + y = 4\n  c1: x + z = 6\nBounds\n  x <= 5\n  y <= 5\n  z <= 5\n  a <= 5\n  b <= 5\n  c <= 5\nGenerals\n  x y z a b c\nEnd\n",
            code,
        )


if __name__ == "__main__":
    unittest.main()
