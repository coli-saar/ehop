import sys
import unittest

sys.path.insert(1, "../ehop")  # To be run from the top-level ehop directory

from utils.llm_output_utils import extract_csloi


class CSLOIExtractionGood(unittest.TestCase):
    def test_singleton(self):
        input_string = "3"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [3])

    def test_duo(self):
        input_string = "3,2"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [3, 2])

    def test_duo_with_space(self):
        input_string = "3, 2"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [3, 2])

    def test_quartet(self):
        input_string = "1,2,1,2"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [1, 2, 1, 2])

    def test_zero(self):
        input_string = "0"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [0])

    def test_multidigit_numbers(self):
        input_string = "10, 2, 35"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [10, 2, 35])

    def test_with_whitespace_1(self):
        input_string = " 0, 5   "
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [0, 5])

    def test_with_more_whitespace(self):
        input_string = "\n1, 2, 1, 2 \n\t"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [1, 2, 1, 2])

    def test_with_even_more_whitespace(self):
        input_string = "\t\n 3, \t2 \n\n "
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [3, 2])

    def test_with_explanation(self):
        input_string = (
            "I've thought about this.\nHere's what you should do.\nPick these: 3, 2"
        )
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [3, 2])

    def test_with_intro(self):
        input_string = "You should pack the following objects: 3, 2"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [3, 2])

    def test_with_text_followup(self):
        input_string = (
            "You should pack the following objects: 3, 2.\n I hope that helps."
        )
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [3, 2])

    def test_with_intro_and_period(self):
        input_string = "Here is the coloring: 1,2,1,2."
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [1, 2, 1, 2])

    def test_with_intro_and_characters_after(self):
        input_string = "Here is the coloring: 1,2,1,2 :)"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [1, 2, 1, 2])

    def test_with_intro_containing_number(self):
        input_string = "Here is the coloring, which only uses 2 colors: 1,2,1,2"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [1, 2, 1, 2])


class CSLOIExtractionTricky(unittest.TestCase):
    def test_bad_multiline(self):
        input_string = "2, 3\n4, 1"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [4, 1])

    def test_bad_newlines_with_intro(self):
        input_string = "Here are the objects:\n3\n2"
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [2])

    def test_bad_with_intersticial_text(self):
        input_string = "You should take object 1 and object 5."
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [5])


class CSLOIExtractionClearlyBad(unittest.TestCase):
    def test_no_numbers(self):
        input_string = "This is an invalid response."
        csloi = extract_csloi(input_string)
        self.assertEqual(csloi, [])


if __name__ == "__main__":
    unittest.main()
