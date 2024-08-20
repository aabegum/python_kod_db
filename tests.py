import unittest
from utils import wrap_text_over_words

class TestWrapTitle(unittest.TestCase):

    def test_short_title(self):
        title = "Short title"
        expected = "Short title"
        result = wrap_text_over_words(title, max_line_length=50)
        self.assertEqual(result, expected)

    def test_moderately_long_title(self):
        title = "This is a moderately long title that might need wrapping"
        expected = "This is a moderately\nlong title that\nmight need wrapping"
        result = wrap_text_over_words(title, max_line_length=20)
        self.assertEqual(result, expected)

    def test_very_long_title(self):
        title = ("A very very long title that will certainly need to be "
                 "wrapped across several lines to ensure it fits properly")
        expected = ("A very very long title that\nwill certainly need to be\n"
                    "wrapped across several lines\nto ensure it fits properly")
        result = wrap_text_over_words(title, max_line_length=30)
        self.assertEqual(result, expected)

    def test_different_lengths(self):
        title = ("Another title, but this time with different lengths "
                 "to test how well the function handles different cases")
        expected = ("Another title, but this time with\ndifferent lengths to test how well the\n"
                    "function handles different cases")
        result = wrap_text_over_words(title, max_line_length=40)
        self.assertEqual(result, expected)
