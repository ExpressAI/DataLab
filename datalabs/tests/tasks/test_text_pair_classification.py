import unittest

from datalabs.tasks.text_pair_classification import (
    NaturalLanguageInference,
    TextPairClassification,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        # text pair classification
        self.assertEqual(TextPairClassification().task, "text-pair-classification")
        self.assertEqual(TextPairClassification().task_categories, ["ROOT"])
        self.assertEqual(TextPairClassification().text1_column, "text1")
        self.assertEqual(TextPairClassification().text2_column, "text2")

        # nli
        self.assertEqual(NaturalLanguageInference().task, "natural-language-inference")
        self.assertEqual(
            NaturalLanguageInference().task_categories, ["text-pair-classification"]
        )
        self.assertEqual(NaturalLanguageInference().text1_column, "text1")
        self.assertEqual(NaturalLanguageInference().text2_column, "text2")


if __name__ == "__main__":
    unittest.main()
