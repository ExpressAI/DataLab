import unittest

from datalabs.tasks.span_text_classification import (
    AspectBasedSentimentClassification,
    SpanTextClassification,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        print(AspectBasedSentimentClassification().input_schema)
        print(AspectBasedSentimentClassification())
        print(SpanTextClassification())

        self.assertEqual(
            AspectBasedSentimentClassification().task,
            "aspect-based-sentiment-classification",
        )
        self.assertEqual(
            AspectBasedSentimentClassification().task_categories,
            ["span-text-classification"],
        )
        self.assertIsNotNone(AspectBasedSentimentClassification().input_schema)


if __name__ == "__main__":
    unittest.main()
