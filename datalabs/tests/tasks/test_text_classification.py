import unittest

from datalabs.tasks.text_classification import (
    QuestionClassification,
    SentimentClassification,
    TopicClassification,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        self.assertEqual(TopicClassification().task, "topic-classification")
        self.assertEqual(TopicClassification().task_categories, ["text-classification"])

        self.assertEqual(SentimentClassification().task, "sentiment-classification")
        self.assertEqual(
            SentimentClassification().task_categories, ["text-classification"]
        )

        self.assertEqual(QuestionClassification().task, "question-classification")
        self.assertEqual(
            QuestionClassification().task_categories, ["text-classification"]
        )


if __name__ == "__main__":
    unittest.main()
