import unittest

from datalabs.tasks.span_relation_prediction import (
    RelationExtraction,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        # Summarization
        self.assertEqual(RelationExtraction().task, "summarization")
        self.assertEqual(
            RelationExtraction().task_categories, ["span-relation-extraction"]
        )
        self.assertEqual(RelationExtraction().source_column, "text")
        self.assertEqual(RelationExtraction().reference_column, "summary")


if __name__ == "__main__":
    unittest.main()
