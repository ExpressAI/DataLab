import unittest

from datalabs.tasks.summarization import (
    DialogSummarization,
    MultiDocSummarization,
    MultiRefSummarization,
    QuerySummarization,
    Summarization,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        print(Summarization())
        print(DialogSummarization())
        print(MultiDocSummarization())
        print(MultiRefSummarization())
        print(QuerySummarization())

        # Summarization
        self.assertEqual(Summarization().task, "summarization")
        self.assertEqual(
            Summarization().task_categories, ["conditional-text-generation"]
        )
        self.assertEqual(Summarization().source_column, "text")
        self.assertEqual(Summarization().reference_column, "summary")

        # DialogSummarization
        self.assertEqual(DialogSummarization().task, "dialog-summarization")
        self.assertEqual(DialogSummarization().task_categories, ["summarization"])
        self.assertEqual(DialogSummarization().source_column, "dialogue")
        self.assertEqual(DialogSummarization().reference_column, "summary")

        # Multi-Doc Summarization
        self.assertEqual(MultiDocSummarization().task, "multi-doc-summarization")
        self.assertEqual(MultiDocSummarization().task_categories, ["summarization"])
        self.assertEqual(MultiDocSummarization().source_column, "texts")
        self.assertEqual(MultiDocSummarization().reference_column, "summary")

        # Multi-Ref Summarization
        self.assertEqual(MultiRefSummarization().task, "multi-ref-summarization")
        self.assertEqual(MultiRefSummarization().task_categories, ["summarization"])
        self.assertEqual(MultiRefSummarization().source_column, "text")
        self.assertEqual(MultiRefSummarization().reference_column, "summaries")

        # Query Summarization
        # self.assertEqual(QuerySummarization().task,
        #                    'query-summarization')
        # self.assertEqual(QuerySummarization().task_categories,
        #                    ['summarization', 'guided-conditional-text-generation'])
        # self.assertEqual(QuerySummarization().source_column,
        #                    'text')
        # self.assertEqual(QuerySummarization().reference_column,
        #                    'summary')
        # self.assertEqual(QuerySummarization().guidance_column,
        #                    'query')


if __name__ == "__main__":
    unittest.main()
