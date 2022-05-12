import unittest

from datalabs.tasks.sequence_labeling import (
    Chunking,
    NamedEntityRecognition,
    WordSegmentation,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        self.assertEqual(NamedEntityRecognition().task, "named-entity-recognition")
        self.assertEqual(
            NamedEntityRecognition().task_categories, ["sequence-labeling"]
        )

        self.assertEqual(Chunking().task, "chunking")
        self.assertEqual(Chunking().task_categories, ["sequence-labeling"])

        self.assertEqual(WordSegmentation().task, "word-segmentation")
        self.assertEqual(WordSegmentation().task_categories, ["sequence-labeling"])


if __name__ == "__main__":
    unittest.main()
