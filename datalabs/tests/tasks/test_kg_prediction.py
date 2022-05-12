import unittest

from datalabs.tasks.kg_prediction import KGLinkTailPrediction, KGPrediction


class MyTestCase(unittest.TestCase):
    def test_something(self):

        print(KGLinkTailPrediction().input_schema)
        print(KGLinkTailPrediction())
        print(KGPrediction())

        self.assertEqual(KGLinkTailPrediction().task, "kg-link-tail-prediction")
        self.assertEqual(KGLinkTailPrediction().task_categories, ["kg-prediction"])


if __name__ == "__main__":
    unittest.main()
