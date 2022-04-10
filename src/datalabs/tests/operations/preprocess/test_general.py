import unittest

from preprocess import *

from datalabs import load_dataset, operations


class MyTestCase(unittest.TestCase):
    def test_general(self):

        dataset = load_dataset("ag_news")

        res = dataset["test"].apply(tokenize_nltk)
        print(next(res))


if __name__ == "__main__":
    unittest.main()
