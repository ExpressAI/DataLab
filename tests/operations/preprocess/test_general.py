import unittest

from datalabs import operations, load_dataset
from preprocess import *


class MyTestCase(unittest.TestCase):



    def test_general(self):

        dataset = load_dataset("qc")

        res = dataset["test"].apply(tokenize_nltk)
        print(next(res))




if __name__ == '__main__':
    unittest.main()






