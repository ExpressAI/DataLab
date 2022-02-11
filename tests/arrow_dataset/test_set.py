import unittest
import math
from datalabs import load_dataset
from aggregate import *

class MyTestCase(unittest.TestCase):
    def test_Data_featurize(self):
        dataset = load_dataset("ag_news")
        new_dataset = dataset['test'].apply(get_average_length)

        print(new_dataset._stat)
        self.assertEqual(math.floor(new_dataset._stat["average_length"]), 38)

if __name__ == '__main__':
    unittest.main()
