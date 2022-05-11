import unittest

from aggregate import get_average_length
from featurize import get_length

from datalabs import load_dataset


class MyTestCase(unittest.TestCase):
    def test_Data_featurize(self):
        dataset = load_dataset("qc")["test"]
        new_dataset_one = dataset.apply(get_length, mode="memory", prefix="test")
        new_dataset_two = dataset.apply(get_average_length)

        print(new_dataset_one[0]["test_length"])
        print(new_dataset_two._stat)

        self.assertEqual(len(new_dataset_one[0].keys()), 3)
        self.assertEqual(len(new_dataset_two._stat.keys()), 1)


if __name__ == "__main__":
    unittest.main()
