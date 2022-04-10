import unittest

from featurize import *

from datalabs import load_dataset


class MyTestCase(unittest.TestCase):
    def test_Data_featurize(self):
        dataset = load_dataset("mr")
        dataset_test_new = dataset["test"].apply_save(
            get_length, "length"
        )  #  save new features in memory

        print(dataset_test_new.features.keys())
        self.assertEqual(
            list(dataset_test_new.features.keys()), ["text", "label", "length"]
        )

        # dataset_train_new2 = dataset['train'].apply_local(get_length, "length") # save new features into local arrow


if __name__ == "__main__":
    unittest.main()
