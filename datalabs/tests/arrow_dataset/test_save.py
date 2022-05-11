import unittest

from featurize import get_length

from datalabs import load_dataset


class MyTestCase(unittest.TestCase):
    def test_Data_featurize(self):
        dataset = load_dataset("qc")
        dataset_test_new = dataset["test"].apply(get_length)  # noqa
        dataset_test_new2 = dataset["test"].apply(get_length, mode="realtime")  # noqa
        dataset_test_new3 = dataset["test"].apply(get_length, mode="memory")
        # dataset_test_new4 = dataset['test'].apply(get_length, mode="local")

        print(dataset_test_new3.features.keys())
        self.assertEqual(
            list(dataset_test_new3.features.keys()), ["text", "label", "length"]
        )

        # dataset_train_new2 = dataset['train'].apply_local(get_length, "length")
        # save new features into local arrow


if __name__ == "__main__":
    unittest.main()
