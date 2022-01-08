import unittest
import datasets
from datasets import load_dataset

from datasets import Dataset
from featurize import *

class MyTestCase(unittest.TestCase):


    def test_summarization_featurize(self):
        print("\n---- test_Data_PREPROCESSING_REGISTRY ---")


        dataset = load_dataset("cnn_dailymail", "3.0.0")
        res = dataset["test"].apply(get_compression)
        print(next(res))

        res = dataset["test"].apply(get_density)
        print(next(res))

        res = dataset["test"].apply(get_novelty)
        print(next(res))

        res = dataset["test"].apply(get_copy_len)
        print(next(res))

        res = dataset["test"].apply(get_all_features)
        print(next(res))

        subdataset = Dataset.from_dict(dataset["test"][0:3])
        res = subdataset.apply(get_oracle_summary)
        print(next(res))

        subdataset = Dataset.from_dict(dataset["test"][0:3])
        res = subdataset.apply(get_lead_k_summary)
        print(next(res))



if __name__ == '__main__':
    unittest.main()
