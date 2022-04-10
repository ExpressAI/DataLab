import unittest

from aggregate.sequence_labeling import get_statistics as get_statistics_sl
from aggregate.summarization import get_statistics as get_statistics_sum
from aggregate.text_classification import get_statistics as get_statistics_tc
from aggregate.text_matching import get_statistics as get_statistics_tm

from datalabs import load_dataset


class MyTestCase(unittest.TestCase):
    def test_Data_featurize(self):
        print("\n---- test aggregate operation ---")

        # text classification
        dataset = load_dataset("mr")
        res = dataset["test"].apply(get_statistics_tc)
        print(res._stat["dataset-level"]["length_info"]["max_text_length"])
        self.assertEqual(
            res._stat["dataset-level"]["length_info"]["max_text_length"], 61
        )

        # text matching
        dataset = load_dataset("sick")
        res = dataset["test"].apply(get_statistics_tm)
        print(res._stat["dataset-level"]["length_info"]["max_text1_length"])
        # self.assertEqual(res._stat["dataset-level"]["length_info"]["max_text1_length"], 61)

        # ner
        dataset = load_dataset("conll2003", "ner")
        print(get_statistics_sl)
        print(get_statistics_sl._type)

        res = dataset["test"].apply(get_statistics_sl)
        print(res)
        print(res._stat["dataset-level"]["length_info"]["max_text_length"])
        # self.assertEqual(res._stat["dataset-level"]["length_info"]["max_text_length"], 61)


if __name__ == "__main__":
    unittest.main()
