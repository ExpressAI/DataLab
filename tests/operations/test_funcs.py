import unittest

from data import TextData
from featurize import *

from datalabs import load_dataset

"""
from datalab.operations.edit.core import add_typos_checklist
from datalab import load_dataset
dataset = load_dataset("ag_news")
dataset["test"].apply(add_typos_checklist)



from datalab.operations.featurize.text_classification import get_length
from datalab import load_dataset
dataset = load_dataset("ag_news")
res = dataset["test"].apply(get_length)


"""


class MyTestCase(unittest.TestCase):
    def test_Data_PREPROCESSING_REGISTRY(self):
        print("\n---- test_Data_PREPROCESSING_REGISTRY ---")

        a = ["I love this movie", "do you love this movie"]
        A = TextData(a)

        # print(A.data)

        B = A.apply(get_length)
        # for b in B:
        #     print(b)

    # def test_Data_PREPROCESSING_REGISTRY(self):
    #     print("\n---- test_Data_PREPROCESSING_REGISTRY ---")
    #
    #     a = ['I love this movie', 'do you love this movie']
    #     A = TextData(a)
    #
    #     print(A.data)
    #     B = A.apply(preprocess.core.tokenize_nltk)
    #     for b in B:
    #         print(b)
    # #
    # #
    def test_Data_featurize(self):
        print("\n---- test_Data_featurize ---")

        a = [
            "I love this movie.",
            "apple is looking at buying U.K. startup for $1 billion.",
        ]
        A = TextData(a)

        B = A.apply(get_length)
        # for b in B:
        #     print(b)

        B = A.apply(get_entities_spacy)
        # for b in B:
        #     print(b)

        B = A.apply(get_postag_nltk)
        # for b in B:
        #     print(b)

        B = A.apply(get_postag_spacy)
        # for b in B:
        #     print(b)


if __name__ == "__main__":
    unittest.main()
