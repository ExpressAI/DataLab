import unittest

from datalabs.operations.data import TextData
from datalabs.operations.featurize.general import (
    get_basic_words,
    get_gender_bias,
    get_lexical_richness,
)


class MyTestCase(unittest.TestCase):
    def test_Data_featurize(self):
        print("\n---- test_Data_featurize ---")

        a = [
            "I love this movie.",
            "apple is looking at buying U.K. startup for $1 billion.",
        ]
        A = TextData(a)

        B = A.apply(get_gender_bias)
        for b in B:
            print(b)

        B = A.apply(get_lexical_richness)
        for b in B:
            print(b)

        B = A.apply(get_basic_words)
        for b in B:
            print(b)


if __name__ == "__main__":
    unittest.main()
