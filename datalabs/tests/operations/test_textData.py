import unittest

from datalabs.operations.data import Data, TextData

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
    def test_Data(self):
        a = ["I love this movie", "do you love this movie"]
        A = Data(a)
        print(A.data)

        self.assertEqual(A.data, a)

    def test_TextData(self):
        a = ["I love this movie", "do you love this movie"]
        A = TextData(a)
        print(A.data)

        self.assertEqual(A.data, [{"text": text} for text in a])


if __name__ == "__main__":
    unittest.main()
