import unittest

from datalabs import load_dataset
from datalabs.operations.preprocess.general import tokenize
from datalabs.operations.tokenizer import (
    get_default_tokenizer,
    get_tokenizer,
    tokenizer_registry,
)


class MyTestCase(unittest.TestCase):
    def test_tokenizer_registry(self):
        print(tokenizer_registry)

        my_tokenizer = get_default_tokenizer("text-classification", "zh")
        print(my_tokenizer)

        my_tokenizer2 = get_tokenizer("SingleSpaceTokenizer")
        print(my_tokenizer2)

        text_zh = "我喜欢这一部电影"
        print(my_tokenizer(text_zh))

        text_en = "I love this movie"
        print(my_tokenizer2(text_en))

    def test_tokenizer_operation(self):

        dataset = load_dataset("waimai")
        dataset_tokenizer = dataset["train"].apply(tokenize, mode="memory")

        print(dataset_tokenizer[0:10])
