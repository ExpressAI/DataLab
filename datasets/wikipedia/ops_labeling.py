import json
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Mapping, Optional

from lxml import etree

from datalabs.operations.featurize.featurizing import Featurizing, featurizing


class WikipediaLabeling(Featurizing):
    def __init__(self, *args, **kwargs):

        super(WikipediaFeaturizing, self).__init__(*args, **kwargs)
        self._type = "WikipediaLabeling"
        self._data_type = "wikipedia"


class wikipedia_labeling(featurizing):
    def __init__(self, *args, **kwargs):

        super(wikipedia_labeling, self).__init__(*args, **kwargs)
        # print(self.__dict__)

    def __call__(self, *param_arg):
        if callable(self.name):

            tf_class = WikipediaLabeling(name=self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]

            name = self.name or f.__name__
            tf_cls = WikipediaLabeling(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                task=self.task,
                description=self.description,
                processed_fields=self.processed_fields,
            )
            return tf_cls


"""
from datalabs import load_dataset
data = load_dataset('./wikipedia')
from wikipedia.featurize import *
 
res = data.apply(get_number_of_tokens)
print(next(res))
"""


@wikipedia_featurizing(
    name="get_number_of_tokens1",
    contributor="datalab",
    processed_fields="text",
    task="text-classification",
    description="this function is used to calculate the text length",
)
def get_number_of_tokens1(sample: dict):
    return len(sample["text"])
