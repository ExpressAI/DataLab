from typing import Dict, List, Any, Optional
from .aggregating import Aggregating, aggregating
from typing import Callable, Mapping, Iterator
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from featurize import *


class TextMatchingAggregating(Aggregating, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text1", "text2"],
                 generated_field: str = None,
                 task = "text-matching",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources = resources, contributor = contributor,
                         task = task,description=description)
        self._type = 'TextMatchingAggregating'
        self.processed_fields = ["text1", "text2"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"




class text_matching_aggregating(aggregating, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text1", "text2"],
                 generated_field:str = None,
                 task = "text-matching",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor, description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = TextMatchingAggregating(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = TextMatchingAggregating(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,)
            return tf_cls






@text_matching_aggregating(name = "get_label_distribution", contributor= "datalab", processed_fields= ["text1", "text2"],
                                 task="text-matching", description="this function is used to calculate the overall statistics")
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text1":
     "text2":
    }]
    Output:
        dict:

    usage:
    you can test it with following code:

    from datalabs import load_dataset
    from aggregate.text_matching import *
    dataset = load_dataset('snli')
    res = dataset['test'].apply(get_statistics)
    print(next(res))

    """
    text1_lengths = []
    text2_lengths = []

    for sample in tqdm(samples):

        text1, text2 = sample["text1"], sample["text2"]

        # average length of text1
        text1_length = len(text1.split(" "))
        text1_lengths.append(text1_length)

        # average length of text2
        text2_length = len(text2.split(" "))
        text2_lengths.append(text2_length)




    res = {
            "average_text1_length":np.average(text1_lengths),
            "average_text2_length":np.average(text2_lengths),
    }

    return res