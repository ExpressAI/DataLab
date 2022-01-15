from typing import Dict, List, Any, Optional
from .aggregating import Aggregating, aggregating
from typing import Callable, Mapping, Iterator
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation


class SequenceLabelingAggregating(Aggregating, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["tokens","tags"],
                 generated_field: str = None,
                 task = "sequence-labeling",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources = resources, contributor = contributor,
                         task = task,description=description)
        self._type = 'SequenceLabelingAggregating'
        self.processed_fields = ["tokens","tags"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "SequenceLabelingDataset"




class sequence_labeling_aggregating(aggregating, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["tokens", "tags"],
                 generated_field:str = None,
                 task = "sequence-labeling",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor, description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = SequenceLabelingAggregating(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = SequenceLabelingAggregating(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,)
            return tf_cls





@sequence_labeling_aggregating(name="get_statistics", contributor="datalab", processed_fields=["tokens","tags"],
                                 task="sequence-labeling",
                                 description="this function is used to compute the overall statistics of the summarization")
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "tokens":
     "tags":
    }]
    Output:dict:

    usage:
    you can test it with following code:

    from datalabs import load_dataset
    from aggregate import *
    dataset = load_dataset('wnut_17')
    res = dataset['test'].apply(get_statistics)
    print(next(res))

    """

    text_lengths = []

    for sample in tqdm(samples):

        tokens, tags = sample["tokens"], sample["tags"]

        # average length of text
        text_length = len(tokens)
        text_lengths.append(text_length)




    res = {
            "average_text_length":np.average(text_lengths),
    }

    return res