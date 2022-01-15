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
from data import TextData

class SummarizationAggregating(Aggregating, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text","summary"],
                 generated_field: str = None,
                 task = "summarization",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources = resources, contributor = contributor,
                         task = task,description=description)
        self._type = 'SummarizationAggregating'
        self.processed_fields = ["text","summary"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "SummarizationDataset"




class summarization_aggregating(aggregating, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "summary"],
                 generated_field:str = None,
                 task = "summarization",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor, description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = SummarizationAggregating(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = SummarizationAggregating(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,)
            return tf_cls





@summarization_aggregating(name="get_statistics", contributor="datalab", processed_fields=["text","summary"],
                                 task="summarization",
                                 description="this function is used to compute the overall statistics of the summarization")
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "summary":
    }]
    Output:dict:

    usage:
    you can test it with following code:

    from datalabs import load_dataset
    from aggregate import *
    dataset = load_dataset('xsum')
    res = dataset['test'].apply(get_statistics)
    print(next(res))

    """

    summary_lengths = []
    text_lengths = []

    for sample in tqdm(samples):

        text, summary = sample["text"], sample["summary"]

        # average length of text
        text_length = len(text.split(" "))
        text_lengths.append(text_length)

        # average length of summary
        summary_length = len(summary.split(" "))
        summary_lengths.append(summary_length)




    res = {
            "average_text_length":np.average(text_lengths),
            "average_summary_length":np.average(summary_lengths),
    }

    return res