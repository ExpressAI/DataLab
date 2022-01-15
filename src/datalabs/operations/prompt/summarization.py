from typing import Dict, List, Any, Optional
from typing import Callable, Mapping

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from prompt.prompting import *

class SummarizationPrompting(Prompting, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "summary"],
                 generated_field: str = None,
                 task = "summarization",
                 description = None,
                 template = None,
                 ):
        super().__init__(name = name, func = func, resources = resources, contributor = contributor,
                         task = task,description=description)
        self._type = 'SummarizationPrompting'
        self.processed_fields = ["text", "summary"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"
        self.template = template

    def __call__(self, sample) -> Any: # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        return self.func(sample,  **self.resources)


class summarization_prompting(prompting, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text", "summary"],
                 generated_field:str = None,
                 task = "summarization",
                 description = None,
                 template = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor, description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task
        self.template = template


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = SummarizationPrompting(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = SummarizationPrompting(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,
                                    template = self.template,)
            return tf_cls



"""
 


Test Example:

from datalabs import load_dataset
dataset = load_dataset('xsum')
from prompt.summarization import *
res = dataset['test'].apply(template_p1)
print(next(res))



"""

@summarization_prompting(name = "template_p1", contributor= "datalab", processed_fields= ['text', 'summary'],
                               template = "TEXT: {text} QUERY: In around {len(title.split(\' \'))} words, write a TLDR (Too Long Didn''t Read) summary for the above text.",
                                 task="summarization", description="this function is used to calculate the text length")
def template_p1(sample:dict):

    tp1 = "TEXT: {text} QUERY: Write a TLDR (Too Long Didn\'\'t Read) summary for the above text."

    # prompting process
    text = sample["text"]
    summary = sample["summary"]

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))



    return {"text_prompt": text_prompt,
            "summary_prompt": summary}

