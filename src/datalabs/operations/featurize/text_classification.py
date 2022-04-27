from typing import Dict, List, Any, Optional
from .featurizing import Featurizing, featurizing
from typing import Callable, Mapping
# from ..operation import DatasetOperation, dataset_operation

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from .general import get_features_sample_level as get_features_sample_level_general


class TextClassificationFeaturizing(Featurizing, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text"],
                 generated_field: str = None,
                 task = "text-classification",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources = resources, contributor = contributor,
                         task = task,description=description)
        self._type = 'TextClassificationFeaturizing'
        self.processed_fields = ["text"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"




class text_classification_featurizing(featurizing, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text"],
                 generated_field:str = None,
                 task = "text-classification",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources, contributor = contributor, description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = TextClassificationFeaturizing(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = TextClassificationFeaturizing(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,)
            return tf_cls



@text_classification_featurizing(name = "get_text_length", contributor= "datalab", processed_fields= "text",
                                 task="text-classification", description="This function is used to calculate the text length")
def get_text_length(sample:dict):
    return {"length":len(sample['text'].split(" "))}





@text_classification_featurizing(name = "get_features_sample_level", contributor= "datalab", processed_fields= "text",
                                 task="text-matching", description="sample-level features")
def get_features_sample_level(sample:dict):


    text = sample["text"]



    res_info_general_new = {}
    res_info_general = get_features_sample_level_general.func(text)
    for k,v in res_info_general.items():
        res_info_general_new["text" + "_" + k] =v


    # update the res_info_general_new
    res_info_general_new.update(res_info_general)

    # res_info_general_new.update({"answer_length":answer_length,
    #                          "option1_length":option1_length,
    #                          "option2_length":option2_length,
    #                           # "option_index":int(option_index),
    #                              })

    return res_info_general_new