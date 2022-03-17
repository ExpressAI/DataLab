from typing import Dict, List, Any, Optional
from .featurizing import Featurizing, featurizing
from typing import Callable, Mapping
# from ..operation import DatasetOperation, dataset_operation

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from .general import get_features_sample_level as get_features_sample_level_general

class TextMatchingFeaturizing(Featurizing, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text"],
                 generated_field: str = None,
                 task = "text-matching",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources = resources, contributor = contributor,
                         task = task,description=description)
        self._type = 'TextMatchingFeaturizing'
        self.processed_fields = ["text"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"




class text_matching_featurizing(featurizing, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text"],
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
            tf_class = TextMatchingFeaturizing(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = TextMatchingFeaturizing(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,)
            return tf_cls




def get_schema_of_sample_level_features():
    return {
            "text1_length":1,
            "text1_lexical_richness":0.2,
            "text1_basic_words":0.2,
            "text1_gender_bias_word_male":1,
            "text1_gender_bias_word_female":2,
            "text1_gender_bias_single_name_male":1,
            "text1_gender_bias_single_name_female":1,
            "text2_length": 1,
            "text2_lexical_richness": 0.2,
            "text2_basic_words": 0.2,
            "text2_gender_bias_word_male": 1,
            "text2_gender_bias_word_female": 2,
            "text2_gender_bias_single_name_male": 1,
            "text2_gender_bias_single_name_female": 1,
            "text1_minus_text2":0.0,
            }



@text_matching_featurizing(name = "get_features_sample_level", contributor= "datalab", processed_fields= "text",
                                 task="text-matching", description="sample-level features")
def get_features_sample_level(sample:dict):


    text1 = sample["text1"]
    text2 = sample["text2"]


    res_info_general_new = {}
    res_info_general = get_features_sample_level_general.func(text1)
    for k,v in res_info_general.items():
        res_info_general_new["text1" + "_" + k] =v

    res_info_general = get_features_sample_level_general.func(text2)
    for k,v in res_info_general.items():
        res_info_general_new["text2" + "_" + k] =v

    # get task-dependent features
    summary_features = {
        "text1_minus_text2":len(text1.split(" ")) - len(text2.split(" ")),
    }


    # update the res_info_general_new
    res_info_general_new.update(summary_features)

    # res_info_general_new.update({"answer_length":answer_length,
    #                          "option1_length":option1_length,
    #                          "option2_length":option2_length,
    #                           # "option_index":int(option_index),
    #                              })

    return res_info_general_new