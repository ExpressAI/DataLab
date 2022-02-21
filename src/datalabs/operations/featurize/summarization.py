from typing import Dict, List, Any, Optional
from .featurizing import Featurizing, featurizing
# from ..operation import DatasetOperation, dataset_operation
from typing import Callable, Mapping


from .plugins.summarization.sum_attribute import *
from .plugins.summarization.extractive_methods import _ext_oracle
from .plugins.summarization.extractive_methods import _lead_k
from .plugins.summarization.extractive_methods import _compute_rouge
# store all featurizing class

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from .general import get_features_sample_level as get_features_sample_level_general


class SummarizationFeaturizing(Featurizing, DatasetOperation):


    def __init__(self,
                 name:str = None,
                 func:Callable[...,Any] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields: List = ["text"],
                 generated_field: str = None,
                 task = "summarization",
                 description = None,
                 ):
        super().__init__(name = name, func = func, resources = resources,
                         contributor = contributor,
                         description= description)
        self._type = 'SummarizationFeaturizing'
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"
        self.task = task


class summarization_featurizing(featurizing, dataset_operation):
    def __init__(self,
                 name: Optional[str] = None,
                 resources: Optional[Mapping[str, Any]] = None,
                 contributor: str = None,
                 processed_fields:List = None,
                 generated_field:str = None,
                 task = "summarization",
                 description = None,
                 ):
        super().__init__(name = name, resources = resources,
                         contributor = contributor, task = task,
                         description=description)
        self.processed_fields = processed_fields
        self.generated_field = generated_field


    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = SummarizationFeaturizing(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = SummarizationFeaturizing(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task, description=self.description)
            return tf_cls



@summarization_featurizing(name = "get_density", contributor="datalab",
                           task = "summarization", description="This function measures to what extent a summary covers the content in the source text.")
def get_density(sample:dict):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(sample['text'], sample['summary'])
    return {"density":attribute_info["attr_density"]}

@summarization_featurizing(name = "get_coverage", contributor="datalab",
                           task = "summarization", description="This function measures to what extent a summary covers the content in the source text.")
def get_coverage(sample:dict):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(sample['text'], sample['summary'])
    return {"coverage":attribute_info["attr_coverage"]}

@summarization_featurizing(name = "get_compression", contributor="datalab",
                           task = "summarization", description="This function measures the compression ratio from the source text to the generated summary.")
def get_compression(sample:dict):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(sample['text'], sample['summary'])
    return {"compression":attribute_info["attr_compression"]}


@summarization_featurizing(name = "get_repetition", contributor="datalab",
                           task = "summarization", description="This function measures the rate of repeated segments in summaries. The segments are instantiated as trigrams.")
def get_repetition(sample:dict):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(sample['text'], sample['summary'])
    return {"repetition":attribute_info["attr_repetition"]}


@summarization_featurizing(name = "get_novelty", contributor="datalab",
                           task = "summarization", description="This measures the proportion of segments in the summaries that haven’t appeared in source documents. The segments are instantiated as bigrams.")
def get_novelty(sample:dict):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(sample['text'], sample['summary'])
    return {"novelty":attribute_info["attr_novelty"]}


@summarization_featurizing(name = "get_copy_len", contributor="datalab",
                           task = "summarization", description="Measures the average length of segments in summary copied from source document.")
def get_copy_len(sample:dict):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(sample['text'], sample['summary'])
    return {"copy_len":attribute_info["attr_copy_len"]}


@summarization_featurizing(name = "get_all_features", contributor="datalab",
                           task = "summarization", description="Calculate all features for summarization datasets (density, coverage, compression, repetition, novelty, copy lenght)")
def get_all_features(sample:dict):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(sample['text'], sample['summary'])
    return {
        "density":attribute_info["attr_density"],
        "coverage":attribute_info["attr_coverage"],
        "compression": attribute_info["attr_compression"],
        "repetition": attribute_info["attr_repetition"],
        "novelty": attribute_info["attr_novelty"],
        "copy_len": attribute_info["attr_copy_len"],
    }


@summarization_featurizing(name = "get_oracle_summary", contributor="datalab",
                           task = "summarization", description="This function extract the oracle summaries for text summarization")
def get_oracle_summary(sample:dict) -> Dict:
    """
    Input:
        SummarizationDataset: dict
    Output:
        return {"source":src,
            "reference":ref,
            "oracle_summary":oracle,
            "oracle_labels":labels,
            "oracle_score":max_score}
    """
    document = sent_tokenize(sample["text"]) # List
    summary = sample['summary']
    oracle_info = _ext_oracle(document, summary, _compute_rouge, max_sent=3)
    return oracle_info
#
#
#
@summarization_featurizing(name = "get_lead_k_summary", contributor="datalab",
                           task = "summarization", description="This function extract the lead k summary for text summarization datasets")
def get_lead_k_summary(sample:dict) -> Dict:
    """
    Input:
        SummarizationDataset: dict
    Output:
        return {"source":src,
                "reference":ref,
                "lead_k_summary":src,
                "lead_k_score":score}
    """
    document = sent_tokenize(sample["text"]) # List
    summary = sample['summary']
    lead_k_info = _lead_k(document, summary, _compute_rouge, k = 3)
    return lead_k_info



def get_schema_of_sample_level_features():
    return {
            "text_length":1,
            "text_lexical_richness":0.2,
            "text_basic_words":0.2,
            "text_gender_bias_word_male":1,
            "text_gender_bias_word_female":2,
            "text_gender_bias_single_name_male":1,
            "text_gender_bias_single_name_female":1,
            "summary_length": 1,
            "summary_lexical_richness": 0.2,
            "summary_basic_words": 0.2,
            "summary_gender_bias_word_male": 1,
            "summary_gender_bias_word_female": 2,
            "summary_gender_bias_single_name_male": 1,
            "summary_gender_bias_single_name_female": 1,
            "density": 0.1,
            "coverage": 0.1,
            "compression": 0.1,
            "repetition": 0.1,
            "novelty": 0.1,
            "copy_len": 0.1,
            }



@summarization_featurizing(name = "get_features_sample_level", contributor= "datalab", processed_fields= "text",
                                 task="summarization", description="This function is used to calculate the text length")
def get_features_sample_level(sample:dict):


    text = sample["text"]
    summary = sample["summary"]



    res_info_general = get_features_sample_level_general.func(text)
    res_info_general_new = {}
    for k,v in res_info_general.items():
        res_info_general_new["text" + "_" + k] =v

    res_info_general = get_features_sample_level_general.func(summary)
    for k,v in res_info_general.items():
        res_info_general_new["summary" + "_" + k] =v

    # get task-dependent features
    summary_features = get_all_features.func(sample)


    # update the res_info_general_new
    res_info_general_new.update(summary_features)

    # res_info_general_new.update({"answer_length":answer_length,
    #                          "option1_length":option1_length,
    #                          "option2_length":option2_length,
    #                           # "option_index":int(option_index),
    #                              })

    return res_info_general_new