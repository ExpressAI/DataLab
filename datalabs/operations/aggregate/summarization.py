from typing import Dict, List, Any, Optional
from .aggregating import Aggregating, aggregating
from typing import Callable, Mapping, Iterator
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from featurize.summarization import get_all_features
from featurize import *

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





@summarization_aggregating(name="get_statistics", contributor="datalab",
                                 task="summarization",
                                 description="Calculate the overall statistics (e.g., density) of a given summarization dataset")
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
from aggregate.summarization import *
dataset = load_dataset('govreport')
res = dataset['test'].apply(get_statistics)
print(next(res))

    """

    # for hate speech
    # from hatesonar import Sonar
    # sonar = Sonar()


    sample_infos = []

    summary_lengths = []
    text_lengths = []
    attr_jsons = []
    number_of_tokens = 0
    gender_results = []
    vocab = {}
    hatespeech = {
                     "hate_speech":{"ratio":0,"texts":[]},
                        "offensive_language":{"ratio":0,"texts":[]},
                        "neither":{"ratio":0,"texts":[]}}

    for sample in tqdm(samples):

        text, summary = sample["text"], sample["summary"]

        # average length of text
        text_length = len(text.split(" "))
        text_lengths.append(text_length)

        # average length of summary
        summary_length = len(summary.split(" "))
        summary_lengths.append(summary_length)


        # update the number of tokens
        number_of_tokens += len(text.split())
        number_of_tokens += len(summary.split())


        # Others
        # attr_json = get_all_features(sample)
        # attr_jsons.append(attr_json)

        # Vocabulary info
        for w in (text + summary).split(" "):

            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

        # Gender info
        # gender_result_text = get_gender_bias.func(text)
        # gender_result_summary = get_gender_bias.func(summary)
        # gender_results.append(gender_result_text)
        # gender_results.append(gender_result_summary)



        # hataspeech
        # results = sonar.ping(text=text)
        # class_1 = results['top_class']
        # confidence = 0
        # for value in results['classes']:
        #     if value['class_name'] == class_1:
        #         confidence = value['confidence']
        #         break
        #
        # hatespeech[class_1]["ratio"] += 1
        # if class_1 != "neither":
        #     hatespeech[class_1]["texts"].append(text)
        #
        #
        # results = sonar.ping(text=summary)
        # class_2 = results['top_class']
        # confidence = 0
        # for value in results['classes']:
        #     if value['class_name'] == class_2:
        #         confidence = value['confidence']
        #         break
        #
        # hatespeech[class_2]["ratio"] += 1
        # if class_2 != "neither":
        #     hatespeech[class_2]["texts"].append(summary)



        # sample_info = {
        #     "text":text,
        #     "summary": summary,
        #     "text_length": text_length,
        #     "summary_length": summary_length,
        #     "gender_result_text":gender_result_text,
        #     "gender_result_summary":gender_result_summary,
        #     "text1_hate_speech_class":class_1,
        #     "text2_hate_speech_class":class_2,
        #     # "density":attr_json["density"],
        #     # "coverage": attr_json["coverage"],
        #     # "compression": attr_json["compression"],
        #     # "repetition": attr_json["repetition"],
        #     # "novelty": attr_json["novelty"],
        #     # "copy_len": attr_json["copy_len"],
        # }
        # if len(sample_infos) < 10000:
        #     sample_infos.append(sample_info)


    ## --------------------- Dataset-level -----------------------
    # get vocabulary
    vocab_sorted = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))


    # compute dataset-level gender_ratio
    gender_ratio = {"word":
                        {"male": 0, "female": 0},
                    "single_name":
                        {"male": 0, "female": 0},
                    }
    # for result in gender_results:
    #     res_word = result['word']
    #     gender_ratio['word']['male'] += result['word']['male']
    #     gender_ratio['word']['female'] += result['word']['female']
    #     gender_ratio['single_name']['male'] += result['single_name']['male']
    #     gender_ratio['single_name']['female'] += result['single_name']['female']
    #
    # n_gender = (gender_ratio['word']['male'] + gender_ratio['word']['female'])
    # if n_gender != 0:
    #     gender_ratio['word']['male'] /= n_gender
    #     gender_ratio['word']['female'] /= n_gender
    # else:
    #     gender_ratio['word']['male'] = 0
    #     gender_ratio['word']['female'] = 0
    #
    #
    # n_gender = (gender_ratio['single_name']['male'] + gender_ratio['single_name']['female'])
    # if n_gender != 0:
    #     gender_ratio['single_name']['male'] /= n_gender
    #     gender_ratio['single_name']['female'] /= n_gender
    # else:
    #     gender_ratio['single_name']['male'] = 0
    #     gender_ratio['single_name']['female'] = 0


    # get ratio of hate_speech:offensive_language:neither
    # for k,v in hatespeech.items():
    #     hatespeech[k]["ratio"] /= 2* len(samples)

    # attr_avg = {}
    # for attr_json in attr_jsons:
    #     for attr_name, val in attr_json.items():
    #         if attr_name not in attr_avg.keys():
    #             attr_avg[attr_name] = val
    #         else:
    #             attr_avg[attr_name] += val
    # for attr_name, val in attr_avg.items():
    #     attr_avg[attr_name] /= len(attr_jsons)






    res = {
        "dataset-level":{
                "average_text_length":np.average(text_lengths),
                "average_summary_length":np.average(summary_lengths),
                "length_info": {
                    "max_text_length": np.max(text_lengths),
                    "min_text_length": np.min(text_lengths),
                    "average_text_length": np.average(text_lengths),
                    "max_summary_length": np.max(summary_lengths),
                    "min_summary_length": np.min(summary_lengths),
                    "average_summary_length": np.average(summary_lengths),
                },
                "number_of_samples": len(samples),
                "number_of_tokens": number_of_tokens,
                # "vocabulary_info": vocab_sorted,
                # "gender_info": gender_ratio,
                # "hatespeech_info": hatespeech,
                # **attr_avg,
        },
        # "sample-level": sample_infos,
    }

    return res