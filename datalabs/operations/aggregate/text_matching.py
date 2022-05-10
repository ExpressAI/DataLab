from typing import Dict, List, Any, Optional
from .aggregating import Aggregating, aggregating
from typing import Callable, Mapping, Iterator
import numpy as np
from tqdm import tqdm
import os
import sys
import sacrebleu
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




def get_similarity_by_sacrebleu(text1, text2):
    # pip install sacrebleu
    references = [text1]
    hypothesis = text2
    score = sacrebleu.sentence_bleu(hypothesis, references).score

    return score



@text_matching_aggregating(name = "get_statistics", contributor= "datalab",
                                 task="text-matching, natural-language-inference",
                           description="Calculate the overall statistics (e.g., average length) of a given "
                                       "text pair classification datasets. e,g. natural language inference")
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
dataset = load_dataset('sick')
res = dataset['test'].apply(get_statistics)
print(next(res))

    """
    # for hate speech
    # from hatesonar import Sonar
    # sonar = Sonar()

    sample_infos = []

    text1_lengths = []
    text2_lengths = []
    labels_to_number = {}
    vocab = {}
    number_of_tokens = 0
    gender_results = []
    # hatespeech = {
    #                  "hate_speech":{"ratio":0,"texts":[]},
    #                     "offensive_language":{"ratio":0,"texts":[]},
    #                     "neither":{"ratio":0,"texts":[]}}
    text1_divided_text2 = []
    similarities = []

    for sample in tqdm(samples):

        text1, text2, label = sample["text1"], sample["text2"], sample["label"]


        similarity_of_text_pair = get_similarity_by_sacrebleu(text1, text2)
        similarities.append(similarity_of_text_pair)

        # average length of text1
        text1_length = len(text1.split(" "))
        text1_lengths.append(text1_length)

        # average length of text2
        text2_length = len(text2.split(" "))
        text2_lengths.append(text2_length)

        # text1/text2
        text1_divided_text2.append(len(text1.split(" "))/len(text2.split(" ")))


        # label info
        if label in labels_to_number.keys():
            labels_to_number[label] += 1
        else:
            labels_to_number[label] = 1


        # update the number of tokens
        number_of_tokens += len(text1.split())
        number_of_tokens += len(text2.split())

        # Vocabulary info
        for w in (text1 + text2).split(" "):

            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

        # Gender info
        gender_result1 = get_gender_bias.func(text1)
        gender_result2 = get_gender_bias.func(text2)
        gender_results.append(gender_result1["gender_bias_info"])
        gender_results.append(gender_result2["gender_bias_info"])


        # hataspeech
        # results = sonar.ping(text=text1)
        # class_1 = results['top_class']
        # confidence = 0
        # for value in results['classes']:
        #     if value['class_name'] == class_1:
        #         confidence = value['confidence']
        #         break
        #
        # hatespeech[class_1]["ratio"] += 1
        # if class_1 != "neither":
        #     hatespeech[class_1]["texts"].append(text1)


        # results = sonar.ping(text=text2)
        # class_2 = results['top_class']
        # confidence = 0
        # for value in results['classes']:
        #     if value['class_name'] == class_2:
        #         confidence = value['confidence']
        #         break
        #
        # hatespeech[class_2]["ratio"] += 1
        # if class_2 != "neither":
        #     hatespeech[class_2]["texts"].append(text2)

        sample_info = {
            "text1":text1,
            "text2": text2,
            "label":label,
            "text1_length": text1_length,
            "text2_length": text2_length,
            "text1_gender":gender_result1,
            "text2_gender":gender_result2,
            # "text1_hate_speech_class":class_1,
            # "text2_hate_speech_class":class_2,
            "text1_divided_text2":len(text1.split(" "))/len(text2.split(" ")),
            "similarity_of_text_pair":similarity_of_text_pair,
        }
        if len(sample_infos) < 10000:
            sample_infos.append(sample_info)
    # ------------------ Dataset-level ----------------
    # get vocabulary
    vocab_sorted = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))


    # compute dataset-level gender_ratio
    gender_ratio = {"word":
                        {"male": 0, "female": 0},
                    "single_name":
                        {"male": 0, "female": 0},
                    }
    for result in gender_results:
        res_word = result['word']
        gender_ratio['word']['male'] += result['word']['male']
        gender_ratio['word']['female'] += result['word']['female']
        gender_ratio['single_name']['male'] += result['single_name']['male']
        gender_ratio['single_name']['female'] += result['single_name']['female']

    n_gender = (gender_ratio['word']['male'] + gender_ratio['word']['female'])
    if n_gender != 0:
        gender_ratio['word']['male'] /= n_gender
        gender_ratio['word']['female'] /= n_gender
    else:
        gender_ratio['word']['male'] = 0
        gender_ratio['word']['female'] = 0


    n_gender = (gender_ratio['single_name']['male'] + gender_ratio['single_name']['female'])
    if n_gender != 0:
        gender_ratio['single_name']['male'] /= n_gender
        gender_ratio['single_name']['female'] /= n_gender
    else:
        gender_ratio['single_name']['male'] = 0
        gender_ratio['single_name']['female'] = 0

    # get ratio of hate_speech:offensive_language:neither
    # for k,v in hatespeech.items():
    #     hatespeech[k]["ratio"] /= 2* len(samples)


    res = {
            "dataset-level":{
                "length_info": {
                    "max_text1_length": np.max(text1_lengths),
                    "min_text1_length": np.min(text1_lengths),
                    "average_text1_length": np.average(text1_lengths),
                    "max_text2_length": np.max(text2_lengths),
                    "min_text2_length": np.min(text2_lengths),
                    "average_text2_length": np.average(text2_lengths),
                    "text1_divided_text2":np.average(text1_divided_text2),
                },
                "label_info": {
                    "ratio": min(labels_to_number.values()) * 1.0 / max(labels_to_number.values()),
                    "distribution": labels_to_number,
                },
                "vocabulary_info":vocab_sorted,
                "number_of_samples": len(samples),
                "number_of_tokens": number_of_tokens,
                "gender_info": gender_ratio,
                "average_similarity": np.average(similarities),
                # "hatespeech_info": hatespeech,
            },
        "sample-level": sample_infos
    }

    return res