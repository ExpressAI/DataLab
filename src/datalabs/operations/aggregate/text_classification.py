from typing import Dict, List, Any, Optional
from .aggregating import Aggregating, aggregating
from typing import Callable, Mapping, Iterator
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from operation import DatasetOperation, dataset_operation
from featurize.general import *
from data import TextData

class TextClassificationAggregating(Aggregating, DatasetOperation):


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
        self._type = 'TextClassificationAggregating'
        self.processed_fields = ["text"]
        if isinstance(processed_fields,str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"




class text_classification_aggregating(aggregating, dataset_operation):
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
            tf_class = TextClassificationAggregating(name = self.name.__name__, func=self.name)
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = TextClassificationAggregating(name=name, func = f,
                                   resources = self.resources,
                                   contributor = self.contributor,
                                    processed_fields = self.processed_fields,
                                    generated_field = self.generated_field,
                                    task = self.task,
                                    description=self.description,)
            return tf_cls


@text_classification_aggregating(name="get_features_dataset_level", contributor="datalab",
             task="text-classification",
             description="Get the average length of a list of texts")
def get_features_dataset_level(samples:Iterator):
    """
    Package: python
    Input:
        texts: Iterator
    Output:
        int
    """

    res_info = {}
    for sample in samples:
        for feature_name, value in sample.items():
            if feature_name == "label":
                continue
            if isinstance(value, int) or isinstance(value, float):
                if feature_name not in res_info.keys():
                    res_info[feature_name] = value
                else:
                    res_info[feature_name] += value


    for feature_name, value in res_info.items():
        res_info[feature_name] /= len(samples)

    return res_info



@text_classification_aggregating(name = "get_label_distribution", contributor= "datalab", processed_fields= "text",
                                 task="text-classification", description="Calculate the label distribution of a given text classification dataset")
def get_label_distribution(samples:Iterator):
    """
    Input:
    samples: [{
     "text":
     "label":
    }]
    Output:
        dict:
        "label":n_samples
    """
    labels_to_number = {}
    for sample in samples:
        text, label = sample["text"], sample["label"]




        if label in labels_to_number.keys():
            labels_to_number[label] += 1
        else:
            labels_to_number[label] = 1

    res = {
        "imbalance_ratio": min(labels_to_number.values())*1.0/max(labels_to_number.values()),
        "label_distribution":labels_to_number
    }

    return res


@text_classification_aggregating(name="get_statistics", contributor="datalab",
                                 task="text-classification",
                                 description="Calculate the overall statistics (e.g., average length) of a given text classification dataset")
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "label":
    }]
    Output:
        dict:
        "label":n_samples

    usage:
    you can test it with following code:

from datalabs import load_dataset
from aggregate.text_classification import *
dataset = load_dataset('./datasets/mr')
res = dataset['test'].apply(get_statistics)
print(res._stat)


    """
    # Grammar checker
    # from spellchecker import SpellChecker
    # spell = SpellChecker()
    #spell = SpellChecker(distance=1)  # set at initialization

    scriptpath = os.path.dirname(__file__)
    with open(os.path.join(scriptpath, '../edit/resources/spell_corrections.json'), 'r') as file:
        COMMON_MISSPELLINGS_DICT = json.loads(file.read())

    # print(COMMON_MISSPELLINGS_DICT)
    # exit()







    # for hate speech
    # from hatesonar import Sonar
    # sonar = Sonar()




    sample_infos = []

    labels_to_number = {}
    lengths = []
    gender_results = []
    vocab = {}
    number_of_tokens = 0
    hatespeech = {
                     "hate_speech":{"ratio":0,"texts":[]},
                        "offensive_language":{"ratio":0,"texts":[]},
                        "neither":{"ratio":0,"texts":[]}}
    spelling_errors = []

    for sample in tqdm(samples):
        text, label = sample["text"], sample["label"]



        # grammar checker
        for word in text.split(" "):
            #word_corrected = spell.correction(word)
            if word.lower() in COMMON_MISSPELLINGS_DICT.keys():
                spelling_errors.append((word, COMMON_MISSPELLINGS_DICT[word.lower()]))


        # hataspeech
        # results = sonar.ping(text=text)
        # class_ = results['top_class']
        # confidence = 0
        # for value in results['classes']:
        #     if value['class_name'] == class_:
        #         confidence = value['confidence']
        #         break
        #
        # hatespeech[class_]["ratio"] += 1
        # if class_ != "neither":
        #     hatespeech[class_]["texts"].append(text)



        # update the number of tokens
        number_of_tokens += len(text.split())

        # update vocabulary
        for w in text.split(" "):

            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1



        # gender bias
        """
        result = {
        'word': {
            'male': one_words_results['words_m'],
            'female': one_words_results['words_f']
        },
        'single_name': {
            'male': one_words_results['single_name_m'],
            'female': one_words_results['single_name_f']
        },
        }
        """
        gender_result = get_gender_bias.func(text)
        gender_results.append(gender_result["gender_bias_info"])


        # average length
        text_length = len(text.split(" "))
        lengths.append(text_length)

        # label imbalance
        if label in labels_to_number.keys():
            labels_to_number[label] += 1
        else:
            labels_to_number[label] = 1


        sample_info = {
            "text":text,
            "label":label,
            "text_length": text_length,
            "gender":gender_result,
            # "hate_speech_class":class_,
        }

        if len(sample_infos) < 10000:
            sample_infos.append(sample_info)

    # -------------------------- dataset-level ---------------------------
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



    # get vocabulary
    vocab_sorted = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

    # get ratio of hate_speech:offensive_language:neither
    # for k,v in hatespeech.items():
    #     hatespeech[k]["ratio"] /= len(samples)

    #print(hatespeech)
    res = {
            "dataset-level":{
                "length_info": {
                    "max_text_length": np.max(lengths),
                    "min_text_length": np.min(lengths),
                    "average_text_length": np.average(lengths),
                },
                "label_info": {
                    "ratio":min(labels_to_number.values()) * 1.0 / max(labels_to_number.values()),
                    "distribution": labels_to_number,
                },
                "gender_info":gender_ratio,
                "vocabulary_info":vocab_sorted,
                "number_of_samples":len(samples),
                "number_of_tokens":number_of_tokens,
                # "hatespeech_info":hatespeech,
                "spelling_errors":len(spelling_errors),
            },
        "sample-level":sample_infos
    }

    return res