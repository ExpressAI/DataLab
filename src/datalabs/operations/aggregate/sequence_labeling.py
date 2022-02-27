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
from collections import Counter

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





@sequence_labeling_aggregating(name="get_statistics", contributor="datalab",
                                 task="sequence-labeling, named-entity-recognition, structure-prediction",
                                 description="Calculate the overall statistics (e.g., average length) of a given sequence labeling datasets (e.g., named entity recognition)")
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

    scriptpath = os.path.dirname(__file__)
    with open(os.path.join(scriptpath, '../edit/resources/spell_corrections.json'), 'r') as file:
        COMMON_MISSPELLINGS_DICT = json.loads(file.read())

    # for hate speech
    # from hatesonar import Sonar
    # sonar = Sonar()

    sample_infos = []
    labels_to_number = {}
    gender_results = []
    vocab = {}
    number_of_tokens = 0
    # hatespeech = {
    #     "hate_speech": {"ratio": 0, "texts": []},
    #     "offensive_language": {"ratio": 0, "texts": []},
    #     "neither": {"ratio": 0, "texts": []}}
    spelling_errors = []

    lengths = []
    chunks = []
    tag_texts = []
    for sample in tqdm(samples):
        tokens, tags = sample["tokens"], sample["tags"]
        text = ' '.join(tokens)
        # grammar checker
        for word in tokens:
            # word_corrected = spell.correction(word)
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
        number_of_tokens += len(tokens)

        # update vocabulary
        for w in tokens:

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



        # convert tag-id to tag-text
        tag_ts = tag_id2text(tags)
        tag_texts += tag_ts
        chunk = get_chunks(tag_ts)
        if len(chunk)!=0:
            chunks.append(chunk)

        # label imbalance
        for label in tag_ts:
            if label in labels_to_number.keys():
                labels_to_number[label] += 1
            else:
                labels_to_number[label] = 1

        sample_info = {
            "tokens": text,
            "tags": tag_ts,
            "text_length": text_length,
            "gender": gender_result,
            # "hate_speech_class": class_,
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
    gender_ratio['word']['male'] /= n_gender
    gender_ratio['word']['female'] /= n_gender

    n_gender = (gender_ratio['single_name']['male'] + gender_ratio['single_name']['female'])
    gender_ratio['single_name']['male'] /= n_gender
    gender_ratio['single_name']['female'] /= n_gender

    # get vocabulary
    vocab_sorted = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

    # get ratio of hate_speech:offensive_language:neither
    # for k, v in hatespeech.items():
    #     hatespeech[k]["ratio"] /= len(samples)

    # other NER features...


    label_distribution = dict(Counter(tag_texts))
    avg_entityLen,avg_entity_nums_inSent,entity_lengths = get_avg_spanLen(chunks)
    entity_length_distribution = dict(Counter(entity_lengths))


    res = {
        "dataset-level": {
            "entity_info":{
                "avg_entity_length": avg_entityLen,
                "avg_entity_on_sentence": avg_entity_nums_inSent,
                "sentence_without_entity": len(samples) - len(chunks),
                "entity_length_distribution": entity_length_distribution,
            },
            "length_info": {
                "max_text_length": np.max(lengths),
                "min_text_length": np.min(lengths),
                "average_text_length": np.average(lengths),
            },
            "label_info": {
                "ratio": min(labels_to_number.values()) * 1.0 / max(labels_to_number.values()),
                "distribution": label_distribution, #labels_to_number,
            },
            "gender_info": gender_ratio,
            "vocabulary_info":vocab_sorted,
            "number_of_samples": len(samples),
            "number_of_tokens": number_of_tokens,
            # "hatespeech_info": hatespeech,
        },
        "sample-level": sample_infos
    }

    return res

def get_avg_spanLen(chunks):
    entity_lengths = []
    entity_nums =[]
    for chunk in chunks:
        entity_nums.append(len(chunk))
        for ck in chunk:
            tag, sid, eid = ck
            entity_lengths.append(eid-sid)

    avg_entityLen = np.mean(entity_lengths)
    avg_entity_nums_inSent = np.mean(entity_nums)
    return avg_entityLen,avg_entity_nums_inSent,entity_lengths

def tag_id2text(tags):
    tag2text_dic = {0: "O", 1: "B-corporation", 2: "I-corporation", 3: "B-creative_work",
                4: "I-creative_work", 5: "B-group", 6: "I-group", 7: "B-location",
                8: "I-location", 9: "B-person", 10: "I-person", 11: "B-product", 12: "I-product"}
    tag_ts = []
    for tid in tags:
        tag_ts.append(tag2text_dic[tid])
    return tag_ts


def get_chunks(seq):
    """
    tags:dic{'per':1,....}
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = 'O'
    # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_chunk_type(tok):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tok_split = tok.split('-')
    return tok_split[0], tok_split[-1]
