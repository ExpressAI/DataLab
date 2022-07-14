from utils_funcs import basic_features, get_summary_features

import datalabs  # # noqa
from datalabs.operations.featurize import nlp_featurizing

"""
Note:
(1) Here are just some example functions, feel free to custmoized them
based on your tasks and needs.
(2) For all functions, the input and output are dictionaries.  # noqa
Specifically, input is one sample:dict whose structure is consistent with
your definition in SDK. For example, in text classification task, one sample can be:
from datalabs import load_dataset
dataset = load_dataset("mr")
sample = dataset["train"][0] # {"text":"I love this movie", "label":1}
"""


@nlp_featurizing(name="text_classification_func")
def text_classification_func(sample: dict):
    """
    sample:{
        "text":str
        "label":int
    }
    """

    res_info_general = basic_features(sample["text_tokenized"])
    """
        res_info_general =
           {"length":length,
            "lexical_richness":lexical_richness,
            "basic_words":basic_words,
            "gender_bias_word_male":one_words_results['words_m'],
            "gender_bias_word_female":one_words_results['words_f'],
            "gender_bias_single_name_male":one_words_results['single_name_m'],
            "gender_bias_single_name_female":one_words_results['single_name_f'],
            # "hate_speech_detection":class_,
            }
    """

    res_info_general_all = {}
    for k, v in res_info_general.items():
        res_info_general_all["text" + "_" + k] = v

    return res_info_general_all


@nlp_featurizing(name="text_matching_func")
def text_matching_func(sample: dict):
    """
    sample:
    {
        text1:str
        text2:str
        label:int
    }
    """

    text1 = sample["text1_" + "tokenized"]
    text2 = sample["text2_" + "tokenized"]

    res_info_general_all = {}

    res_info_general = basic_features(text1)
    for k, v in res_info_general.items():
        res_info_general_all["text1" + "_" + k] = v

    res_info_general = basic_features(text2)
    for k, v in res_info_general.items():
        res_info_general_all["text2" + "_" + k] = v

    # get task-dependent features
    task_dependent_features = {
        "text1_minus_text2": len(text1.split(" ")) - len(text2.split(" ")),
    }

    # update features
    res_info_general_all.update(task_dependent_features)

    return res_info_general_all


"""
from datalabs import load_dataset
from example_funcs import summarization_func
dataset = load_dataset("govreport")
"""


@nlp_featurizing(name="summarization_func")
def summarization_func(sample: dict):

    text = sample["text"]
    summary = sample["summary"]

    res_info_general_all = {}

    res_info_general = basic_features(text)
    for k, v in res_info_general.items():
        res_info_general_all["text" + "_" + k] = v

    res_info_general = basic_features(summary)
    for k, v in res_info_general.items():
        res_info_general_all["summary" + "_" + k] = v

    # get task-dependent features
    task_dependent_features = get_summary_features(text, summary)

    # update the res_info_general_all
    res_info_general_all.update(task_dependent_features)

    return res_info_general_all


_symmetric_relations = [
    "/base/popstra/celebrity/breakup./base/popstra/breakup/participant",
    "/base/popstra/celebrity/canoodled./base/popstra/canoodled/participant",
    "/base/popstra/celebrity/dated./base/popstra/dated/participant",
    "/base/popstra/celebrity/friendship./base/popstra/friendship/participant",
    "/celebrities/celebrity/celebrity_friends./celebrities/friendship/friend",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity",  # noqa: E501
    "/influence/influence_node/peers./influence/peer_relationship/peers",
    "/location/location/adjoin_s./location/adjoining_relationship/adjoins",
    "/people/person/spouse_s./people/marriage/spouse",
    "/people/person/sibling_s./people/sibling relationship/sibling",
]


@nlp_featurizing(name="kg_func")
def kg_func(sample: dict):
    head = sample["head"]
    link = sample["link"]
    tail = sample["tail"]

    relation_symmetric = ""
    if link in _symmetric_relations:
        relation_symmetric = "symmetric"
    else:
        relation_symmetric = "asymmetric"

    res_info_general_all = {}

    res_info_general = basic_features(head)
    for k, v in res_info_general.items():
        res_info_general_all["head" + "_" + k] = v

    res_info_general = basic_features(tail)
    for k, v in res_info_general.items():
        res_info_general_all["tail" + "_" + k] = v

    # get task-dependent features
    task_dependent_features = {
        "relation_symmetric": relation_symmetric,
    }

    # update the res_info_general_all
    res_info_general_all.update(task_dependent_features)

    return {}


"""
you can test this operation by:

from datalabs import load_dataset
from example_funcs import qa_multiple_choice_func
dataset = load_dataset("fig_qa", "medium")
dataset_processed = dataset["test"].apply(qa_multiple_choice_func, mode = "memory")

"""


@nlp_featurizing(name="qa_multiple_choice_func")
def qa_multiple_choice_func(sample: dict):

    context = sample["context"]
    options = sample["options"]
    answer = sample["answers"]["text"]
    option_index = sample["answers"]["option_index"]  # noqa

    answer_length = len(answer.split(" "))
    option1_length = len(options[0].split(" "))
    option2_length = len(options[1].split(" "))

    res_info_general = basic_features(context)
    res_info_general_all = {}
    for k, v in res_info_general.items():
        res_info_general_all["context" + "_" + k] = v

    res_info_general_all.update(
        {
            "answer_length": answer_length,
            "option1_length": option1_length,
            "option2_length": option2_length,
            # "option_index":int(option_index),
        }
    )

    return res_info_general_all
