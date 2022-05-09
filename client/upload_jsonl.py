import requests
import json
import os
from datalabs import load_dataset
from datalabs.operations.preprocess.general import tokenize
import multiprocessing
from aggregate.text_classification import get_features_dataset_level as get_features_dataset_level_text_classification
from datalabs.utils.more_features import prefix_dict_key, get_features_dataset
from datalabs.tasks.text_classification import TextClassification
from dataclasses import asdict
from example_funcs import text_classification_func # it depends on your task; you could also customized the function


"""
pip install --upgrade pip
pip install datalabs
python -m nltk.downloader omw-1.4 # to support more feature calculation
"""
def get_info(split_name, path_dataset, language, task):

    dataset = load_dataset("json", data_files=path_dataset)
    dataset[split_name]._info.task_templates = [TextClassification(task)]
    dataset[split_name]._info.languages = [language]



    raw_features = asdict(dataset[split_name]._info)["features"]
    dataset[split_name] = dataset[split_name].apply(tokenize,
                                                    num_proc=multiprocessing.cpu_count(),
                                                    mode="memory")
    dataset[split_name] = dataset[split_name].apply(text_classification_func,
                                                    num_proc=multiprocessing.cpu_count(),
                                                    mode="memory")

    all_features = asdict(dataset[split_name]._info)["features"]

    # turn on advanced fields
    for feature_name, feature_info in all_features.items():
        # this is defined for the case when feature is `text_tokenized`
        if feature_name.find("tokenize") != -1 and \
                all_features[feature_name]["dtype"] == "string":
            feature_info["raw_feature"] = True
            feature_info["is_bucket"] = False
        elif feature_name not in raw_features.keys():
            feature_info["raw_feature"] = False
            feature_info["is_bucket"] = True

    # add sample-level features
    features_mongodb = {}
    features_mongodb.update(all_features)
    # calculate dataset-level features
    dataset[split_name] = dataset[split_name].apply(get_features_dataset_level_text_classification,
                                                    mode="memory",
                                                    prefix="avg")

    features_dataset = get_features_dataset(dataset[split_name]._stat)
    for attr, feat_info in features_dataset.items():
        feat_info = asdict(feat_info)
        value = dataset[split_name]._stat[attr]
        feat_info["value"] = value
        features_dataset[attr] = feat_info
    features_dataset_new = prefix_dict_key(features_dataset, prefix=split_name)

    # add dataset-level features
    features_mongodb.update(features_dataset_new)
    metadata = asdict(dataset[split_name]._info)
    metadata["features"] = features_mongodb

    return metadata, dataset




# ----------- Example -----------------
# directory_of_files: the path of user-uploaded data
directory_of_files = "./"

for file_name in sorted(os.listdir(directory_of_files)):
    print(file_name)
    if file_name not in ["train.json", "validation.json", "test.json"]:
        continue
    else:
        split_name = file_name.split(".json")[0]

        language = "en"
        task = "text-classification"
        metadata, dataset = get_info(split_name, directory_of_files+"/" + file_name ,language, task)

        # print(metadata["features"])
        # print(dataset)

