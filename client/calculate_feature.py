from datalabs import load_dataset
from featurize.general import get_features_sample_level
from aggregate.general import get_features_dataset_level
from dataclasses import asdict, dataclass, field
import json
from datalabs.utils.more_features import prefix_dict_key




def get_info(dataset_name:str, field = "text"):
    """
    Input:
    dataset_name: the dataset name of dataloader script, for example, mr
    field: the field to be featurized
    Output:
    asdict(dataset['train']._info): metadata information
    features_mongodb: features of metadata information
    dataset: detailed sample of all dataset splits
    """


    # load dataset
    dataset = load_dataset(dataset_name)

    # Feature
    all_splits = dataset['train']._info.splits.keys()


    features_mongodb = {}
    for split_name in all_splits:

        # add sample-level features
        features_mongodb.update(asdict(dataset[split_name]._info)["features"])

        # add dataset-level features
        dataset[split_name] = dataset[split_name].apply(get_features_dataset_level, mode="memory", prefix=field)
        features_dataset = asdict(dataset[split_name]._info)["features_dataset"]


        for attr, feat_info in features_dataset.items():
            value = dataset[split_name]._stat[attr]
            feat_info["value"] = value
            features_dataset[attr] = feat_info

        features_dataset_new = prefix_dict_key(features_dataset, prefix=split_name)

        features_mongodb.update(features_dataset_new)



    return asdict(dataset['train']._info), features_mongodb, dataset




dataset_name = "../datasets/mr"
metadata, metadata_features, dataset = get_info(dataset_name, field="text")


print(metadata)
print(json.dumps(metadata_features, indent=4))
print(dataset)
