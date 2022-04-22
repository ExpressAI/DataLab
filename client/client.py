import sys
import json
from collections import defaultdict
from typing import List, Dict
import requests
from utils_datalab import get_info
from utils_datalab import validate_generate_db_metadata
from utils_datalab import generate_db_metadata_from_sdk


class Client:
    def __init__(self,
                 user_name,
                 password,
                 role='user',
                 dataset_name_db=None,
                 dataset_name_sdk=None,
                 sub_dataset_name_sdk=None,
                 version="origin",
                 languages=['en'],
                 tasks=['text-classification'],
                 task_categories=['text-classification'],
                 split={"train": 0, "validation": 0, "test": 0},
                 transformation={'type': 'origin'},
                 calculate_features=False,
                 feature_func=None,
                 data_typology="textdataset",
                 end_point_add_dataset = "https://datalab.nlpedia.ai/api/upload_new_dataset",
                 ):
        self._end_point_add_dataset = end_point_add_dataset

        self.user_name = user_name
        self.password = password
        self.role = role

        self.dataset_name_sdk: str = dataset_name_sdk
        self.dataset_name_db: str = dataset_name_db
        self.sub_dataset_name_sdk = sub_dataset_name_sdk
        self.version: str = version
        self.languages: List[str] = languages
        self.tasks: List[str] = tasks
        self.task_categories: str = task_categories
        self.split: Dict = split
        self.transformation: Dict = transformation
        self.calculate_features: bool = calculate_features
        self.feature_func = feature_func
        self.data_typology = data_typology

        if dataset_name_db == None:
            raise ValueError(f"the dataset_name_db should not be none:{dataset_name_db}")

    def add_dataset_metadata(self):
        """
        This is a quick introduction of a new dataset into DB by simply adding several pieces of  metadata information
        without any detailed samples
        """

        metadata_db = validate_generate_db_metadata(dataset_name=self.dataset_name_db,
                                                    transformation=self.transformation,
                                                    version=self.version,
                                                    task_categories=self.task_categories,
                                                    tasks=self.tasks,
                                                    split=self.split,
                                                    languages=self.languages)

        samples = [
            {"split_name": "train",
             "features": {
             },
             }
        ]
        data_json = {
            'metadata': metadata_db,
            'samples': samples,
            'user_name': self.user_name,
            'password': self.password,
            'role': self.role

        }

        response = requests.post(self._end_point_add_dataset, json=data_json)
        if response.status_code != 200:
            raise ConnectionError(f"[Error on metric: {rjson['metric']}]\n[Error Message]: {rjson['message']}")

        print(response.status_code)

    def add_dataset_from_sdk(self):
        """
            This method of introducing new datasets assumes that we have finished the dataloader of the dataset to be added
            in the folder: https://github.com/ExpressAI/DataLab/tree/main/datasets
        """

        # get metadata and dataset information from sdk by passing the dataset name of the sdk
        metadata_sdk, metadata_features_sdk, dataset_sdk = get_info(self.dataset_name_sdk,
                                                                    self.sub_dataset_name_sdk,
                                                                    calculate_features=self.calculate_features,
                                                                    feature_func = self.feature_func)

        # reformat the metadata information for db
        metadata_db = generate_db_metadata_from_sdk(metadata=metadata_sdk,
                                                    features=metadata_features_sdk,
                                                    dataset_name_db=self.dataset_name_db,
                                                    transformation=self.transformation,
                                                    version=self.version,
                                                    languages=self.languages,
                                                    data_typology=self.data_typology, )

        # reformat the sample information for db
        MAX_NUMBER_OF_SAMPLES = 100000
        samples_db = []
        for split in dataset_sdk.keys():
            for idx, sample in enumerate(dataset_sdk[split]):
                if idx > MAX_NUMBER_OF_SAMPLES:
                    break
                samples_db.append({
                    'split_name': split,
                    'features': sample
                })

        # prepare the data to be uploaded
        data_json = {
            'metadata': metadata_db,
            'samples': samples_db,
            'user_name': self.user_name,
            'password': self.password,
            'role': self.role
        }

        response = requests.post(self._end_point_add_dataset, json=data_json)
        if response.status_code != 200:
            raise ConnectionError("connection error")

        print(response.status_code)
