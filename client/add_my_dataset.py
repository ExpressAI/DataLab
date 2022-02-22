import sys
from typing import List, Dict
from client import Client




# add samsum datasets
# client = Client(dataset_name_db="test_samsum2", dataset_name_sdk="samsum", calculate_features = True, data_typology = 'textdataset')
# client.add_dataset_from_sdk()



# client = Client(dataset_name_db="test_govreport", dataset_name_sdk="govreport", calculate_features = True, data_typology = 'textdataset')
# client.add_dataset_from_sdk()

from datalabs import load_dataset

#dataset = load_dataset("qmsum", "document",feature_expanding=True)
#dataset = load_dataset("dialogsum", "document",feature_expanding=True)
#dataset = load_dataset("multinews", "raw-single",feature_expanding=True)
#dataset = load_dataset("multi_xscience", "single-document",feature_expanding=True)
#dataset = load_dataset("bigpatent", feature_expanding=True)
dataset = load_dataset("reddit_tifu", feature_expanding=True)

print(dataset)



