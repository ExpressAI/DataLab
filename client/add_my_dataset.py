from client import Client
from example_funcs import text_classification_func  # it depends on your task; you could also customized the function

# This is for adding datasets through scripts
client = Client(user_name="XXX",  # the user name of your account: https://datalab.nlpedia.ai/user
                password="YYY",  # the password of your account: https://datalab.nlpedia.ai/user
                dataset_name_db="qc",  # you can specify this based on your interest
                dataset_name_sdk="qc",  # this should be the name of your data loader script
                sub_dataset_name_sdk="default",
                feature_func=text_classification_func,
                # we provide some example functions (example_funcs.py) but you could customize them as well
                )
client.add_dataset_from_sdk()  # if you could successfully run this, you can find the dataset here: https://datalab.nlpedia.ai/datasets_explore/user_dataset

# This is for adding datasets through files
client = Client(user_name="XXX",
                password="YYY",
                dataset_name_db="sst",  # you can specify this based on your interest
                dataset_name_sdk={"train": "files/train.json", "validation": "files/dev.json",
                                  "metadata": "files/metadata.json"}, # The metadata.json is required
                sub_dataset_name_sdk="default",
                feature_func=text_classification_func,
                )
client.add_dataset_from_sdk()
