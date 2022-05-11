from client import Client
import requests
import json

from example_funcs import text_classification_func # it depends on your task; you could also customized the function


client = Client(user_name="xx", # the user name of your account: https://datalab.nlpedia.ai/user
                password="yy",  # the password of your account: https://datalab.nlpedia.ai/user
                dataset_name_db="waimai", # you can specify this based on your interest
                dataset_name_sdk="../datasets/waimai", # this should be the name of your data loader script
                sub_dataset_name_sdk="default",
                status = "public",
                feature_func = text_classification_func, # we provide some example functions (example_funcs.py) but you could customize them as well
                end_point_add_dataset="https://datalab.nlpedia.ai/api/upload_new_dataset",
                )
client.add_dataset_from_sdk() # if you could successfully run this, you can find the dataset here: https://datalab.nlpedia.ai/datasets_explore/user_dataset

