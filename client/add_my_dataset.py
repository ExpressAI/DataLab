from example_funcs import (  # it depends on your task; you could also customized the function
    text_classification_func,
)

from client import Client


client = Client(
    user_name="xxx",  # the user name of your account: https://datalab.nlpedia.ai/user
    password="yyy",  # the password of your account: https://datalab.nlpedia.ai/user
    dataset_name_db="qc6",  # you can specify this based on your interest
    dataset_name_sdk="../datasets/qc",  # this should be the name of your data loader script
    sub_dataset_name_sdk="default",
    feature_func=text_classification_func,  # we provide some example functions (example_funcs.py) but you could customize them as well
)
client.add_dataset_from_sdk()  # if you could successfully run this, you can find the dataset here: https://datalab.nlpedia.ai/datasets_explore/user_dataset
