from client import Client

from example_funcs import text_classification_func # it depends on your task; you could also customized the function


client = Client(user_name="xy_admin", # the user name of your account: https://datalab.nlpedia.ai/user
                password="Xy008623",  # the password of your account: https://datalab.nlpedia.ai/user
                dataset_name_db="qc", # you can specify this based on your interest
                dataset_name_sdk="../datasets/qc", # this should be the name of your data loader script
                sub_dataset_name_sdk=None,
                feature_func = text_classification_func, # we provide some example functions (example_funcs.py) but you could customize them as well
                end_point_add_dataset = "http://chinese.datalab.nlpedia.ai/api/upload_new_dataset",
                )
client.add_dataset_from_sdk() # if you could successfully run this, you can find the dataset here: https://datalab.nlpedia.ai/datasets_explore/user_dataset





