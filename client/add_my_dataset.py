from client import Client
from example_funcs import text_classification_func

# the user name of your account: https://datalab.nlpedia.ai/user
client = Client(
    user_name="xx",
    password="yy",
    dataset_name_db="waimai",
    dataset_name_sdk="../datasets/waimai",
    sub_dataset_name_sdk="default",
    status="public",
    feature_func=text_classification_func,
    end_point_add_dataset="https://datalab.nlpedia.ai/api/upload_new_dataset",
)
client.add_dataset_from_sdk()
