# Add New Datasets into the DataLab Web Platform

DataLab provides an API which users can use to upload their own datasets into the DataLab web platform (privately or
publicly) so that:

* more deep analysis (bias, artifacts) could be made on the dataset in an interactive way
* you can compare your dataset with existing similar ones to understand their own characteristics
* more researcher will know your datasets

This involves two steps:

#### (1) Write a data loader script for your new dataset. You can refer to [`how to add new datasets into sdk`](https://github.com/ExpressAI/DataLab/blob/main/docs/add_new_datasets.md)

#### (2) Create an account at DataLab [web platform](https://datalab.nlpedia.ai/user)

#### (3) Call the client function to add your dataset

```python
 # suppose that python script is located at DataLab/client/
from client import Client

"""
this depends on your task; you could also customized the function by modifying 
the script DataLab/client/example_funcs.py
"""
from example_funcs import text_classification_func 


client = Client(
                user_name="xxx", # the user name of your account: https://datalab.nlpedia.ai/user
                password="yyy",  # the password of your account: https://datalab.nlpedia.ai/user
                dataset_name_db="zzz", # you can specify any name based on your preference
                dataset_name_sdk="../datasets/ttt", # this should be the name of your data loader script, 
                sub_dataset_name_sdk="default",
                feature_func = text_classification_func, # we provide some example functions (example_funcs.py) but you could customize them as well
)
client.add_dataset_from_sdk() # if you could successfully run this, you can find the dataset here: https://datalab.nlpedia.ai/datasets_explore/user_dataset
```

where

* `dataset_name_db`: denotes the name of the dataset to be stored in the database. You can specify any name based on your preference
* `dataset_name_sdk`: denotes your dataset's data loader script path.
* `feature_func`: is used to calculate sample-level (e.g., text length) and dataset-level features (e.g., the average of text length), which are usually
 task-dependent.
  * DataLab provide [some templates](https://github.com/ExpressAI/DataLab/blob/main/client/example_funcs.py) for four types of task. You can either use them directly or develop new ones based on them.
  * Usually, the process of feature calculation will take some time.
  * We also provide a [template script](https://github.com/ExpressAI/DataLab/blob/main/client/add_my_dataset.py) for adding your dataset into web platform, feel free to instantiate it based on your needs.
  * If your dataset is too large (>200M), please contact us.

#### (4) View your dataset in DataLab web platform

Once you successfully finished the above steps,  you can find the dataset
in your [private space]((https://datalab.nlpedia.ai/datasets_explore/user_dataset)) of DataLab web.

#### (5) Make your dataset Public (Optional)
