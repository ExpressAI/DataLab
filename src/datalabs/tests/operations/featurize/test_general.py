# pip install datalab
from datalabs import operations, load_dataset
from featurize import *


# dataset = load_dataset("cnn_dailymail","3.0.0")
# res = dataset["test"].apply(get_oracle_summary)
# print(next(res))


dataset = load_dataset("ag_news")
res = dataset["test"].apply(get_length.set("text"))
print(next(res))













