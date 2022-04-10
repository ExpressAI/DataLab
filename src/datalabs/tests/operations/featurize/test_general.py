# pip install datalab
from featurize import *

from datalabs import load_dataset, operations


# dataset = load_dataset("cnn_dailymail","3.0.0")
# res = dataset["test"].apply(get_oracle_summary)
# print(next(res))


dataset = load_dataset("ag_news")
res = dataset["test"].apply(get_length.set("text"))
print(next(res))
