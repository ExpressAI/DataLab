from datalabs import operations, load_dataset
from featurize import *
from aggregate import *

dataset = load_dataset("ag_news")

# data operators
res = dataset["test"].apply_test(get_average_length, mode="local")

# res = dataset["test"].apply(get_text_length)
# print(next(res))
