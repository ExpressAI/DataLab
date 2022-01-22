from datalabs import operations, load_dataset
from featurize import *

dataset = load_dataset("ag_news")

# print(task schema)
print(dataset['test']._info.task_templates)

# data operators
res = dataset["test"].apply_local(get_text_length, "length")
print(res[0])
