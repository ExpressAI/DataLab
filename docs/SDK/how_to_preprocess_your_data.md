# How to Preprocess you Data?

Data preprocessing (e.g., tokenization) is an indispensable step in training deep  learning and machine learning models,
and the quality of the dataset directly affects the learning of models. Currently, DataLab supports both general preprocessing functions
and task-specific ones, which are built based on different sources, such as Spacy, NLTK and Huggingface.

## Interface

```python
from datalabs import load_dataset
from preprocess import *


# realtime mode
dataset = load_dataset("ag_news")
res=dataset["test"].apply(lower, mode = "realtime") # res:Iterator
print(next(res))

# memory mode
res=dataset["test"].apply(lower, mode="memory") # res:Dataset
print(res)


# local mode
res=dataset["test"].apply(lower, mode="local") # res:Dataset
print(res)
```

## Supported Operations

You can find more operations supported by DataLab SDK [here](http://datalab.nlpedia.ai/operations)

## Define your own Operations

You can define the operation by yourself

```python
from datalabs import load_dataset

@datalabs.preprocessing
def lower(text):
    return text.lower()
    
dataset = load_dataset("ag_news")
res=dataset["test"].apply(lower, mode = "realtime") # res:Iterator    
    
```
