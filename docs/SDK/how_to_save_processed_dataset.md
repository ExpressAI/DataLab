# How to Save Processed Datasets?

DataLab provides different modes for saving processed datasets. We will walk through them using the `ag_new` as an example.

(Note: the default mode is `realtime`.)

### 1: `realtime`

```python
from datalabs import load_dataset
from featurize import *

# load dataset
dataset = load_dataset("ag_news")
# calculate the text length for each sample and return dataset_iterator:Iterator
dataset_iterator = dataset['test'].apply(get_length, mode="realtime") # dataset_iterator is an Iterator
print(next(dataset_iterator))

"""
printed results:
{'text_length': 27}
"""

```

### 2: `memory`

```python
from datalabs import load_dataset
from featurize import *

# load dataset
dataset = load_dataset("ag_news")
# calculate the text length for each sample and return dataset_new:Dataset (not an iterator)
dataset_new = dataset['test'].apply(get_length, mode="memory") # dataset_new is the same as dataset but with a new feature `text_length`
print(dataset_new)
"""
printed results of dataset_new
Dataset({
    features: ['text', 'label', 'text_length'],
    num_rows: 7600
})
"""
```

### 3: `local`

```python
from datalabs import load_dataset
from featurize import *

# load dataset
dataset = load_dataset("ag_news")
# calculate the text length for each sample and (1) return dataset_new:Dataset (not an iterator) (2) save the dataset_new locally,
# so that you can directly load the new version next time.
dataset_new = dataset['test'].apply(get_length, mode="local") # dataset_new is the same as dataset but with a new feature `text_length`
print(dataset_new)
"""
printed results of dataset_new
Dataset({
    features: ['text', 'label', 'text_length'],
    num_rows: 7600
})
"""
```
