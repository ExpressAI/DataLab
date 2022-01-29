# How to arrange the relationship between Data and Operation


```python
    from datalabs import load_dataset
    from featurize import *
    
    dataset = load_dataset("mr")
    res = dataset.apply(get_length)
```
