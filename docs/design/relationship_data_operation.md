# How to arrange the relationship between Data and Operation

## Different Choices



(1) Choice 1
```python
    from datalabs import load_dataset
    from featurize import *
    
    dataset = load_dataset("mr")
    res = get_length(dataset['train'])
```

(2) Choice 2
```python
    from datalabs import load_dataset
    from featurize import *
    
    dataset = load_dataset("mr")
    res = dataset['train'].get_length
```


(3) Choice 3
```python
    from datalabs import load_dataset
    from featurize import *
    
    dataset = load_dataset("mr")
    res = dataset['train'].apply(get_length)
```






## Use Cases
(1) modify default parameters of `get_length`
```python
    res = dataset['train'].apply(get_length.set_param(processed_filed="text"))
```


(2) modify default parameters of `apply`
```python
    res = dataset['train'].apply(get_length, save = "in_memory")
```

