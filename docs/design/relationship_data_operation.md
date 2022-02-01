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

## How to make operation easily being contributed?

## Put `if-else` in `apply` function
Question: do we have better design w.r.t following code?


```python
    def apply(self, func):
          yield template_p1(sample, labels_to_answers, func)

        if func._type == 'Aggregating':
            yield func(self[func.processed_fields[0]])
        elif func._type.find("Aggregating")!=-1:
            yield func(self)
        elif func._type in ["Editing","Preprocessing", "Featurizing","OperationFunction"]:
            for sample in self.__iter__():
                yield func(sample[func.processed_fields[0]])
        elif func._type  in ["TopicClassificationPrompting", "SentimentClassificationPrompting", "NLIPrompting"]:
            for sample in self.__iter__():
                labels = self._info.task_templates[0].labels
                labels_to_answers = dict(zip(range(len(labels)), labels))
                yield func(sample, labels_to_answers)

        else:
            for sample in self.__iter__():
                yield func(sample)
```

## Multi-Process/Batch Processing
(1) Question: enable operations multiple-processable?
Hint: probably we can learn the implementation from huggingface dataset 
[map](https://huggingface.co/docs/datasets/processing.html#processing-data-with-map)
[multiprocessing](https://huggingface.co/docs/datasets/processing.html#multiprocessing)

(2) how to support batch?
Learn from [batch](https://huggingface.co/docs/datasets/processing.html#processing-data-in-batches)


## Unify `apply`, `apply_save`, `apply_local`?


## Unify the returned format of operations

## Design Optimization
- [ ] Different Operation Classes
- [ ] 3-party library
- [ ] pre-computed models, such as `spacy`

