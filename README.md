# DataLab API CN

## Installation
#### Install


    ```shell
    git clone https://github.com/ExpressAI/Datalab.git
    cd Datalab
    pip install .
    ```

#### Uninstall
    ```shell
    pip uninstall datalab
    ```

#### Dataset Operation

* Step 1:
```shell
cd Datalab/datasets/

```

* Step 2:

```python

# pip install datalab
from datalabs import operations, load_dataset
from featurize import *

# don't forget `./`
dataset = load_dataset("./ag_news")

# print(task schema)
print(dataset['test']._info.task_templates)

# data operators
res = dataset["test"].apply(get_text_length)
print(next(res))


# get entity
res = dataset["test"].apply(get_entity_spacy)
print(next(res))

# get postag
res = dataset["test"].apply(get_postag_spacy)
print(next(res))

from edit import *
# add typos
res = dataset["test"].apply(add_typo)
print(next(res))

#  change person name
res = dataset["test"].apply(change_person_name)
print(next(res))



```

### Task Schema

* `text-classification`
    * `text`:str
    * `label`:ClassLabel
    
* `text-matching`
    * `text1`:str
    * `text2`:str
    * `label`:ClassLabel
    
* `summarization`
    * `text`:str
    * `summary`:str
    
* `sequence-labeling`
    * `tokens`:List[str]
    * `tags`:List[ClassLabel]
    
* `question-answering-extractive`:
    * `context`:str
    * `question`:str
    * `answers`:List[{"text":"","answer_start":""}]


one can use `dataset[SPLIT]._info.task_templates` to get more useful task-dependent information, where
`SPLIT` could be `train` or `validation` or `test`.


### Supported Datasets
* [here](https://github.com/ExpressAI/DataLab/tree/main/datasets)

   



