<p align="center">
    <br>
    <img src="./docs/resources/figs/readme_logo.png" width="400"/>
    <br>
  <a href="https://github.com/expressai/DataLab/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/expressai/DataLab" /></a>
  <a href="https://github.com/expressai/DataLab/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/expressai/DataLab" /></a>
  <a href="https://pypi.org/project//"><img alt="PyPI" src="https://img.shields.io/pypi/v/datalabs" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
  <a href=".github/workflows/ci.yml"><img alt="Integration Tests", src="https://github.com/neulab/ExplainaBoard/actions/workflows/ci.yml/badge.svg?event=push" />
</p>


[DataLab](http://datalab.nlpedia.ai/) is a unified platform that allows for NLP researchers to perform a number of data-related tasks in an efficient and easy-to-use manner. In particular, DataLab supports the following functionalities:

    

    
    
<!-- <p align="center"> -->
<img src="./docs/resources/figs/datalab_overview.png" width="300"/>
<!-- </p> -->

* **Data Diagnostics**: DataLab allows for analysis and understanding of data to uncover undesirable traits such as hate speech, gender bias, or label imbalance.
* **Operation Standardization**: DataLab provides and standardizes a large number of data processing operations, including aggregating, preprocessing, featurizing, editing and prompting operations.
* **Data Search**: DataLab provides a semantic dataset search tool to help identify appropriate datasets given a textual description of an idea.
* **Global Analysis**: DataLab provides tools to perform global analyses over a variety of datasets.


# Installation
DataLab can be installed from PyPi
```bash
pip install --upgrade pip
pip install datalabs
```
or from the source
```bash
# This is suitable for SDK developers
pip install --upgrade pip
git clone git@github.com:ExpressAI/DataLab.git
cd Datalab
pip install .
```

# Getting started
Here we give several examples to showcase the usage of DataLab. For more information, please refer to the corresponding sections in our [documentation](https://expressai.github.io/DataLab/).

```python
# pip install datalabs
from datalabs import operations, load_dataset
from featurize import *

 
dataset = load_dataset("ag_news")

# print(task schema)
print(dataset['test']._info.task_templates)

# data operators
res = dataset["test"].apply(get_text_length)
print(next(res))


# get entity
res = dataset["test"].apply(get_entities_spacy)
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
# Task Schema

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


# Supported Datasets
* [here](https://github.com/ExpressAI/DataLab/tree/main/datasets)

   

# Acknowledgment
DataLab originated from a fork of the awesome [Huggingface Datasets](https://github.com/huggingface/datasets) and [TensorFlow Datasets](https://github.com/tensorflow/datasets). We highly thank the Huggingface/TensorFlow Datasets for building this amazing library. More details on the differences between DataLab and them can be found in the section



