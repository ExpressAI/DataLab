<p align="center">
    <br>
    <img src="./docs/Resources/figs/readme_logo.png" width="400"/>
    <br>
  <a href="https://github.com/expressai/DataLab/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/expressai/DataLab" /></a>
  <a href="https://github.com/expressai/DataLab/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/expressai/DataLab" /></a>
  <a href="https://pypi.org/project//"><img alt="PyPI" src="https://img.shields.io/pypi/v/datalabs" /></a>
  <a href=".github/workflows/ci.yml"><img alt="Integration Tests", src="https://github.com/expressai/DataLab/actions/workflows/ci.yml/badge.svg?event=push" />
</p>


[DataLab](http://datalab.nlpedia.ai/) is a unified platform that allows for NLP researchers to perform a number of data-related tasks in an efficient and easy-to-use manner. In particular, DataLab supports the following functionalities:

    

    
    
<p align="center"> 
<img src="./docs/Resources/figs/datalab_overview.png" width="300"/>
 </p> 

* **Data Diagnostics**: DataLab allows for analysis and understanding of data to uncover undesirable traits such as hate speech, gender bias, or label imbalance.
* **Operation Standardization**: DataLab provides and standardizes a large number of data processing operations, including aggregating, preprocessing, featurizing, editing and prompting operations.
* **Data Search**: DataLab provides a semantic dataset search tool to help identify appropriate datasets given a textual description of an idea.
* **Global Analysis**: DataLab provides tools to perform global analyses over a variety of datasets.

## Table of Content
* Installation SDK
* Supported Datasets
    * Datasets in [SDK](https://github.com/ExpressAI/DataLab/tree/main/datasets)
    * Datasets in [Web Platform](http://datalab.nlpedia.ai/datasets_explore/text_dataset)
* [Task Normalization](https://github.com/ExpressAI/DataLab/blob/main/docs/task_normalization/progress.md)
* Documentation for Web Users
    * [how to identify artifacts of datasets using DataLab?](https://github.com/ExpressAI/DataLab/blob/main/docs/functionality/how_to_identify_artifacts.md)
    * [how to analyze summarization datasets](https://github.com/ExpressAI/DataLab/blob/main/docs/introduction_of_summarization_datasets.md)  
  
* Documentation for SDK Users
    * [how to add a new task schema](https://github.com/ExpressAI/DataLab/blob/main/docs/add_new_task_schema.md)
    * [how to add language information into your dataset](https://github.com/ExpressAI/DataLab/blob/main/docs/add_language_info.md)
    * [how to add a new dataset into sdk](https://github.com/ExpressAI/DataLab/blob/main/docs/add_new_datasets_into_sdk.md)
    * [how to add a new dataset into the web platform](https://github.com/ExpressAI/DataLab/blob/main/docs/add_new_datasets_into_web_platform.md)
    
 


## Installation
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

## Getting started
Here we give several examples to showcase the usage of DataLab. For more information, please refer to the corresponding sections in our [documentation](https://expressai.github.io/DataLab/).





```python
# pip install datalabs
from datalabs import load_dataset
dataset = load_dataset("ag_news")


# Preprocessing operation
from preprocess import *
res=dataset["test"].apply(lower)
print(next(res))

# Featurizing operation
from featurize import *
res = dataset["test"].apply(get_text_length) # get length
print(next(res))

res = dataset["test"].apply(get_entities_spacy) # get entity
print(next(res))

# Editing/Transformation operation
from edit import *
res = dataset["test"].apply(change_person_name) #  change person name
print(next(res))

# Prompting operation
from prompt import *
res = dataset["test"].apply(template_tc1)
print(next(res))

# Aggregating operation
from aggregate.text_classification import *
res = dataset["test"].apply(get_statistics)
```
 

## Acknowledgment
DataLab originated from a fork of the awesome [Huggingface Datasets](https://github.com/huggingface/datasets) and [TensorFlow Datasets](https://github.com/tensorflow/datasets). We highly thank the Huggingface/TensorFlow Datasets for building this amazing library. More details on the differences between DataLab and them can be found in the section



