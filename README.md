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


 

## Installation
DataLab can be installed from PyPi
```bash
pip install --upgrade pip
pip install datalabs
python -m nltk.downloader omw-1.4 # to support more feature calculation
```
or from the source
```bash
# This is suitable for SDK developers
pip install --upgrade pip
git clone git@github.com:ExpressAI/DataLab.git
cd Datalab
pip install -e .[dev]
python -m nltk.downloader omw-1.4 # to support more feature calculation
```
By adding `[dev]`, some [extra libraries](https://github.com/ExpressAI/DataLab/blob/03f69e5424859e3e9dbcbb487d3e1ce3de45a599/setup.py#L66) will be installed, such as `pre-commit`.



#### Code Quality Check?
If you would like to contribute to DataLab, checking the code style and quality before your pull
request is highly recommended. In this project, three types of checks will be expected: (a) black
(2) flake8 (3) isort

you could achieve this in two ways:

##### Manually (suitable for developers using Github Destop)
```shell
pre-commit install
git init .
pre-commit run --all-files or
```
where `pre-commit run -all-files` is equivalent to
```shell
pre-commit run black   # (this is also equivalent to python -m black .)
pre-commit run isort   # (this is also equivalent to isort .)
pre-commit run flake8  # (this is  also equivalent to flake8)
```
Notably, `black` and `isort` can help us fix code style automatically, while `flake8` only
provide hints with us, which means we need to fix these issues raised by `flake8`.



##### Automatically (suitable for developers using Git CLI)

```shell
pre-commit install
git init .
git commit -m "your update message"
```
The `git commit` will automatically activate the command `pre-commit run -all-files`

## Using DataLab
Below we give several simple examples to showcase the usage of DataLab:

You can also view documentation:
* [**Online Tutorial**](https://expressai.github.io/DataLab/)
* [**Adding Datasets to the SDK**](docs/SDK/add_new_datasets_into_sdk.md)
* [**Adding a new Task to the SDK**](docs/SDK/add_new_task_schema.md)


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
DataLab originated from a fork of the awesome [Huggingface Datasets](https://github.com/huggingface/datasets) and [TensorFlow Datasets](https://github.com/tensorflow/datasets). We highly thank the Huggingface/TensorFlow Datasets for building this amazing library. More details on the differences between DataLab and them can be found in the section.
We thank Antonis Anastasopoulos for sharing the mapping data between countries and languages, and thank Alissa Ostapenko, Yulia Tsvetkov, Jie Fu, Ziyun Xu, Hiroaki Hayashi, and Zhengfu He for useful discussion and suggestions for the first version.



