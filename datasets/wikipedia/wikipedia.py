# coding=utf-8


"""Dataset config script for ag_news （this code is originally from huggingface, them modified by datalab）"""

import os
import csv
import sys
# we must expand the sys.path so that we can import local ops.py by:
# importlib.import_module('ops')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import importlib
import py7zr
from typing import Dict, List, Any, Optional, Iterator
import datalabs
from datalabs.tasks import TextClassification
from datalabs import StructuredTextData
from datalabs.operations.featurize.featurizing import featurizing, Featurizing

"""Caveat
We cannot use following statement otherwise we can not download the ops.py file from remote server. 
This restrict is made by the function load.py:get_imports 
`from . import ops`
"""
from .ops import *



 

_DESCRIPTION = """\
AG is a collection of more than 1 million news articles. News articles have been
gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of
activity. ComeToMyHead is an academic news search engine which has been running
since July, 2004. The dataset is provided by the academic comunity for research
purposes in data mining (clustering, classification, etc), information retrieval
(ranking, search, etc), xml, data compression, data streaming, and any other
non-commercial activity. For more information, please refer to the link
http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

The AG's news topic classification dataset is constructed by Xiang Zhang
(xiang.zhang@nyu.edu) from the dataset above. It is used as a text
classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann
LeCun. Character-level Convolutional Networks for Text Classification. Advances
in Neural Information Processing Systems 28 (NIPS 2015).
"""

_CITATION = """\
@inproceedings{Zhang2015CharacterlevelCN,
  title={Character-level Convolutional Networks for Text Classification},
  author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
  booktitle={NIPS},
  year={2015}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"



def get_operations(module_path:str):
    all_operations = {}
    module = importlib.import_module(module_path)


    target_class = None
    # print(module)
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, Featurizing):
            target_class = obj
            break


    if target_class == None:
        raise ValueError("target class is none!")

    for name, obj in module.__dict__.items():
        if isinstance(obj, target_class):
            all_operations[obj] = name

    return all_operations


# class AGNewsDataset(Dataset):
#     def apply(self, func):
#         if func._type == 'Aggregating':


class Wikipedia(StructuredTextData):


    def __init__(self, data:Iterator = None):

        self.data = [{"text":"This is a good movie"}]
        self.homedir = os.environ['HOME']
        self.data_remote = "https://dumps.wikimedia.org/other/static_html_dumps/current/en/wikipedia-en-html.tar.7z"
        self.max_cpu = os.cpu_count()
        self.data_local = self.download_to_local()

        self.documents = None
        self.tokens = None
        self.size = None


    def __repr__(self):

        module_path = "ops"
        all_operations = get_operations(module_path)


        repr = "\n\t" + "data_local_directory: " + self.data_local + "\n\t"
        repr += f"Following operations can be applied: \n\t"
        repr += "\n\t\t - ".join([v for k, v in all_operations.items()])
        repr += "\n\t\n\t" + f"Example: data.apply({list(all_operations.values())[0]}))"
        return f"StructureTextData({{\n{repr}\n}})"



    def download_to_local(self):
        homedir = self.homedir
        os.makedirs(f"{homedir}/.cache", exist_ok=True)
        if not os.path.exists(f"{homedir}/.cache/wikipedia/en"):
            os.makedirs(f"{homedir}/.cache/wikipedia", exist_ok=True)
            os.makedirs(f"{homedir}/.cache/wikipedia/en")
            # Download and extract data
            # TODO: Use python package instead of 7z. pip install py7zr
            os.system(f"wget {self.data_remote} -O {homedir}/.cache/wikipedia-en-html.tar.7z")
            os.system(f"7z x -so {homedir}/.cache/wikipedia-en-html.tar.7z | tar xf - -C {homedir}/.cache/wikipedia")
        return f"{homedir}/.cache/wikipedia/en"

    # def download_to_local(self):
    #     homedir = self.homedir
    #     os.makedirs(f"{homedir}/.cache", exist_ok=True)
    #     if not os.path.exists(f"{homedir}/.cache/wikipedia/en"):
    #         os.makedirs(f"{homedir}/.cache/wikipedia", exist_ok=True)
    #         os.makedirs(f"{homedir}/.cache/wikipedia/en")
    #         # Download and extract data
    #         os.system(f"wget {self.data_remote} -O {homedir}/.cache/wikipedia-en-html.tar.7z")
    #         with py7zr.SevenZipFile("wikipedia-en-html.tar.7z", "r") as z:
    #             z.extractall(f"{homedir}/.cache/")
    #         with tarfile.open("wikipedia-en-html.tar", "r") as f:
    #             f.extractall(f"{homedir}/.cache/wikipedia")
    #         # os.system(f"7z x -so {homedir}/.cache/wikipedia-en-html.tar.7z | tar xf - -C {homedir}/.cache/wikipedia")
    #     else:
    #         with py7zr.SevenZipFile(f"{homedir}/.cache/wikipedia-en-html.tar.7z", "r") as z:
    #             z.extractall(f"{homedir}/.cache/")
    #         with tarfile.open(f"{homedir}/.cache/wikipedia-en-html.tar", "r") as f:
    #             f.extractall(f"{homedir}/.cache/wikipedia")
    #     return f"{homedir}/.cache/wikipedia/en"






    def apply(self, func):
        for sample in self.data:
            yield func(sample)
