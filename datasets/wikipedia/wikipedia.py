# coding=utf-8


"""Dataset config script for ag_news （this code is originally from huggingface, them modified by datalab）"""


import csv
from typing import Dict, List, Any, Optional, Iterator
import datalabs
from datalabs.tasks import TextClassification
from datalabs import StructuredTextData

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






# class AGNewsDataset(Dataset):
#     def apply(self, func):
#         if func._type == 'Aggregating':


class Wikipedia(StructuredTextData):


    def __init__(self, data:Iterator):

        self.data = {"text":["This is a good movie", "I love this movie"]}

    def apply(self, func):
        for sample in self.data:
            yield func(sample)
