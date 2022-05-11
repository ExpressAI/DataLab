# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Iterator

from tqdm import tqdm

from datalabs.operations.aggregate import Aggregating, aggregating


class KGLinkPredictionAggregating(Aggregating):
    def __init__(self, *args, **kwargs):

        super(KGLinkPredictionAggregating, self).__init__(*args, **kwargs)
        self._data_type = "ag_news"


class kg_link_prediction_aggregating(aggregating):
    def __init__(self, *args, **kwargs):

        super(kg_link_prediction_aggregating, self).__init__(*args, **kwargs)
        # print(self.__dict__)

    def __call__(self, *param_arg):
        if callable(self.name):

            tf_class = KGLinkPredictionAggregating(
                name=self.name.__name__, func=self.name
            )
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = KGLinkPredictionAggregating(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                task=self.task,
                description=self.description,
            )
            return tf_cls


"""
from datalabs import load_dataset
dataset = load_dataset('./ag_news')
from ag_news.featurize import *
get_number_of_tokens.__dict__
res = dataset['test'].apply(get_number_of_tokens)
print(next(res))
"""


@kg_link_prediction_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="kg-link-prediction",
    description="aggregation function",
)
def get_statistics(samples: Iterator):
    dict_head = {}
    dict_link = {}
    dict_tail = {}

    for sample in tqdm(samples):

        if sample["tail"] not in dict_tail.keys():
            dict_tail[sample["tail"]] = 1
        else:
            dict_tail[sample["tail"]] += 1

        if sample["head"] not in dict_head.keys():
            dict_head[sample["head"]] = 1
        else:
            dict_head[sample["head"]] += 1

        if sample["link"] not in dict_link.keys():
            dict_link[sample["link"]] = 1
        else:
            dict_link[sample["link"]] += 1

    return {
        "head_fre": dict_head,
        "link_fre": dict_link,
        "tail_fre": dict_tail,
    }
