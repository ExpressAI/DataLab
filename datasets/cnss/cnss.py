# coding=utf-8
# Copyright 2020 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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


import json
import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Chinese News Same Story dataset (CNSS) can be used for a long text matching task, 
the task goal is to determine whether a pair of long news texts are reporting the same story.
"""

_CITATION = """\
@inproceedings{liu-etal-2019-matching,
    title = "Matching Article Pairs with Graphical Decomposition and Convolutions",
    author = "Liu, Bang  and
      Niu, Di  and
      Wei, Haojie  and
      Lin, Jinghong  and
      He, Yancheng  and
      Lai, Kunfeng  and
      Xu, Yu",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1632",
    doi = "10.18653/v1/P19-1632",
    pages = "6284--6294",
    abstract = "Identifying the relationship between two articles, e.g., whether two articles published from different sources describe the same breaking news, is critical to many document understanding tasks. Existing approaches for modeling and matching sentence pairs do not perform well in matching longer documents, which embody more complex interactions between the enclosed entities than a sentence does. To model article pairs, we propose the Concept Interaction Graph to represent an article as a graph of concepts. We then match a pair of articles by comparing the sentences that enclose the same concept vertex through a series of encoding techniques, and aggregate the matching signals through a graph convolutional network. To facilitate the evaluation of long article matching, we have created two datasets, each consisting of about 30K pairs of breaking news articles covering diverse topics in the open domain. Extensive evaluations of the proposed methods on the two datasets demonstrate significant improvements over a wide range of state-of-the-art methods for natural language matching.",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CNSS/train_revised.json"
)
_VALIDATION_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CNSS/validation_revised.json"
)
_TEST_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CNSS/test_revised.json"
)

_HOMEPAGE = "https://github.com/BangLiu/ArticlePairMatching"


class CNSSConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(CNSSConfig, self).__init__(**kwargs)


class CNSS(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CNSSConfig(
            name="origin",
            version=datalabs.Version("1.0.0"),
            description="origin",
        ),
    ]

    DEFAULT_CONFIG_NAME = "origin"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text1": datalabs.Value("string"),
                    "text2": datalabs.Value("string"),
                    "title1": datalabs.Value("string"),
                    "title2": datalabs.Value("string"),
                    "keywords1": datalabs.Value("string"),
                    "keywords2": datalabs.Value("string"),
                    "main_keywords1": datalabs.Value("string"),
                    "main_keywords2": datalabs.Value("string"),
                    "ner_keywords1": datalabs.Value("string"),
                    "ner_keywords2": datalabs.Value("string"),
                    "ner1": datalabs.Value("string"),
                    "ner2": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["0", "1"]),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.text_pair_classification)(
                    text1_column="text1", text2_column="text2", label_column="label"
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        valid_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": valid_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                yield id_, {
                    "text1": line["text1"],
                    "text2": line["text2"],
                    "title1": line["title1"],
                    "title2": line["title2"],
                    "keywords1": line["keywords1"],
                    "keywords2": line["keywords2"],
                    "main_keywords1": line["main_keywords1"],
                    "main_keywords2": line["main_keywords2"],
                    "ner_keywords1": line["ner_keywords1"],
                    "ner_keywords2": line["ner_keywords2"],
                    "ner1": line["ner1"],
                    "ner2": line["ner2"],
                    "label": line["label"],
                }
