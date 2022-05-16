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

import csv
from email import header
import datalabs
from datalabs import get_task, TaskType


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
For more information, please refer to "https://www.datafountain.cn/competitions/350/datasets".   
"""

# You can copy an official description
_DESCRIPTION = """\
Task：Emotional Analysis of Internet News.
The goal of this task is to accurately distinguish the emotional polarity of text in a big data set. 
Emotions can be divided into three types: positive, negative and neutral.
2019 CCF Big Data & Computing Intelligence Contest (CCF BDCI) is an international challenge contest of intelligent algorithm, innovative application and big data system launched by Task Force on Big Data of China Computer Federation in 2013. 
It is one of the most influential events in the field of big data and AI across the world.
For more information, please refer to "https://www.datafountain.cn/competitions/350/datasets". 
"""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "N/A"

_HOMEPAGE = "https://www.datafountain.cn/competitions/350/datasets"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/BDCI2019/train.csv"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/BDCI2019/test.csv"

class BDCI2019(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(

            description=_DESCRIPTION,

            features=datalabs.Features(
                {
                    "title": datalabs.Value("string"),
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "neutral", "negative"]),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.sentiment_classification)(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path})
        ]

    def _generate_examples(self, filepath):
        
        textualize_label = {
            "0": "positive",
            "1": "neutral",
            "2": "negative"
        }

        header = 0
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for id_, row in enumerate(csv_reader):
                if header > 0:
                    num, news_id, label, title, text = row
                    if label in textualize_label:
                        label = textualize_label[label]
                        yield id_, {"title": title, "text": text, "label": label}
                header = header + 1
