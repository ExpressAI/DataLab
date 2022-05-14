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
import datalabs
from datalabs import get_task, TaskType

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
For more information, please refer to "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb".   
"""

# You can copy an official description
_DESCRIPTION = """\
The data set contains more than 100,000 posts on Sina Weibo, China's twitter-like microblogging service, 
with about 50,000 positive messages and about 50,000 negative ones. 
For more information, please refer to "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb". 
"""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "N/A"

_HOMEPAGE = "https://github.com/SophonPlus/ChineseNlpCorpus"

_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/weibo_senti/weibo_senti.csv"

class WeiboSenti(datalabs.GeneratorBasedBuilder):
    def _info(self):

        return datalabs.DatasetInfo(

            description=_DESCRIPTION,

            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative"]),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.sentiment_classification)(
                text_column="text",
                label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_URL)
        print(f"train_path: \t{train_path}")
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path})
        ]

    def _generate_examples(self, filepath):
        
        textualize_label = {
            "1": "positive",
            "0": "negative"
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for id_, row in enumerate(csv_reader):
                label, text = row
                if label == "0" or label == "1":
                    label = textualize_label.get(label)
                    yield id_, {"text": text, "label": label}
