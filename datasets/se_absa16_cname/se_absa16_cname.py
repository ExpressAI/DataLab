# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
The task type of this dataset is Aspect-level Sentiment Classification, and the task definition is as follows: 
for a given text (d) and an evaluation object (a) described in the text, the sentiment category (s) for the evaluation object (a) is given.
The category (s) generally only contains positive and negative. 
Each sample in the dataset is a triplet: Input text (d), Evaluation Object (a), and Emotion Category(s).
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=18.
"""

_CITATION = """\
@inproceedings{pontiki-etal-2016-semeval,
    title = "SemEval-2016 Task 5: Aspect Based Sentiment Analysis",
    author = {Pontiki, Maria  and
        Galanis, Dimitris  and
        Papageorgiou, Haris  and
        Androutsopoulos, Ion  and
        Manandhar, Suresh  and
        AL-Smadi, Mohammad  and
        Al-Ayyoub, Mahmoud  and
        Zhao, Yanyan  and
        Qin, Bing  and
        De Clercq, Orphee  and
        Hoste, Veronique  and
        Apidianaki, Marianna  and
        Tannier, Xavier  and
        Loukachevitch, Natalia  and
        Kotelnikov, Evgeniy  and
        Bel, Nuria  and
        Jimenez-Zafra, Salud Maria  and
        Eryigit, Gulsen},
    booktitle = "Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016)",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S16-1002",
    doi = "10.18653/v1/S16-1002",
    pages = "19--30",
}
"""

_LICENSE = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/SE-ABSA16/License.pdf"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/SE-ABSA16/SE-ABSA16-CAME-train.tsv"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/SE-ABSA16/SE-ABSA16-CAME-test.tsv"


class SEABSA16CAME(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=["positive", "negative"]
                    ),
                }
            ),
            homepage="https://aclanthology.org/S16-1002",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.sentiment_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            )
        ]

    def _generate_examples(self, filepath):

        textualize_label = {"1": "positive", "0": "negative"}

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                label, object, text = row
                if label == ("0" or "1"):
                    label = textualize_label[label]
                    yield id_, {"text": text, "label": label}
