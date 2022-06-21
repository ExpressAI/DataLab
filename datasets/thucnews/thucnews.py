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


import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Thucnews is a long news text data set from Sina News and each text is labelled with one of 14 categories of news. 
The categories: (0) Sports, (1) Entertainment, (2) Home, (3) Lottery, (4) Real estate, (5) Education, 
(6) Fashion, (7) Politics, (8) Constellation, (9) Game, (10) Society, (11) Science, (12) Stock, (13) Finance.
For more information, please refer http://thuctc.thunlp.org. 
"""

_CITATION = """\
@inproceedings{li-sun-2007-scalable,
    title = "Scalable Term Selection for Text Categorization",
    author = "Li, Jingyang  and Sun, Maosong",
    booktitle = "Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning ({EMNLP}-{C}o{NLL})",
    month = jun,
    year = "2007",
    address = "Prague, Czech Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D07-1081",
    pages = "774--782",
}
"""


_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/thucnews/train.txt"
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/thucnews/dev.txt"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/thucnews/test.txt"
)


class THUCNEWS(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=[
                            "sports",
                            "entertainment",
                            "home",
                            "lottery",
                            "real estate",
                            "edu",
                            "fashion",
                            "politics",
                            "constellation",
                            "game",
                            "society",
                            "science",
                            "stock",
                            "finance",
                        ]
                    ),
                }
            ),
            homepage="http://thuctc.thunlp.org",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.topic_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):

        textualize_label = {
            "0": "sports",
            "1": "entertainment",
            "2": "home",
            "3": "lottery",
            "4": "real estate",
            "5": "edu",
            "6": "fashion",
            "7": "politics",
            "8": "constellation",
            "9": "game",
            "10": "society",
            "11": "science",
            "12": "stock",
            "13": "finance",
        }

        with open(filepath, encoding="utf-8") as txt_file:
            for id_, line in enumerate(txt_file):
                line_l = line.split("_!_")
                if len(line_l) == 4:
                    label = line_l[0]
                    text = line_l[3]
                    if label in textualize_label:
                        label = textualize_label[label]
                        yield id_, {"text": text, "label": label}
