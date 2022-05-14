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
The task type of this dataset is sentence-level sentiment classification, and the task definition is as follows: 
For a given text (d), the system needs to give the corresponding emotional score (s) according to the content of the text (d). 
And score (s) is an integer ranging from 1 to 5.
Each sample in the dataset is a two-tuple: Input text (d), emotional score (s).
For convenience, we have converted scores to the following labels in advance:
"Excellent","Good","Average","Fair", and "Poor".
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=17. 
"""

_CITATION = """\
@inproceedings{bu-etal-2021-asap,
    title = "ASAP: A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction",
    author = "Bu, Jiahao  and
      Ren, Lei  and
      Zheng, Shuang  and
      Yang, Yang  and
      Wang, Jingang  and
      Zhang, Fuzheng  and
      Wu, Wei",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.167",
    doi = "10.18653/v1/2021.naacl-main.167",
    pages = "2069--2079",
    abstract = "Sentiment analysis has attracted increasing attention in e-commerce. The sentiment polarities underlying user reviews are of great value for business intelligence. Aspect category sentiment analysis (ACSA) and review rating prediction (RP) are two essential tasks to detect the fine-to-coarse sentiment polarities. ACSA and RP are highly correlated and usually employed jointly in real-world e-commerce scenarios. While most public datasets are constructed for ACSA and RP separately, which may limit the further exploitation of both tasks. To address the problem and advance related researches, we present a large-scale Chinese restaurant review dataset ASAP including 46, 730 genuine reviews from a leading online-to-offline (O2O) e-commerce platform in China. Besides a 5-star scale rating, each review is manually annotated according to its sentiment polarities towards 18 pre-defined aspect categories. We hope the release of the dataset could shed some light on the field of sentiment analysis. Moreover, we propose an intuitive yet effective joint model for ACSA and RP. Experimental results demonstrate that the joint model outperforms state-of-the-art baselines on both tasks.",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/ASAP/ASAP_SENT/train.tsv"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/ASAP/ASAP_SENT/dev.tsv"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/ASAP/ASAP_SENT/test.tsv"

class ASAPSENT(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["Excellent","Good","Average","Fair","Poor"])
                }
            ),
            homepage="https://aclanthology.org/2021.naacl-main.167",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.sentiment_classification)(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]
       

    def _generate_examples(self, filepath):

        textualize_label = {
            "1": "Poor",
            "2": "Fair",
            "3": "Average",
            "4": "Good",
            "5": "Excellent",
            "1.0": "Poor",
            "2.0": "Fair",
            "3.0": "Average",
            "4.0": "Good",
            "5.0": "Excellent"
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = '\t')
            for id_, row in enumerate(csv_reader):
                if len(row) == 2:
                    text, label = row
                    if label in textualize_label:
                        label = textualize_label[label]
                        yield id_, {"text": text, "label": label}
                elif len(row) == 3:
                    id, text, label = row
                    if label in textualize_label:
                        label = textualize_label[label]
                        yield id_, {"text": text, "label": label}
