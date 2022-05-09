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
from datalabs.tasks import TextClassification
from datalabs import Dataset


_CITATION = """\
For more information, please refer to "http://tcci.ccf.org.cn/conference/2014/pages/page04_sam.html".   
"""

_DESCRIPTION = """\
This dataset is from NLPCC 2014 Sentiment Classification Task, which aims to evaluate the effect of deep learning methods on sentiment classification tasks. 
The dataset covers multiple fields (eg. books, DVD, electronic products, etc.), including positive and negative labels. 
For more information, please refer to "https://www.luge.ai/#/luge/dataDetail?id=20". 
"""

_LICENSE = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/NLPCC14-SC/License.pdf"

_HOMEPAGE = "http://tcci.ccf.org.cn/conference/2014/pages/page04_sam.html"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/NLPCC14-SC/train.tsv"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/NLPCC14-SC/test.tsv"

class NLPCC14SC(datalabs.GeneratorBasedBuilder):
    def _info(self):
        
        return datalabs.DatasetInfo(

            description=_DESCRIPTION,

            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative"])
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[TextClassification(text_column="text", label_column="label", task="sentiment-classification")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
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
            csv_reader = csv.reader(csv_file, delimiter = '\t')
            for id_, row in enumerate(csv_reader):
                label, text = row
                if label == ("0" or "1"):
                    label = textualize_label[label]
                    yield id_, {"text": text, "label": label}
