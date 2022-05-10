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
from datalabs.tasks import TextClassification
import os
_DESCRIPTION = """\
Dataset from Amazon Reviews.
For more information, please refer to https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_amazon/intro.ipynb. 
"""

_CITATION = """\
xx
"""

_LICENSE = "NA"

_DATA_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/yf_amazon/ratings.zip"

class YFAmazon(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "user_id": datalabs.Value("string"),
                    "product_id": datalabs.Value("string"),
                    "rating": datalabs.features.ClassLabel(names=["Excellent","Good","Average","Fair","Poor"]),
                    "timestamp": datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                    "comment": datalabs.Value("string"),

                }
            ),
            homepage="xx",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[TextClassification(text_column="title", label_column="rating")],
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        data_dir = os.path.join(dl_dir, "")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "ratings.csv")}
            )
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
            csv_reader = csv.reader(csv_file, delimiter = ',')
            id_actual = 0
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    continue
                if len(row) !=6:
                    continue

                user_id, product_id, rating, timestamp, title, comment = row
                if str(rating) in textualize_label:
                    id_actual += 1
                    yield id_actual, {
                        "user_id": user_id,
                        "product_id": product_id,
                        "rating": textualize_label[str(rating)],
                        "timestamp": timestamp,
                        "title": title,
                        "comment": comment
                    }



