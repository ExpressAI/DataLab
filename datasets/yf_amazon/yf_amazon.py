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
from datalabs.tasks import TextClassification
import os

_DESCRIPTION = """\
The dataset contains 7.2 million reviews for 520,000 products in more than 1,100 categories from Amazon.
For more information, please refer to https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_amazon/intro.ipynb. 
"""

_CITATION = """\
@inproceedings{10.1145/2736277.2741087,
author = {Zhang, Yongfeng and Zhang, Min and Zhang, Yi and Lai, Guokun and Liu, Yiqun and Zhang, Honghui and Ma, Shaoping},
title = {Daily-Aware Personalized Recommendation Based on Feature-Level Time Series Analysis},
year = {2015},
isbn = {9781450334693},
publisher = {International World Wide Web Conferences Steering Committee},
address = {Republic and Canton of Geneva, CHE},
url = {https://doi.org/10.1145/2736277.2741087},
doi = {10.1145/2736277.2741087},
abstract = {The frequently changing user preferences and/or item profiles have put essential importance on the dynamic modeling of users and items in personalized recommender systems. However, due to the insufficiency of per user/item records when splitting the already sparse data across time dimension, previous methods have to restrict the drifting purchasing patterns to pre-assumed distributions, and were hardly able to model them rather directly with, for example, time series analysis. Integrating content information helps to alleviate the problem in practical systems, but the domain-dependent content knowledge is expensive to obtain due to the large amount of manual efforts.In this paper, we make use of the large volume of textual reviews for the automatic extraction of domain knowledge, namely, the explicit features/aspects in a specific product domain. We thus degrade the product-level modeling of user preferences, which suffers from the lack of data, to the feature-level modeling, which not only grants us the ability to predict user preferences through direct time series analysis, but also allows us to know the essence under the surface of product-level changes in purchasing patterns. Besides, the expanded feature space also helps to make cold-start recommendations for users with few purchasing records.Technically, we develop the Fourier-assisted Auto-Regressive Integrated Moving Average (FARIMA) process to tackle with the year-long seasonal period of purchasing data to achieve daily-aware preference predictions, and we leverage the conditional opportunity models for daily-aware personalized recommendation. Extensive experimental results on real-world cosmetic purchasing data from a major e-commerce website (JD.com) in China verified both the effectiveness and efficiency of our approach.},
booktitle = {Proceedings of the 24th International Conference on World Wide Web},
pages = {1373–1383},
numpages = {11},
keywords = {sentiment analysis, collaborative filtering, time series analysis, recommender systems},
location = {Florence, Italy},
series = {WWW '15}
}
"""

_LICENSE = "NA"

_DATA_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/yf_amazon/ratings.zip"

class YFAMAZON(datalabs.GeneratorBasedBuilder):
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
            homepage="https://doi.org/10.1145/2736277.2741087",
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
