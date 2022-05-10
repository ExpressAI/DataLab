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
The dataset contains 4.4 million reviews of 240,000 restaurants from Dianping, a Chinese life service review app. 
For more information, please refer to https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_dianping/intro.ipynb. 
"""

_CITATION = """\
@inproceedings{10.1145/2488388.2488520,
author = {Zhang, Yongfeng and Zhang, Min and Liu, Yiqun and Ma, Shaoping and Feng, Shi},
title = {Localized Matrix Factorization for Recommendation Based on Matrix Block Diagonal Forms},
year = {2013},
isbn = {9781450320351},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/2488388.2488520},
doi = {10.1145/2488388.2488520},
abstract = {Matrix factorization on user-item rating matrices has achieved significant success in collaborative filtering based recommendation tasks. However, it also encounters the problems of data sparsity and scalability when applied in real-world recommender systems. In this paper, we present the Localized Matrix Factorization (LMF) framework, which attempts to meet the challenges of sparsity and scalability by factorizing Block Diagonal Form (BDF) matrices. In the LMF framework, a large sparse matrix is first transformed into Recursive Bordered Block Diagonal Form (RBBDF), which is an intuitionally interpretable structure for user-item rating matrices. Smaller and denser submatrices are then extracted from this RBBDF matrix to construct a BDF matrix for more effective collaborative prediction. We show formally that the LMF framework is suitable for matrix factorization and that any decomposable matrix factorization algorithm can be integrated into this framework. It has the potential to improve prediction accuracy by factorizing smaller and denser submatrices independently, which is also suitable for parallelization and contributes to system scalability at the same time. Experimental results based on a number of real-world public-access benchmarks show the effectiveness and efficiency of the proposed LMF framework.},
booktitle = {Proceedings of the 22nd International Conference on World Wide Web},
pages = {1511–1520},
numpages = {10},
keywords = {matrix factorization, block diagonal form, graph partitioning, collaborative filtering},
location = {Rio de Janeiro, Brazil},
series = {WWW '13}
}
"""

_LICENSE = "NA"

_DATA_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/yf_dianping/ratings.zip"

class YFDIANPING(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "user_id": datalabs.Value("string"),
                    "restaurant_id": datalabs.Value("string"),
                    "rating": datalabs.features.ClassLabel(names=["Excellent","Good","Average","Fair","Poor"]),
                    "rating_env": datalabs.features.ClassLabel(names=["Excellent","Good","Average","Fair","Poor"]),
                    "rating_flavor": datalabs.features.ClassLabel(names=["Excellent","Good","Average","Fair","Poor"]),
                    "rating_service": datalabs.features.ClassLabel(names=["Excellent","Good","Average","Fair","Poor"]),
                    "timestamp": datalabs.Value("string"),
                    "comment": datalabs.Value("string")
                }
            ),
            homepage="https://doi.org/10.1145/2488388.2488520",
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
            csv_reader = csv.reader( (line.replace('\0','') for line in csv_file) , delimiter = ',')
            id_actual = 0
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    continue
                if len(row) !=8:
                    continue
                user_id, restaurant_id, rating, rating_env, rating_flavor, rating_service, timestamp, comment = row
                if str(rating) in textualize_label and str(rating_env) in textualize_label and str(rating_flavor) in textualize_label and str(rating_service) in textualize_label: 
                    id_actual += 1
                    yield id_actual, {
                        "user_id": user_id,
                        "restaurant_id": restaurant_id,
                        "rating": textualize_label[str(rating)],
                        "rating_env": textualize_label[str(rating_env)],
                        "rating_flavor": textualize_label[str(rating_flavor)],
                        "rating_service": textualize_label[str(rating_service)],
                        "timestamp": timestamp,
                        "comment": comment
                    }

