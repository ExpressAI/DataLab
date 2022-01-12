# coding=utf-8
# Copyright 2020 The TensorFlow datalab Authors and the HuggingFace datalab Authors.
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

# Lint as: python3
"""Software Applications User Reviews"""


import csv

import datalabs
from datalabs.tasks import TextClassification

_DESCRIPTION = """\
It is a large dataset of Android applications belonging to 23 differentapps categories, which provides an overview of the types of feedback users report on the apps and documents the evolution of the related code metrics. The dataset contains about 395 applications of the F-Droid repository, including around 600 versions, 280,000 user reviews (extracted with specific text mining approaches)
"""

_CITATION = """\
@InProceedings{Zurich Open Repository and
Archive:dataset,
title = {Software Applications User Reviews},
authors={Grano, Giovanni; Di Sorbo, Andrea; Mercaldo, Francesco; Visaggio, Corrado A; Canfora, Gerardo;
Panichella, Sebastiano},
year={2017}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/sealuzh/user_quality/master/csv_files/reviews.csv"


class AppReviews(datalabs.GeneratorBasedBuilder):
    """Software Application Reviews by Users."""

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "package_name": datalabs.Value("string"),
                    "text": datalabs.Value("string"),
                    "date": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["1 star",
                                                                 "2 stars",
                                                                 "3 stars",
                                                                 "4 stars",
                                                                 "5 stars"]),
                }
            ),
            homepage="https://giograno.me/assets/pdf/workshop/wama17.pdf",
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate Distaster Response Messages examples."""
        textualize_label = {"1":"1 star",
                                 "2":"2 stars",
                                 "3":"3 stars",
                                 "4":"4 stars",
                                 "5":"5 stars"}
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            next(csv_reader, None)
            for id_, row in enumerate(csv_reader):
                row = row[1:5]
                (package_name, review, date, star) = row


                yield id_, {
                    "package_name": (package_name),
                    "text": review,
                    "date": (date),
                    "label": textualize_label[star],
                }
