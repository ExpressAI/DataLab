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
This dataset contains more than 2 million reviews and scores for 28 films. 
The score is an integer ranging from 1 to 5. 
For convenience, we have converted scores to the following labels in advance:
"Excellent","Good","Average","Fair", and "Poor".
For more information, please refer to https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/dmsc_v2/intro.ipynb. 
"""

_CITATION = """\
For more information, please refer to https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/dmsc_v2/intro.ipynb.
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/douban_movie/douban_movie.csv"

class DOUBANMOVIE(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["Excellent","Good","Average","Fair","Poor"])
                }
            ),
            homepage="https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/dmsc_v2/intro.ipynb",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.sentiment_classification)(
                text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]
       

    def _generate_examples(self, filepath):

        textualize_label = {
            "1": "Poor",
            "2": "Fair",
            "3": "Average",
            "4": "Good",
            "5": "Excellent"
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for id_, row in enumerate(csv_reader):
                userId,movieId,label,timestamp,text,like = row
                if label in textualize_label:
                    label = textualize_label[label]
                    yield id_, {"text": text, "label": label}