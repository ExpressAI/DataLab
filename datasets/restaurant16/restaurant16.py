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
Restaurant16 contains annotated reviews of restaurants reviews. Each sample is labeled as positive,
neutral or negative w.r.t. a specific aspect. For more information, please refer to 
https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
"""

_CITATION = """\
See here https://aclanthology.org/S16-1002/
"""


_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1jBd8zTBUOwV6TmWHiURDldpbaB3GKjPK&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1f40Ie7mcwSHQKoJzGhnTq2SbFtclLyoa&export=download"


class Restaurant16(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "aspect": datalabs.Value("string"),
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=["positive", "negative", "neutral"]
                    ),
                }
            ),
            homepage="https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.aspect_based_sentiment_classification)(
                    span_column="aspect", text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate Restaurant16 examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                aspect, text, label = row
                yield id_, {"aspect": aspect, "text": text, "label": label}
