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


import json

import datalabs
from datalabs import get_task, TaskType
from datalabs.utils import private_utils

_DESCRIPTION = """\
These are datasets from George Mason University's Advanced NLP course.
* 2023 Version: https://nlp.cs.gmu.edu/course/cs678-spring23/
"""

_CITATION = """\
"""

_PRIVATE_PREFIX = f"{private_utils.PRIVATE_LOC}/gmu_anlp"

class CmuAnlp(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(name='textclass'),
    ]

    @staticmethod
    def _get_labels_for_subdataset(subdataset: str) -> dict[str, str]:
        if subdataset.startswith('textclass'):
            return {
              "0": "Economic",
              "1": "Capacity and Resources",
              "2": "Morality",
              "3": "Fairness and Equality",
              "4": "Legality, Constitutionality, Jurisdiction",
              "5": "Policy Prescription and Evaluation",
              "6": "Crime and Punishment",
              "7": "Security and Defense",
              "8": "Health and Safety",
              "9": "Quality of Life",
              "10": "Cultural Identity",
              "11": "Public Sentiment",
              "12": "Political",
              "13": "External Regulation and Reputation",
              "14": "Other",
            }
        else:
            raise ValueError(f'bad subdataset {subdataset}')

    def _info(self):

        labels = list(self._get_labels_for_subdataset(self.config.name).values())
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=labels
                    ),
                    "language": datalabs.Value("string"),
                }
            ),
            homepage="https://nlp.cs.gmu.edu/course/cs678-spring23/",
            citation=_CITATION,
            languages=['bn', 'de', 'el', 'en', 'hi', 'it', 'ne', 'ru', 'sw', 'te', 'tr', 'zh'],
            task_templates=[
                get_task(TaskType.topic_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):

        test_path = dl_manager.download_and_extract(f'{_PRIVATE_PREFIX}/textclass/gmu-textclass-test.json')
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""

        with open(filepath, encoding="utf-8") as data_file:
            data = json.load(data_file)
            for id_, line in enumerate(data):
                yield id_, {"text": line["text"], "label": line["label"], "language": line["language"]}
