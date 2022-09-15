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
from datalabs.utils import private_utils

_DESCRIPTION = """\
These are datasets from CMU's Advanced NLP course.
* 2022 Version: https://phontron.com/class/anlp2022/
"""

_CITATION = """\
"""

_MINBERT_PREFIX = "https://raw.githubusercontent.com/neubig/minbert-assignment/main/data"
_PRIVATE_PREFIX = f"{private_utils.PRIVATE_LOC}/cmu_anlp"

class CmuAnlp(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(name='sst'),
        datalabs.BuilderConfig(name='cfimdb'),
    ]

    @staticmethod
    def _get_labels_for_subdataset(subdataset: str) -> dict[str, str]:
        if subdataset.startswith('sst'):
            return {
                "0": "very negative",
                "1": "negative",
                "2": "neutral",
                "3": "positive",
                "4": "very positive",
            }
        elif subdataset.startswith('cfimdb'):
            return {
                "0": "negative",
                "1": "positive",
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
                }
            ),
            homepage="https://phontron.com/class/anlp2022/",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.sentiment_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(f'{_MINBERT_PREFIX}/{self.config.name}-train.txt')
        validation_path = dl_manager.download_and_extract(f'{_MINBERT_PREFIX}/{self.config.name}-dev.txt')
        test_path = dl_manager.download_and_extract(f'{_PRIVATE_PREFIX}/textclass/{self.config.name}-test.txt')
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
        """Generate SST2 examples."""

        # map the label into textual string
        textualize_label = self._get_labels_for_subdataset(self.config.name)

        with open(filepath, encoding="utf-8") as data_file:
            for id_, line in enumerate(data_file):
                label, text = line.strip().split(' ||| ')
                label = textualize_label[label]
                yield id_, {"text": text, "label": label}
