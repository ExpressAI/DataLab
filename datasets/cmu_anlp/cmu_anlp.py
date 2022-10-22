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
        datalabs.BuilderConfig(name='sciner'),
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
        elif subdataset.startswith('sciner'):
            return {
                "0": "O",
                "1": "B-DatasetName",
                "2": "B-HyperparameterName",
                "3": "B-HyperparameterValue",
                "4": "B-MethodName",
                "5": "B-MetricName",
                "6": "B-MetricValue",
                "7": "B-TaskName",
                "8": "I-DatasetName",
                "9": "I-HyperparameterName",
                "10": "I-HyperparameterValue",
                "11": "I-MethodName",
                "12": "I-MetricName",
                "13": "I-MetricValue",
                "14": "I-TaskName",
            }
        else:
            raise ValueError(f'bad subdataset {subdataset}')

    def _info(self):

        labels = list(self._get_labels_for_subdataset(self.config.name).values())

        if self.config.name == 'sciner':
            return datalabs.DatasetInfo(
                description=_DESCRIPTION,
                features=datalabs.Features(
                    {
                        "id": datalabs.Value("string"),
                        "tokens": datalabs.Sequence(datalabs.Value("string")),
                        "tags": datalabs.Sequence(
                            datalabs.features.ClassLabel(names=labels)
                        ),
                    }
                ),
                supervised_keys=None,
                homepage="https://phontron.com/class/anlp2022/",
                citation=_CITATION,
                task_templates=[
                    get_task(TaskType.named_entity_recognition)(
                        tokens_column="tokens", tags_column="tags"
                    )
                ],
                languages=["en"],
            )
        else:
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

        if self.config.name == 'sciner':
            test_path = dl_manager.download_and_extract(f'{_PRIVATE_PREFIX}/ner/{self.config.name}-test.conll')
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
                ),
            ]
        else:
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
        """Generate examples."""

        if self.config.name == 'sciner':

            with open(filepath, encoding="utf-8") as f:
                guid = 0
                tokens = []
                tags = []
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if tokens:
                            yield guid, {
                                "id": str(guid),
                                "tokens": tokens,
                                "tags": tags,
                            }
                            guid += 1
                            tokens = []
                            tags = []
                    else:
                        # conll2003 tokens are space separated
                        splits = line.split(" ")
                        tokens.append(splits[0])
                        tags.append(splits[3].rstrip())

                # last example
                if len(tokens) != 0:
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "tags": tags,
                    }

        else:
            # map the label into textual string
            textualize_label = self._get_labels_for_subdataset(self.config.name)

            with open(filepath, encoding="utf-8") as data_file:
                for id_, line in enumerate(data_file):
                    label, text = line.strip().split(' ||| ')
                    label = textualize_label[label]
                    yield id_, {"text": text, "label": label}
