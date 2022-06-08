# coding=utf-8
# Copyright 2020 The TensorFlow datasets Authors and the HuggingFace, DataLab Authors.
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
"""Hate speech dataset"""


import csv
import os

import datalabs
from datalabs import get_task, TaskType


_DESCRIPTION = """\
    Contains 10k tweets (training set) that are labeled as hate speech or non-hate speech. Released with 4,232 validation and 4,232 testing samples. Collected during the 2016 Philippine Presidential Elections.
"""

_CITATION = """\
@article{Cabasag-2019-hate-speech,
  title={Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing.},
  author={Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng},
  journal={Philippine Computing Journal},
  volume={XIV},
  number={1},
  month={August},
  year={2019}
}
"""

_HOMEPAGE = "https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URL = "https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/hatenonhate/hatespeech_raw.zip"


class HateSpeechFilipino(datalabs.GeneratorBasedBuilder):
    """Hate Speech Text Classification Dataset in Filipino."""

    VERSION = datalabs.Version("1.0.0")

    def _info(self):
        # Labels: 0="Non-hate Speech", 1="Hate Speech"
        features = datalabs.Features(
            {"text": datalabs.Value("string"), "label": datalabs.features.ClassLabel(names=["0", "1"])}
        )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[get_task(TaskType.hatespeech_identification)(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)
        train_path = os.path.join(data_dir, "hatespeech", "train.csv")
        test_path = os.path.join(data_dir, "hatespeech", "train.csv")
        validation_path = os.path.join(data_dir, "hatespeech", "valid.csv")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": validation_path,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            next(csv_reader)
            for id_, row in enumerate(csv_reader):
                try:
                    text, label = row
                    yield id_, {"text": text, "label": label}
                except ValueError:
                    pass
