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
"""TODO(empathetic_dialogues): Add a description here."""


import csv

import datalabs
from datalabs import get_task, TaskType



_CITATION = """\
@inproceedings{rashkin2019towards,
  title = {Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset},
  author = {Hannah Rashkin and Eric Michael Smith and Margaret Li and Y-Lan Boureau},
  booktitle = {ACL},
  year = {2019},
}
"""

_DESCRIPTION = """\
PyTorch original implementation of Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset
"""
_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"


class EmpatheticDialogues(datalabs.GeneratorBasedBuilder):
    """TODO(empathetic_dialogues): Short description of my dataset."""

    # TODO(empathetic_dialogues): Set up version.
    VERSION = datalabs.Version("0.1.0")

    def _info(self):
        # TODO(empathetic_dialogues): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "conv_id": datalabs.Value("string"),
                    "utterance_idx": datalabs.Value("int32"),
                    "emotion": datalabs.Value("string"), # context --> emotion
                    "situation": datalabs.Value("string"), # prompt -> situation
                    "speaker_idx": datalabs.Value("int32"),
                    "utterance": datalabs.Value("string"),
                    "selfeval": datalabs.Value("string"),
                    "tags": datalabs.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/facebookresearch/EmpatheticDialogues",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.dialogue_empathetic)(
                    situation_column="situation",
                    utterance_column="utterance",
                    emotion_column="emotion",
                )
            ],
        )



    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(empathetic_dialogues): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        archive = dl_manager.download(_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split_file": "empatheticdialogues/train.csv"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split_file": "empatheticdialogues/valid.csv"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split_file": "empatheticdialogues/test.csv"},
            ),
        ]

    def _generate_examples(self, files, split_file):
        """Yields examples."""
        for path, f in files:
            if split_file == path:
                data = csv.DictReader(line.decode("utf-8") for line in f)
                for id_, row in enumerate(data):
                    utterance = row["utterance"]
                    speaker_id = int(row["speaker_idx"])
                    context = row["context"]
                    conv_id = row["conv_id"]
                    tags = row["tags"] if row["tags"] else ""
                    selfeval = row["selfeval"] if row["selfeval"] else ""
                    utterance_id = int(row["utterance_idx"])
                    prompt = row["prompt"]
                    yield id_, {
                        "conv_id": conv_id,
                        "utterance_idx": utterance_id,
                        "emotion": context,
                        "situation": prompt,
                        "speaker_idx": speaker_id,
                        "utterance": utterance,
                        "selfeval": selfeval,
                        "tags": tags,
                    }
                break




