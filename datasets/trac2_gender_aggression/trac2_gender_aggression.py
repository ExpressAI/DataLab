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

import os

import pandas as pd

# import datasets

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Sub-task B: Misogynistic Aggression Identification Shared Task: This task will be to develop a binary classifier for classifying the text as ‘gendered’ or ‘non-gendered’. We will provide a dataset of 5,000 annotated data from social media each in Bangla (in both Roman and Bangla script), Hindi (in both Roman and Devanagari script) and English for training and validation. We will release additional data for testing your system.
"""

_HOMEPAGE = "https://docs.google.com/spreadsheets/d/1fZNZEi52i5GcsI2SqlTpRFC6OGUbhnJNSEDLlo1H624/edit#gid=0"


_URL = "https://s3.amazonaws.com/datalab-hub/toxicity_detection/trac2_taskAB.zip"




class TRAC2AggressionPred(datalabs.GeneratorBasedBuilder):
    """This is a dataset of comments from Wikipedia’s talk page edits which have been labeled by human raters for toxic behavior."""

    VERSION = datalabs.Version("1.1.0")

    def _info(self):

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            # license=_LICENSE,
            task_templates=[get_task(TaskType.hatespeech_identification)(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name
        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "trac2_taskAB", "trac2_eng_train.csv"),
                            "split": "train"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "trac2_taskAB", "trac2_eng_dev.csv"),
                            "split": "dev"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "trac2_taskAB", "trac2_eng_test.csv"),
                            "split": "test"},
            ),

        ]

    def _generate_examples(self, filepath,split):
        """Yields examples."""
        # This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)

        df4 = pd.read_csv(filepath)

        for _, row in df4.iterrows():
            example = {}
            example["text"] = row["Text"]
            example["label"] = row["Sub-task B"]
            yield (row["ID"], example)
