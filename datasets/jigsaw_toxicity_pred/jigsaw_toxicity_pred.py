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
This dataset consists of a large number of Wikipedia comments which have been labeled by human raters for toxic behavior.
"""

_HOMEPAGE = "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"


_URL = "https://s3.amazonaws.com/datalab-hub/toxicity_detection/jigsaw_toxicity_pred.zip"


_LICENSE = 'The "Toxic Comment Classification" dataset is released under CC0, with the underlying comment text being governed by Wikipedia\'s CC-SA-3.0.'



class JigsawToxicityPredConfig(datalabs.BuilderConfig):
    """BuilderConfig for JigsawToxicityPred"""

    def __init__(self,
                 text_column=None,
                 label_column=None,
                 task_templates = None,
                 **kwargs):
        """BuilderConfig for JigsawToxicityPred.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(JigsawToxicityPredConfig, self).__init__(**kwargs)
        self.text_column = text_column
        self.label_column = label_column
        self.task_templates = task_templates


class JigsawToxicityPred(datalabs.GeneratorBasedBuilder):
    """This is a dataset of comments from Wikipedia’s talk page edits which have been labeled by human raters for toxic behavior."""

    VERSION = datalabs.Version("1.1.0")
    BUILDER_CONFIGS = [
        JigsawToxicityPredConfig(name=key,
                            version=datalabs.Version("1.0.0"),
                            description="JigsawToxicity predict",
                            text_column="text",
                            label_column="label",
                            task_templates=[
                                get_task(TaskType.toxicity_identification)(text_column="text", label_column="label")],
                            )
        for key in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.ClassLabel(names=["false", "true"]),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            task_templates=[get_task(TaskType.toxicity_identification)(text_column="text", label_column="label")],
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
                gen_kwargs={"train_path": os.path.join(data_dir, "jigsaw_toxicity_pred", "train.csv"), "split": "train"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "test_text_path": os.path.join(data_dir, "jigsaw_toxicity_pred", "test.csv"),
                    "test_labels_path": os.path.join(data_dir, "jigsaw_toxicity_pred", "test_labels.csv"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, split="train", train_path=None, test_text_path=None, test_labels_path=None):
        """Yields examples."""
        # This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)

        if split == "test":
            df1 = pd.read_csv(test_text_path)
            df2 = pd.read_csv(test_labels_path)
            df3 = df1.merge(df2)
            df4 = df3[df3["toxic"] != -1]

        elif split == "train":
            df4 = pd.read_csv(train_path)

        for _, row in df4.iterrows():
            example = {}
            example["text"] = row["comment_text"]
            # example[label] = int(row[label])
            example["label"] = row[self.config.name]

            yield row["id"], example



            # example["comment_text"] = row["comment_text"]

            # for label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
            #     if row[label] != -1:
            #         if label =="toxic":
            #             example["label"] = int(row[label])
            #         else:
            #             example[label] = int(row[label])
            # yield (row["id"], example)