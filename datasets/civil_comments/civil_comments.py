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

"""CivilComments from Jigsaw Unintended Bias Kaggle Competition."""


import csv
import os

import datalabs
from datalabs import get_task, TaskType


_CITATION = """
@article{DBLP:journals/corr/abs-1903-04561,
  author    = {Daniel Borkan and
               Lucas Dixon and
               Jeffrey Sorensen and
               Nithum Thain and
               Lucy Vasserman},
  title     = {Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
               Classification},
  journal   = {CoRR},
  volume    = {abs/1903.04561},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.04561},
  archivePrefix = {arXiv},
  eprint    = {1903.04561},
  timestamp = {Sun, 31 Mar 2019 19:01:24 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-04561},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
The comments in this dataset come from an archive of the Civil Comments
platform, a commenting plugin for independent news sites. These public comments
were created from 2015 - 2017 and appeared on approximately 50 English-language
news sites across the world. When Civil Comments shut down in 2017, they chose
to make the public comments available in a lasting open archive to enable future
research. The original data, published on figshare, includes the public comment
text, some associated metadata such as article IDs, timestamps and
commenter-generated "civility" labels, but does not include user ids. Jigsaw
extended this dataset by adding additional labels for toxicity and identity
mentions. This data set is an exact replica of the data released for the
Jigsaw Unintended Bias in Toxicity Classification Kaggle challenge.  This
dataset is released under CC0, as is the underlying comment text.
"""

_DOWNLOAD_URL = "https://storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments.zip"




class CivilCommentsConfig(datalabs.BuilderConfig):
    """BuilderConfig for CivilComments"""

    def __init__(self,
                 text_column=None,
                 label_column=None,
                 task_templates = None,
                 **kwargs):
        """BuilderConfig for CivilComments.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CivilCommentsConfig, self).__init__(**kwargs)
        self.text_column = text_column
        self.label_column = label_column
        self.task_templates = task_templates


class CivilComments(datalabs.GeneratorBasedBuilder):
    """Classification and tagging of 2M comments on news sites.
    This version of the CivilComments Dataset provides access to the primary
    seven labels that were annotated by crowd workers, the toxicity and other
    tags are a value between 0 and 1 indicating the fraction of annotators that
    assigned these attributes to the comment text.
    The other tags, which are only available for a fraction of the input examples
    are currently ignored, as are all of the attributes that were part of the
    original civil comments release. See the Kaggle documentation for more
    details about the available features.
    """

    VERSION = datalabs.Version("0.9.0")

    BUILDER_CONFIGS = [
        CivilCommentsConfig(name=key,
                        version=datalabs.Version("1.0.0"),
                        description="toxicity classicifation",
                        text_column="text",
                        label_column="label",
                        task_templates = [get_task(TaskType.toxicity_identification)(text_column="text", label_column="label")],
                        )
        for key in ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.Value("float32"),
                }
            ),
            # The supervised_keys version is very impoverished.
            supervised_keys=("text", "toxicity"),
            homepage="https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data",
            citation=_CITATION,
            task_templates=[get_task(TaskType.toxicity_identification)(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_path = dl_manager.download_and_extract(_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filename": os.path.join(dl_path, "train.csv"),
                            },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filename": os.path.join(dl_path, "test_public_expanded.csv"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filename": os.path.join(dl_path, "test_private_expanded.csv"),
                },
            ),
        ]

    def _generate_examples(self, filename):
        """Yields examples.
        Each example contains a text input and then seven annotation labels.
        Args:
          filename: the path of the file to be read for this split.
          toxicity_label: indicates 'target' or 'toxicity' to capture the variation
            in the released labels for this dataset.
        Yields:
          A dictionary of features, all floating point except the input text.
        """
        with open(filename, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # print('row: ',row.keys())
                example = {}
                example["text"] = row["comment_text"]
                label_name = self.config.name
                if self.config.name =="toxicity":
                    label_name = "target"
                    if label_name not in row:
                        label_name = "toxicity"
                        # print('label_name: ',label_name)
                        # print('row: ', row)
                example["label"] = float(row[label_name])
                yield row["id"], example



