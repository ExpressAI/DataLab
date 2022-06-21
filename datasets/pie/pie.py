# coding=utf-8
# Copyright 2022 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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
import json

import datalabs
from datalabs import get_task, TaskType
import os

logger = datalabs.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{awasthi-etal-2019-parallel,
    title = "Parallel Iterative Edit Models for Local Sequence Transduction",
    author = "Awasthi, Abhijeet  and
      Sarawagi, Sunita  and
      Goyal, Rasna  and
      Ghosh, Sabyasachi  and
      Piratla, Vihari",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1435",
    doi = "10.18653/v1/D19-1435",
    pages = "4259--4269",
}
"""

_DESCRIPTION = """Parallel Iterative Edit
"""

_URLS = "https://datalab-hub.s3.amazonaws.com/grammatical_error_correction/pie/pie.zip"


class PieConfig(datalabs.BuilderConfig):
    """BuilderConfig for pie."""

    def __init__(self, **kwargs):
        """BuilderConfig for pie.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PieConfig, self).__init__(**kwargs)


class Pie(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PieConfig(
            name="plain_text",
            version=datalabs.Version("0.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "original": datalabs.Value("string"),
                    "correct":
                        {
                            "m2": datalabs.features.Sequence(datalabs.Value("string")),
                            "sentence": datalabs.Value("string"),
                        }

                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/awasthiabhijeet/PIE",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.grammatical_error_correction)(

                )
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={
                "filepath": os.path.join(downloaded_files, f"pie/train.txt")}),

        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line_list=line.strip().split('\t')
                yield key, {
                    "original": line_list[0],
                    "correct": {
                        'm2': None,
                        'sentence': line_list[1]
                    }
                }
                key += 1
