# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and DataLab Authors.
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
"""WikiHop: Reading Comprehension with Multiple Hops"""


import json
import os

import datalabs
from datalabs.tasks import QuestionAnsweringExtractive



_CITATION = """\
@misc{welbl2018constructing,
      title={Constructing Datasets for Multi-hop Reading Comprehension Across Documents},
      author={Johannes Welbl and Pontus Stenetorp and Sebastian Riedel},
      year={2018},
      eprint={1710.06481},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
WikiHop is open-domain and based on Wikipedia articles; the goal is to recover Wikidata information by hopping through documents. \
The goal is to answer text understanding queries by combining multiple facts that are spread across different documents.
"""

_URL = "https://drive.google.com/uc?export=download&id=1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA"


class WikiHopConfig(datalabs.BuilderConfig):
    """BuilderConfig for WikiHop."""

    def __init__(self, masked=False, **kwargs):
        """BuilderConfig for WikiHop.
        Args:
          masked: `bool`, original or maksed data.
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiHopConfig, self).__init__(**kwargs)
        self.masked = masked


class WikiHop(datalabs.GeneratorBasedBuilder):
    """WikiHop: Reading Comprehension with Multiple Hops"""

    VERSION = datalabs.Version("1.0.0")
    BUILDER_CONFIGS = [
        WikiHopConfig(
            name="original",
            version=datalabs.Version("1.0.0"),
            description="The un-maksed WikiHop dataset",
            masked=False,
        ),
        WikiHopConfig(
            name="masked", version=datalabs.Version("1.0.0"), description="Masked WikiHop dataset", masked=True
        ),
    ]
    BUILDER_CONFIG_CLASS = WikiHopConfig
    DEFAULT_CONFIG_NAME = "original"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    # "answer": datalabs.Value("string"),
                    "answers": # answers -> answer
                        {
                            "text": datalabs.features.Sequence(datalabs.Value("string")),
                            "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                        },
                    "candidates": datalabs.Sequence(datalabs.Value("string")),
                    # "supports": datalabs.Sequence(datalabs.Value("string")), # context->supports
                    "context": datalabs.Value("string"),
                    "annotations": datalabs.Sequence(datalabs.Sequence(datalabs.Value("string"))),
                }
            ),
            supervised_keys=None,
            homepage="http://qangaroo.cs.ucl.ac.uk/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_URL)

        wikihop_path = os.path.join(extracted_path, "qangaroo_v1.1", "wikihop")
        train_file = "train.json" if self.config.name == "original" else "train.masked.json"
        dev_file = "dev.json" if self.config.name == "original" else "dev.masked.json"

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(wikihop_path, train_file), "split": "train"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(wikihop_path, dev_file), "split": "dev"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            examples = json.load(f)
            for i, example in enumerate(examples):
                # there are no annotations for train split, setting it to empty list
                if split == "train":
                    example["annotations"] = []
                example["question"] = example.pop("query")

                answers = [example["answer"].strip()]
                context = ' '.join(example["supports"])
                # yield example["id"], example
                yield example["id"], {
                    "id": example["id"],
                    "question": example["question"],
                    "answers": {
                        "answer_start": [-1]*len(answers),
                        "text": answers,
                    },
                    "candidates": example["annotations"],
                    # "supports": datalabs.Sequence(datalabs.Value("string")), # context->supports
                    "context": context,
                    "annotations": example["annotations"],
                }





