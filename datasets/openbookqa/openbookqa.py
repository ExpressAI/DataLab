# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and DataLab Authors.
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
import os
import textwrap

import datalabs

from datalabs.tasks import QuestionAnsweringMultipleChoicesWithoutContext



# TODO(openBookQA): BibTeX citation
_CITATION = """\
@inproceedings{OpenBookQA2018,
 title={Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
 author={Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
 booktitle={EMNLP},
 year={2018}
}
"""

# TODO(openBookQA):
_DESCRIPTION = textwrap.dedent(
    """\
OpenBookQA aims to promote research in advanced question-answering, probing a deeper understanding of both the topic
(with salient facts summarized as an open book, also provided with the dataset) and the language it is expressed in. In
particular, it contains questions that require multi-step reasoning, use of additional common and commonsense knowledge,
and rich text comprehension.
OpenBookQA is a new kind of question-answering dataset modeled after open book exams for assessing human understanding of
a subject.
"""
)
_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"


class OpenbookqaConfig(datalabs.BuilderConfig):
    def __init__(self, data_dir, **kwargs):
        """BuilderConfig for openBookQA dataset
        Args:
          data_dir: directory for the given dataset name
          **kwargs: keyword arguments forwarded to super.
        """

        super(OpenbookqaConfig, self).__init__(version=datalabs.Version("1.0.0", ""), **kwargs)

        self.data_dir = data_dir


class Openbookqa(datalabs.GeneratorBasedBuilder):
    """TODO(openBookQA): Short description of my dataset."""

    # TODO(openBookQA): Set up version.
    VERSION = datalabs.Version("0.1.0")
    BUILDER_CONFIGS = [
        OpenbookqaConfig(
            name="main",
            description=textwrap.dedent(
                """
                                  It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test),
                                  which probe the understanding of a small “book” of 1,326 core science facts and the application of these facts to novel
                                  situations. For training, the dataset includes a mapping from each question to the core science fact it was designed to
                                  probe. Answering OpenBookQA questions requires additional broad common knowledge, not contained in the book. The questions,
                                  by design, are answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. Strong neural
                                  baselines achieve around 50% on OpenBookQA, leaving a large gap to the 92% accuracy of crowd-workers.
                                """
            ),
            data_dir="Main",
        ),
        OpenbookqaConfig(
            name="additional",
            description=textwrap.dedent(
                """
                                  Additionally, we provide 5,167 crowd-sourced common knowledge facts, and an expanded version of the train/dev/test questions where
                                  each question is associated with its originating core fact, a human accuracy score, a clarity score, and an anonymized crowd-worker
                                  ID (in the “Additional” folder).
                                """
            ),
            data_dir="Additional",
        ),
    ]

    def _info(self):
        # TODO(openBookQA): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"), # question -> question_stem
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers":  # answers -> answerKey
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },
                    # "options": datalabs.features.Sequence(
                    #     {"text": datalabs.Value("string"), "label": datalabs.Value("string")}
                    # ),
                    # "answerKey": datalabs.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://allenai.org/data/open-book-qa",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringMultipleChoicesWithoutContext(
                    question_column="question", answers_column="answers",
                    options_column="options",
                    task="question-answering-multiple-choices-without-context",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(openBookQA): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "OpenBookQA-V1-Sep2018", "Data")
        data_dir = os.path.join(data_dir, self.config.data_dir)
        train_file = (
            os.path.join(data_dir, "train.jsonl")
            if self.config.name == "main"
            else os.path.join(data_dir, "train_complete.jsonl")
        )
        test_file = (
            os.path.join(data_dir, "test.jsonl")
            if self.config.name == "main"
            else os.path.join(data_dir, "test_complete.jsonl")
        )
        dev_file = (
            os.path.join(data_dir, "dev.jsonl")
            if self.config.name == "main"
            else os.path.join(data_dir, "dev_complete.jsonl")
        )
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": train_file},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": test_file},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": dev_file},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(openBookQA): Yields (key, example) tuples from the dataset
        dict_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4,"F": 5,"G": 6,"H": 7,"I": 8,"J": 9,"K": 10}
        id_sample = 0
        with open(filepath, encoding="utf-8") as f:
            for row in f:
                data = json.loads(row)
                id_sample += 1
                options =[choice["text"] for choice in data["question"]["choices"]]
                option_index = dict_map[data["answerKey"]]
                yield data["id"], {
                    "id": str(id_sample-1),
                    "question": data["question"]["stem"],
                    "options":options,
                    "answers":  # answers -> answerKey
                        {
                            "text": options[option_index],
                            "option_index": option_index,
                        },
                    # "choice": {
                    #     "text": [choice["text"] for choice in data["question"]["choices"]],
                    #     "label": [choice["text"] for choice in data["question"]["choices"]],
                    # },
                }

