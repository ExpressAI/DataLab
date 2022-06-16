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

import datalabs
from datalabs import get_task, TaskType

# TODO(commonsense_qa): BibTeX citation
_CITATION = """\
@InProceedings{commonsense_QA,
title={COMMONSENSEQA: A Question Answering Challenge Targeting Commonsense Knowledge},
author={Alon, Talmor and Jonathan, Herzig and Nicholas, Lourie and Jonathan ,Berant},
journal={arXiv preprint arXiv:1811.00937v2},
year={2019}
"""

# TODO(commonsense_qa):
_DESCRIPTION = """\
CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge
 to predict the correct answers . It contains 12,102 questions with one correct answer and four distractor answers.
 The dataset is provided in two major training/validation/testing set splits: "Random split" which is the main evaluation
  split, and "Question token split", see paper for details.
"""

_URL = "https://s3.amazonaws.com/commensenseqa/"
_URLS = {
    "train": _URL + "train_rand_split.jsonl",
    "dev": _URL + "dev_rand_split.jsonl",
    "test": _URL + "test_rand_split_no_answers.jsonl",
}


class CommonsenseQa(datalabs.GeneratorBasedBuilder):
    """TODO(commonsense_qa): Short description of my dataset."""

    # TODO(commonsense_qa): Set up version.
    VERSION = datalabs.Version("0.1.0")

    def _info(self):
        # These are the features of your dataset like images, labels ...
        features = datalabs.Features(
            {
                "id": datalabs.Value("string"),
                "question": datalabs.Value("string"),  # question -> question_stem
                "options": datalabs.features.Sequence(datalabs.Value("string")),
                "answers": {  # answers -> answerKey
                    "text": datalabs.Value("string"),
                    "option_index": datalabs.Value("int32"),
                },
                # "answerKey": datalabs.Value("string"),
                # "question": datalabs.Value("string"),
                # "choices": datalabs.features.Sequence(
                #     {
                #         "label": datalabs.Value("string"),
                #         "text": datalabs.Value("string"),
                #     }
                # ),
            }
        )
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://www.tau-datasets.org/commonsenseqa",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.qa_multiple_choice_without_context)(
                    question_column="question",
                    answers_column="answers",
                    options_column="options",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        download_urls = _URLS

        downloaded_files = dl_manager.download_and_extract(download_urls)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"], "split": "train"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                    "split": "dev",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        # TODO(commonsense_qa): Yields (key, example) tuples from the dataset
        dict_map = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9,
            "K": 10,
        }
        id_sample = 0
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                id_sample += 1
                question = data["question"]
                choices = question["choices"]
                labels = [label["label"] for label in choices]
                options = [text["text"] for text in choices]
                stem = question["stem"]
                if split == "test":
                    option_index = -1
                    answer_text = ""
                else:
                    option_index = dict_map[data["answerKey"]]
                    answer_text = options[option_index]

                yield id_, {
                    "id": str(id_sample - 1),
                    "question": stem,
                    "options": options,
                    "answers": {  # answers -> answerKey
                        "text": answer_text,
                        "option_index": option_index,
                    },
                }
