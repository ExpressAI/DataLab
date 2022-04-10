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
"""DuoRC: A Paraphrased
Reading Comprehension Question Answering Dataset"""


import json

import datalabs
from datalabs.tasks import QuestionAnsweringExtractive

_CITATION = """\
@inproceedings{DuoRC,
author = { Amrita Saha and Rahul Aralikatte and Mitesh M. Khapra and Karthik Sankaranarayanan},\
title = {{DuoRC: Towards Complex Language Understanding with Paraphrased Reading Comprehension}},
booktitle = {Meeting of the Association for Computational Linguistics (ACL)},
year = {2018}
}
"""


_DESCRIPTION = """\
DuoRC contains 186,089 unique question-answer pairs created from a collection of 7680 pairs of movie plots where each pair in the collection reflects two versions of the same movie.
"""

_HOMEPAGE = "https://duorc.github.io/"

_LICENSE = "https://raw.githubusercontent.com/duorc/duorc/master/LICENSE"

_URL = "https://raw.githubusercontent.com/duorc/duorc/master/dataset/"
_URLs = {
    "SelfRC": {
        "train": _URL + "SelfRC_train.json",
        "dev": _URL + "SelfRC_dev.json",
        "test": _URL + "SelfRC_test.json",
    },
    "ParaphraseRC": {
        "train": _URL + "ParaphraseRC_train.json",
        "dev": _URL + "ParaphraseRC_dev.json",
        "test": _URL + "ParaphraseRC_test.json",
    },
}


class DuorcConfig(datalabs.BuilderConfig):
    """BuilderConfig for DuoRC SelfRC."""

    def __init__(self, **kwargs):
        """BuilderConfig for DuoRC SelfRC.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DuorcConfig, self).__init__(**kwargs)


class Duorc(datalabs.GeneratorBasedBuilder):
    """DuoRC Dataset"""

    VERSION = datalabs.Version("1.0.0")
    BUILDER_CONFIGS = [
        DuorcConfig(name="SelfRC", version=VERSION, description="SelfRC dataset"),
        DuorcConfig(
            name="ParaphraseRC", version=VERSION, description="ParaphraseRC dataset"
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datalabs.Features(
                {
                    "plot_id": datalabs.Value("string"),
                    "context": datalabs.Value("string"),  # context-->plot
                    "title": datalabs.Value("string"),
                    "question_id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    # "answers": datasets.features.Sequence(datasets.Value("string")),
                    "answers": {
                        "text": datalabs.features.Sequence(datalabs.Value("string")),
                        "answer_start": datalabs.features.Sequence(
                            datalabs.Value("int32")
                        ),
                    },
                    "no_answer": datalabs.Value("bool"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        downloaded_files = dl_manager.download_and_extract(my_urls)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as f:
            duorc = json.load(f)
            for example in duorc:
                plot_id = example["id"]
                plot = example["plot"].strip()
                title = example["title"].strip()
                for qas in example["qa"]:
                    question_id = qas["id"]
                    question = qas["question"].strip()
                    answers = [answer.strip() for answer in qas["answers"]]
                    no_answer = qas["no_answer"]

                    yield question_id, {
                        "title": title,
                        "context": plot,  # contxt ->plot
                        "question": question,
                        "plot_id": plot_id,
                        "question_id": question_id,
                        # "answers": answers,
                        "answers": {
                            "answer_start": [-1] * len(answers),
                            "text": answers,
                        },
                        "no_answer": no_answer,
                    }
