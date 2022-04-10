# coding=utf-8
# Copyright 2020 The HuggingFace datasets Authors and DataLab Authors.
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

"""TODO(quoref): Add a description here."""

# import sys
# sys.path.append("..")
import json
import os

import datalabs
from datalabs.tasks import QuestionAnsweringExtractive

# TODO(quoref): BibTeX citation
_CITATION = """\
@article{allenai:quoref,
      author    = {Pradeep Dasigi and Nelson F. Liu and Ana Marasovic and Noah A. Smith and  Matt Gardner},
      title     = {Quoref: A Reading Comprehension Dataset with Questions Requiring Coreferential Reasoning},
      journal   = {arXiv:1908.05803v2 },
      year      = {2019},
}
"""

# TODO(quoref):
_DESCRIPTION = """\
Quoref is a QA dataset which tests the coreferential reasoning capability of reading comprehension systems. In this
span-selection benchmark containing 24K questions over 4.7K paragraphs from Wikipedia, a system must resolve hard
coreferences before selecting the appropriate span(s) in the paragraphs for answering questions.
"""

_URL = "https://quoref-dataset.s3-us-west-2.amazonaws.com/train_and_dev/quoref-train-dev-v0.1.zip"


class Quoref(datalabs.GeneratorBasedBuilder):
    """TODO(quoref): Short description of my dataset."""

    # TODO(quoref): Set up version.
    VERSION = datalabs.Version("0.1.0")

    def _info(self):
        # TODO(quoref): Specifies the datalabs.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datalabs page.
            description=_DESCRIPTION,
            # datalabs.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                    "url": datalabs.Value("string"),
                    "answers": {
                        "text": datalabs.features.Sequence(datalabs.Value("string")),
                        "answer_start": datalabs.features.Sequence(
                            datalabs.Value("int32")
                        ),
                    }
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://leaderboard.allenai.org/quoref/submissions/get-started",
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
        # TODO(quoref): Downloads the data and defines the splits
        # dl_manager is a datalabs.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "quoref-train-dev-v0.1")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "quoref-train-v0.1.json")
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "quoref-dev-v0.1.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(quoref): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data["data"]:
                title = article.get("title", "").strip()
                url = article.get("url", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [
                            answer["answer_start"] for answer in qa["answers"]
                        ]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                            "url": url,
                        }
