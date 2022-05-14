# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and DataLab Authors.
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
import datalabs
from datalabs import get_task, TaskType



# TODO(social_i_qa): BibTeX citation
_CITATION = """
"""

# TODO(social_i_qa):
_DESCRIPTION = """\
We introduce Social IQa: Social Interaction QA, a new question-answering benchmark for testing social commonsense intelligence. Contrary to many prior benchmarks that focus on physical or taxonomic knowledge, Social IQa focuses on reasoning about people’s actions and their social implications. For example, given an action like "Jesse saw a concert" and a question like "Why did Jesse do this?", humans can easily infer that Jesse wanted "to see their favorite performer" or "to enjoy the music", and not "to see what's happening inside" or "to see if it works". The actions in Social IQa span a wide variety of social situations, and answer candidates contain both human-curated answers and adversarially-filtered machine-generated candidates. Social IQa contains over 37,000 QA pairs for evaluating models’ abilities to reason about the social implications of everyday events and situations. (Less)
"""
_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"


class SocialIQa(datalabs.GeneratorBasedBuilder):
    """TODO(social_i_qa): Short description of my dataset."""

    # TODO(social_i_qa): Set up version.
    VERSION = datalabs.Version("0.1.0")

    def _info(self):
        # TODO(social_i_qa): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "context": datalabs.Value("string"),  # context ->article
                    "question": datalabs.Value("string"),
                    "answers":  # answers -> label
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },
                    "options": datalabs.features.Sequence(datalabs.Value("string"))
                }

                # {
                #     # These are the features of your dataset like images, labels ...
                #     "context": datalabs.Value("string"),
                #     "question": datalabs.Value("string"),
                #     "answerA": datalabs.Value("string"),
                #     "answerB": datalabs.Value("string"),
                #     "answerC": datalabs.Value("string"),
                #     "label": datalabs.Value("string"),
                # }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://leaderboard.allenai.org/socialiqa/submissions/get-started",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.qa_multiple_choice)(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                    options_column="options",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(social_i_qa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        dl_dir = os.path.join(dl_dir, "socialiqa-train-dev")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "train.jsonl"),
                    "labelpath": os.path.join(dl_dir, "train-labels.lst"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "dev.jsonl"),
                    "labelpath": os.path.join(dl_dir, "dev-labels.lst"),
                },
            ),
        ]

    def _generate_examples(self, filepath, labelpath):
        """Yields examples."""
        # TODO(social_i_qa): Yields (key, example) tuples from the dataset
        with open(labelpath, encoding="utf-8") as f:
            labels = [label.strip() for label in f]

        dict_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10}
        id_sample = 0
        with open(filepath, encoding="utf-8") as f1:
            for id_, row in enumerate(f1):
                id_sample += 1
                data = json.loads(row)
                label = labels[id_]
                context = data["context"]
                answerA = data["answerA"]
                answerB = data["answerB"]
                answerC = data["answerC"]
                options = [answerA,answerB,answerC]
                question = data["question"]
                option_index = int(label)-1
                yield id_, {
                    "id": str(id_sample - 1),
                    "context": context,
                    "question": question,
                    "options": options,
                    "answers": {
                                "option_index": option_index,  # convert A->0, B->1, C->2, D->3
                                "text": options[option_index],
                            },
                }
