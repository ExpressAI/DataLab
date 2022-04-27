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

import datasets

import datalabs
from datalabs.tasks import QuestionAnsweringMultipleChoicesQASC



# TODO(qasc): BibTeX citation
_CITATION = """\
@article{allenai:qasc,
      author    = {Tushar Khot and Peter Clark and Michal Guerquin and Peter Jansen and Ashish Sabharwal},
      title     = {QASC: A Dataset for Question Answering via Sentence Composition},
      journal   = {arXiv:1910.11473v2},
      year      = {2020},
}
"""

# TODO(qasc):
_DESCRIPTION = """
QASC is a question-answering dataset with a focus on sentence composition. It consists of 9,980 8-way multiple-choice
questions about grade school science (8,134 train, 926 dev, 920 test), and comes with a corpus of 17M sentences.
"""
_URl = "http://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz"


class Qasc(datalabs.GeneratorBasedBuilder):
    """TODO(qasc): Short description of my dataset."""

    # TODO(qasc): Set up version.
    VERSION = datalabs.Version("0.1.0")

    def _info(self):
        # TODO(qasc): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datalabs.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "formatted_question": datalabs.Value("string"),
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "context": {
                        "fact1": datalabs.Value("string"),
                        "fact2": datalabs.Value("string"),
                        "combinedfact": datalabs.Value("string"),
                    },
                    "answers":  # answers -> answer
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://allenai.org/data/qasc",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringMultipleChoicesQASC(
                    question_column="question", context_column="context", answers_column="answers",
                    options_column="options",
                    task="question-answering-multiple-choices-with-context-qasc",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(qasc): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        archive = dl_manager.download(_URl)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": "/".join(["QASC_Dataset", "train.jsonl"]),
                    "files": dl_manager.iter_archive(archive),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": "/".join(["QASC_Dataset", "test.jsonl"]),
                    "files": dl_manager.iter_archive(archive),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": "/".join(["QASC_Dataset", "dev.jsonl"]),
                    "files": dl_manager.iter_archive(archive),
                },
            ),
        ]

    def _generate_examples(self, filepath, files):
        """Yields examples."""
        # TODO(qasc): Yields (key, example) tuples from the dataset
        dict_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10}
        id_sample = 0
        for path, f in files:
            if path == filepath:
                for row in f:
                    id_sample +=1
                    data = json.loads(row.decode("utf-8"))
                    answerkey = data.get("answerKey", "")
                    id_ = data["id"]
                    question = data["question"]["stem"]
                    choices = data["question"]["choices"]
                    text_choices = [choice["text"] for choice in choices]
                    label_choices = [choice["label"] for choice in choices]
                    fact1 = data.get("fact1", "")
                    fact2 = data.get("fact2", "")
                    combined_fact = data.get("combinedfact", "")
                    formatted_question = data.get("formatted_question", "")

                    if answerkey=='':
                        option_index=-1
                        answer_text =''
                    else:
                        option_index = dict_map[answerkey]
                        answer_text = text_choices[option_index]
                    yield id_, {
                        # "id": id_,
                        "id": str(id_sample - 1),
                        "question": question,
                        "formatted_question": formatted_question,
                        "options": text_choices,
                        "context": {
                            "fact1": fact1,
                            "fact2": fact2,
                            "combinedfact": combined_fact,
                        },
                        "answers":
                            {
                                "text": answer_text,
                                "option_index": option_index,
                            },
                    }
                break
