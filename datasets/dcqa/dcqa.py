# coding=utf-8
# Copyright 2022 The DataLab Authors.
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
"""SQUAD: The Stanford Question Answering Dataset."""


import json

import datalabs
from datalabs.tasks import QuestionAnsweringExtractive


logger = datalabs.logging.get_logger(__name__)


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

# _URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
_URLS = {
    "train": "https://drive.google.com/uc?export=download&id=1uKKMg9KItYR8jSZFUwqz7UIwy8pHyr2W",
    "dev": "https://drive.google.com/uc?export=download&id=1JX8pdQJaDqwzK7fzNs9mM9UY09be29ci",
    "test": "https://drive.google.com/uc?export=download&id=1lDWCWzArAM0tJ5EJdx-nJtHtp-29869E"
}


class DcqaConfig(datalabs.BuilderConfig):
    """BuilderConfig for DCQA."""

    def __init__(self, **kwargs):
        """BuilderConfig for DCQA.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DcqaConfig, self).__init__(**kwargs)


class Dcqa(datalabs.GeneratorBasedBuilder):
    """Dcqa: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        DcqaConfig(
            name="plain_text",
            version=datalabs.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "context": {
                        "SentenceID": datalabs.features.Sequence(datalabs.Value("int32")),
                        "text": datalabs.features.Sequence(datalabs.Value("string"))
                    },
                    "answer": {
                        "SentenceID": datalabs.Value("int32"),
                        "text": datalabs.Value("string")
                    },
                    "AnchorSentenceID": datalabs.Value("int32"),
                    "QuestionID": datalabs.Value("string")
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answer"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        id_sample = 0
        with open(filepath, encoding="utf-8") as f:
            dcqa = json.load(f)
            for data in dcqa:
                question = data["question"]
                context = data["context"]
                answer = data["answer"]
                AnchorSentenceID = data["AnchorSentenceID"]
                QuestionID = data["QuestionID"]

                yield key, {
                    "id": str(id_sample+1),
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "AnchorSentenceID": AnchorSentenceID,
                    "QuestionID": QuestionID,
                }
                key +=1
