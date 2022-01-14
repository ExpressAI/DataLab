# coding=utf-8
# Copyright 2020 The TensorFlow datalab Authors and the HuggingFace datalab Authors.
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

_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
_URLS = {
    "train": _URL + "train-v1.1.json",
    "dev": _URL + "dev-v1.1.json",
}


class SquadConfig(datalabs.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadConfig, self).__init__(**kwargs)


class Squad(datalabs.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        SquadConfig(
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
                    "title": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    # "answers": datalabs.features.Sequence(
                    #     {
                    #         "text": datalabs.Value("string"),
                    #         "answer_start": datalabs.Value("int32"),
                    #     }
                    # ),
                    "answers":
                        {
                            "text": datalabs.features.Sequence(datalabs.Value("string")),
                            "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                        }

                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]

                        # answer_list = [
                        #     {"text": "abc", "answer_start": -1},
                        #     {"text": "def", "answer_start": -2}
                        # ]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                            # "answers":answer_list,
                        }
                        key += 1
