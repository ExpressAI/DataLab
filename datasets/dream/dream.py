# coding=utf-8
# Copyright 2020 The TensorFlow datasets Authors and the HuggingFace, DataLab Authors.
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
"""DREAM: A Challenge Dataset and Models for Dialogue-Based Reading Comprehension"""


import json

import datalabs
from datalabs import get_task, TaskType

logger = datalabs.logging.get_logger(__name__)


_CITATION = """\
@article{sundream2018,
  title={{DREAM}: A Challenge Dataset and Models for Dialogue-Based Reading Comprehension},
  author={Sun, Kai and Yu, Dian and Chen, Jianshu and Yu, Dong and Choi, Yejin and Cardie, Claire},
  journal={Transactions of the Association for Computational Linguistics},
  year={2019},
  url={https://arxiv.org/abs/1902.00164v1}
}
"""

_DESCRIPTION = """\
DREAM is a multiple-choice Dialogue-based REAding comprehension exaMination dataset. \
In contrast to existing reading comprehension datasets, DREAM is the first to focus on \
in-depth multi-turn multi-party dialogue understanding.
"""

_URL = "https://raw.githubusercontent.com/nlpdata/dream/master/data/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
    "test": _URL + "test.json",
}


class DreamConfig(datalabs.BuilderConfig):
    """BuilderConfig for Dream."""

    def __init__(self, **kwargs):
        """BuilderConfig for Dream.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DreamConfig, self).__init__(**kwargs)


class Dream(datalabs.GeneratorBasedBuilder):
    """DREAM: A Challenge Dataset and Models for Dialogue-Based Reading Comprehension"""

    VERSION = datalabs.Version("1.0.0")

    BUILDER_CONFIGS = [
        DreamConfig(
            name="plain_text",
            version=datalabs.Version("1.0.0"),
            description="plain_text",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("int32"),
                    "dialogue_id": datalabs.Value("string"),
                    "dialogue": datalabs.Sequence(datalabs.Value("string")),
                    "question": datalabs.Value("string"),
                    "choice": datalabs.features.Sequence(datalabs.Value("string")),
                    # "answer": datalabs.Value("string"),
                    "answers":  # answers -> label
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },
                }
            ),
            supervised_keys=None,
            homepage="https://dataset.org/dream/",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.qa_multiple_choice)(
                    question_column="question",
                    context_column="dialogue",
                    answers_column="answers",
                    options_column="choice",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            dialogues = json.load(f)
            counter = 0
            for dialogue in dialogues:
                dialogue_text = dialogue[0]
                questions = dialogue[1]
                dialogue_id = dialogue[2]

                for que in questions:
                    yield counter, {
                        "id": counter,
                        "dialogue_id": dialogue_id,
                        "dialogue": dialogue_text,
                        "question": que["question"],
                        "choice": que["choice"],
                        # "answer": que["answer"],
                        "answers": {
                            "option_index": -1,  # convert A->0, B->1, C->2, D->3
                            "text": que["answer"],
                        },
                    }
                    counter += 1

