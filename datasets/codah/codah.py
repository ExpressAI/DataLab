# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and DataLab Authors.
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
"""The COmmonsense Dataset Adversarially-authored by Humans (CODAH)"""

import csv

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{chen2019codah,
  title={CODAH: An Adversarially-Authored Question Answering Dataset for Common Sense},
  author={Chen, Michael and D'Arcy, Mike and Liu, Alisa and Fernandez, Jared and Downey, Doug},
  booktitle={Proceedings of the 3rd Workshop on Evaluating Vector Space Representations for NLP},
  pages={63--69},
  year={2019}
}
"""

_DESCRIPTION = """\
The COmmonsense Dataset Adversarially-authored by Humans (CODAH) is an evaluation set for commonsense \
question-answering in the sentence completion style of SWAG. As opposed to other automatically \
generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback \
from a pre-trained model and use this information to design challenging commonsense questions. \
Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense.
"""

_URL = "https://raw.githubusercontent.com/Websail-NU/CODAH/master/data/"
_FULL_DATA_URL = _URL + "full_data.tsv"

QUESTION_CATEGORIES_MAPPING = {
    "i": "Idioms",
    "r": "Reference",
    "p": "Polysemy",
    "n": "Negation",
    "q": "Quantitative",
    "o": "Others",
}


class CodahConfig(datalabs.BuilderConfig):
    """BuilderConfig for CODAH."""

    def __init__(self, fold=None, **kwargs):
        """BuilderConfig for CODAH.
        Args:
          fold: `string`, official cross validation fold.
          **kwargs: keyword arguments forwarded to super.
        """
        super(CodahConfig, self).__init__(**kwargs)
        self.fold = fold


class Codah(datalabs.GeneratorBasedBuilder):
    """The COmmonsense Dataset Adversarially-authored by Humans (CODAH)"""

    VERSION = datalabs.Version("1.0.0")
    BUILDER_CONFIGS = [
        CodahConfig(
            name="codah",
            version=datalabs.Version("1.0.0"),
            description="Full CODAH dataset",
            fold=None,
        ),
        CodahConfig(
            name="fold_0",
            version=datalabs.Version("1.0.0"),
            description="Official CV split (fold_0)",
            fold="fold_0",
        ),
        CodahConfig(
            name="fold_1",
            version=datalabs.Version("1.0.0"),
            description="Official CV split (fold_1)",
            fold="fold_1",
        ),
        CodahConfig(
            name="fold_2",
            version=datalabs.Version("1.0.0"),
            description="Official CV split (fold_2)",
            fold="fold_2",
        ),
        CodahConfig(
            name="fold_3",
            version=datalabs.Version("1.0.0"),
            description="Official CV split (fold_3)",
            fold="fold_3",
        ),
        CodahConfig(
            name="fold_4",
            version=datalabs.Version("1.0.0"),
            description="Official CV split (fold_4)",
            fold="fold_4",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question_category": datalabs.Value("string"),
                    "question": datalabs.Value("string"),  # question_prompt->question
                    "options": datalabs.features.Sequence(
                        datalabs.Value("string")
                    ),  # candidate_answers -> options
                    "answers": {  # answers -> answerKey
                        "text": datalabs.Value("string"),
                        "option_index": datalabs.Value("int32"),
                    },
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/Websail-NU/CODAH",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.qa_multiple_choice)(
                    question_column="question",
                    answers_column="answers",
                    options_column="options",
                )
            ],
            languages=["en"],
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "codah":
            data_file = dl_manager.download(_FULL_DATA_URL)
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TRAIN, gen_kwargs={"data_file": data_file}
                )
            ]

        base_url = f"{_URL}cv_split/{self.config.fold}/"
        _urls = {
            "train": base_url + "train.tsv",
            "dev": base_url + "dev.tsv",
            "test": base_url + "test.tsv",
        }
        downloaded_files = dl_manager.download_and_extract(_urls)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"data_file": downloaded_files["train"]},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"data_file": downloaded_files["dev"]},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"data_file": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, data_file):
        question_category_list = [
            "Idioms",
            "Reference",
            "Polysemy",
            "Negation",
            "Quantitative",
            "Others",
        ]
        with open(data_file, encoding="utf-8") as f:
            rows = csv.reader(f, delimiter="\t")
            for i, row in enumerate(rows):
                question_category = (
                    QUESTION_CATEGORIES_MAPPING[row[0]] if row[0] != "" else -1
                )

                options = row[2:-1]
                option_index = int(row[-1])
                yield i, {
                    "id": str(i),
                    "question_category": question_category,
                    "question": row[1],
                    "options": options,
                    "answers": {  # correct_answer_idx -> answers
                        "text": options[option_index],
                        "option_index": option_index,
                    },
                    # "correct_answer_idx": int(row[-1]),
                }
