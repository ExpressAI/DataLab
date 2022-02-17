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
"""PIQA dataset."""


import json
import os


import datalabs
from datalabs.tasks import QuestionAnsweringMultipleChoicesWithoutContext


_CITATION = """\
@inproceedings{Bisk2020,
  author = {Yonatan Bisk and Rowan Zellers and
            Ronan Le Bras and Jianfeng Gao
            and Yejin Choi},
  title = {PIQA: Reasoning about Physical Commonsense in
           Natural Language},
  booktitle = {Thirty-Fourth AAAI Conference on
               Artificial Intelligence},
  year = {2020},
}
"""

_DESCRIPTION = """\
To apply eyeshadow without a brush, should I use a cotton swab or a toothpick?
Questions requiring this kind of physical commonsense pose a challenge to state-of-the-art
natural language understanding systems. The PIQA dataset introduces the task of physical commonsense reasoning
and a corresponding benchmark dataset Physical Interaction: Question Answering or PIQA.
Physical commonsense knowledge is a major challenge on the road to true AI-completeness,
including robots that interact with the world and understand natural language.
PIQA focuses on everyday situations with a preference for atypical solutions.
The dataset is inspired by instructables.com, which provides users with instructions on how to build, craft,
bake, or manipulate objects using everyday materials.
The underlying task is formualted as multiple choice question answering:
given a question `q` and two possible solutions `s1`, `s2`, a model or
a human must choose the most appropriate solution, of which exactly one is correct.
The dataset is further cleaned of basic artifacts using the AFLite algorithm which is an improvement of
adversarial filtering. The dataset contains 16,000 examples for training, 2,000 for development and 3,000 for testing.
"""

_URLs = {
    "train-dev": "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
    "test": "https://yonatanbisk.com/piqa/data/tests.jsonl",
}


class Piqa(datalabs.GeneratorBasedBuilder):
    """PIQA dataset."""

    VERSION = datalabs.Version("1.1.0")

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=VERSION,
        )
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"), # question -> goal
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers":  # answers -> answer
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },
                }
            ),
            supervised_keys=None,
            homepage="https://yonatanbisk.com/piqa/",
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
        data_dir = dl_manager.download_and_extract(_URLs)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "input_filepath": os.path.join(data_dir["train-dev"], "physicaliqa-train-dev", "train.jsonl"),
                    "label_filepath": os.path.join(data_dir["train-dev"], "physicaliqa-train-dev", "train-labels.lst"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "input_filepath": data_dir["test"],
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "input_filepath": os.path.join(data_dir["train-dev"], "physicaliqa-train-dev", "dev.jsonl"),
                    "label_filepath": os.path.join(data_dir["train-dev"], "physicaliqa-train-dev", "dev-labels.lst"),
                },
            ),
        ]

    def _generate_examples(self, input_filepath, label_filepath=None):
        """Yields examples."""
        id_sample = 0
        with open(input_filepath, encoding="utf-8") as input_file:
            inputs = input_file.read().splitlines()

            if label_filepath is not None:
                with open(label_filepath, encoding="utf-8") as label_file:
                    labels = label_file.read().splitlines()
            else:
                # Labels are not available for the test set.
                # Filling the `label` column with -1 by default
                labels = [-1] * len(inputs)

            for idx, (row, lab) in enumerate(zip(inputs, labels)):
                id_sample +=1
                data = json.loads(row)
                goal = data["goal"]
                sol1 = data["sol1"]
                sol2 = data["sol2"]

                answer_text = ''
                if lab in [0, 1]:
                    new_lab = str(int(lab)+1)
                    answer_text = data['sol' + new_lab]

                yield idx, {
                    "id": str(id_sample-1),
                    "question": goal,  # question -> goal
                    "options": [sol1, sol2],
                    "answers":  # answers -> answer
                        {
                            "text": answer_text,
                            "option_index": int(lab),
                        },
                }

                # yield idx, {
                #     "goal": goal,
                #     "sol1": sol1,
                #     "sol2": sol2,
                #     "label": lab
                # }