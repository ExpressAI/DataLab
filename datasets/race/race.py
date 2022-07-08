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

_CITATION = """\
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""

_DESCRIPTION = """\
Race is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The
 dataset is collected from English examinations in China, which are designed for middle school and high school students.
The dataset can be served as the training and test sets for machine comprehension.
"""

_URL = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"


class Race(datalabs.GeneratorBasedBuilder):
    """ReAding Comprehension Dataset From Examination dataset from CMU"""

    VERSION = datalabs.Version("0.1.0")

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="high",
            description="Exams designed for high school students",
            version=VERSION,
        ),
        datalabs.BuilderConfig(
            name="middle",
            description="Exams designed for middle school students",
            version=VERSION,
        ),
        datalabs.BuilderConfig(
            name="all",
            description="Exams designed for both high school and middle school students",
            version=VERSION,
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "document_id": datalabs.Value("string"),
                    "context": datalabs.Value("string"),  # context ->article
                    "question": datalabs.Value("string"),
                    "answers": {  # answers -> answer
                        "text": datalabs.Value("string"),
                        "option_index": datalabs.Value("int32"),
                    },
                    "options": datalabs.features.Sequence(datalabs.Value("string"))
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="http://www.cs.cmu.edu/~glai1/data/race/",
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
        # Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        archive = dl_manager.download(_URL)
        case = str(self.config.name)
        if case == "all":
            case = ""
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "train_test_or_eval": f"RACE/test/{case}",
                    "files": dl_manager.iter_archive(archive),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "train_test_or_eval": f"RACE/train/{case}",
                    "files": dl_manager.iter_archive(archive),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "train_test_or_eval": f"RACE/dev/{case}",
                    "files": dl_manager.iter_archive(archive),
                },
            ),
        ]

    def _generate_examples(self, train_test_or_eval, files):
        """Yields examples."""
        dict_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        id_sample = 0
        for file_idx, (path, f) in enumerate(files):
            if path.startswith(train_test_or_eval) and path.endswith(".txt"):
                data = json.loads(f.read().decode("utf-8"))
                questions = data["questions"]
                answers = data["answers"]
                options = data["options"]
                for i in range(len(questions)):
                    question = questions[i]
                    answer = answers[i]
                    option = options[i]
                    option_index = dict_map[answer]
                    id_sample += 1
                    yield f"{file_idx}_{i}", {
                        "id": str(id_sample - 1),
                        "document_id": data["id"],
                        "context": data["article"],
                        "question": question,
                        "answers": {
                            "option_index": option_index,  # convert A->0, B->1, C->2, D->3
                            "text": option[option_index],
                        },
                        # "answers": answer,
                        "options": option,
                    }
