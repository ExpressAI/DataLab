# coding=utf-8
# Copyright 2020 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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

import json

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
平安医疗科技疾病问答迁移学习比赛数据集 (PingAn Medical Technology Disease Question Transfer Learning Competition Dataset)
Specifically, given pairs of question sentences from five different diseases, 
the system is required to determine whether the semantics of the two questions are the same or similar.
"""

_CITATION = """\
For more information, please refer to http://www.cips-chip.org.cn:8000/home.
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CHIP2019_QM/train_revised.json"
)
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CHIP2019_QM/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CHIP2019_QM/test_revised.json"

_HOMEPAGE = "http://www.cips-chip.org.cn:8000/home"


class CHIP2019QMConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(CHIP2019QMConfig, self).__init__(**kwargs)


class CHIP2019QM(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CHIP2019QMConfig(
            name="question_migration",
            version=datalabs.Version("1.0.0"),
            description="question_migration",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "question1": datalabs.Value("string"),
                    "question2": datalabs.Value("string"),
                    "category": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["0", "1"]),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.text_pair_classification)(
                    text1_column="question1",
                    text2_column="question2",
                    label_column="label",
                )
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                line = json.loads(line)
                yield id_, {
                    "question1": line["question1"],
                    "question2": line["question2"],
                    "category": line["category"],
                    "label": line["label"],
                }
