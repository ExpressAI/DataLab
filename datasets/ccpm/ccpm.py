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
Chinese Classical Poetry Matching Dataset: 
Given the modern translation of Chinese classical poetry, 
the goal is to select the poem that matches the semantics of the modern translation text best 
from the four candidates in the given options.
CCPM dataset has a training set containing 21,778 sentences, a validation set containing 2,720 sentences,
and a test set containing 2,720 sentences without labels. 
"""

_CITATION = """\
@article{li2021CCPM,
  title = {CCPM: A Chinese Classical Poetry Matching Dataset},
  author = {Li, Wenhao and Qi, Fanchao and Sun, Maosong and Yi, Xiaoyuan and Zhang, Jiarui},
  journal={arXiv preprint arXiv:2106.01979},
  year = {2021}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CCPM/train_revised.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CCPM/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/CCPM/test_revised.json"

_HOMEPAGE = "https://github.com/THUNLP-AIPoet/CCPM"


class CCPMConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(CCPMConfig, self).__init__(**kwargs)


class CCPM(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CCPMConfig(
            name="classical_poetry_matching",
            version=datalabs.Version("1.0.0"),
            description="classical_poetry_matching",
        ),
    ]

    DEFAULT_CONFIG_NAME = "classical_poetry_matching"

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "question": datalabs.Value("string"),
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers": {
                        "text": datalabs.Value("string"),
                        "option_index": datalabs.Value("int32"),
                    }
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.qa_multiple_choice_without_context)(
                    question_column = "question",
                    options_column = "options",
                    answers_column = "answers",
                ),
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
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as file:
            for id_, line in enumerate(file):
                line = json.loads(line)
                yield id_, {"question": line["question"], "options": line["options"], "answers": line["answers"]}
