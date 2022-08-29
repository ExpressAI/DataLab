# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
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
Given a contention and a candidate sentence, the goal is to determine whether the sentence is an argument supporting the given contention. 
We will also provide debate topics as auxiliary reference. The output has two labels: 1 for arguments, 0 for non-arguments.
"""

_CITATION = """\
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ccac2022_track2/train.txt"
)

_VALIDATION_DOWNLOAD_URL =(
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ccac2022_track2/dev.txt"
)

_TEST_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ccac2022_track2/test.txt"
)


class CCAC2022_track2(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text1": datalabs.Value("string"),
                    "text2": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["1", "0", "-1"]),
                }
            ),
            supervised_keys=None,
            homepage="http://www.fudan-disc.com/sharedtask/AIDebater22/tracks.html",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.argument_discovery)(
                    text1_column="text1", text2_column="text2", label_column="label"
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path=dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
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
                line = line.strip().split('\t')
                yield id_, {"text1": line[0], "text2": line[1], "label": line[2]}
