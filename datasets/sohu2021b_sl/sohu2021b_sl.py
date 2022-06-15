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
2021搜狐校园文本匹配算法大赛
The matching standard of data set B is more strict.
Two paragraphs of text must be the same event to be considered to be matched. 
This data set contains short-long sentence pairs. 
For more information, please refer to https://www.biendata.xyz/competition/sohu_2021/data. 
"""

_CITATION = """\
For more information, please refer to https://www.biendata.xyz/competition/sohu_2021/data. 
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/sohu2021B/short-long/train.txt"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/sohu2021B/short-long/valid.txt"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/sohu2021B/short-long/test_with_id.txt"


class SOHU2021B_SL(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text1": datalabs.Value("string"),
                    "text2": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["0", "1"]),
                }
            ),
            supervised_keys=None,
            homepage="https://www.biendata.xyz/competition/sohu_2021/data",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.paraphrase_identification)(
                    text1_column="text1", text2_column="text2", label_column="label"
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                text1, text2, label = line["source"], line["target"], line["labelB"]
                if label == ("0" or "1"):
                    yield id_, {"text1": text1, "text2": text2, "label": label}
