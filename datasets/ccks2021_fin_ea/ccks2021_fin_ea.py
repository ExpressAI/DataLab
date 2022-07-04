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

from click import argument

import datalabs
from datalabs import get_task, TaskType
from datalabs.features.features import Sequence

_DESCRIPTION = """\
CCKS2021_fin_ea is an event arguments extraction dataset in the financial field, which is used for event extraction tasks. 
The task goal is to extract some of the 13 arguments of an event based on the given description text and text type.
"""

_CITATION = """\
@misc{
title={CCKS2021金融领域篇章级事件元素抽取数据集},
url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=110904},
author={Tianchi},
year={2021},
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/CCKS2021_fin/train_revised.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/CCKS2021_fin/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/CCKS2021_fin/test_revised.json"

_HOMEPAGE = "https://www.biendata.xyz/competition/ccks_2021_task6_1"


class CCKS2021FinEAConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(CCKS2021FinEAConfig, self).__init__(**kwargs)


class CCKS2021FinEA(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CCKS2021FinEAConfig(
            name="event_arguments_extraction",
            version=datalabs.Version("1.0.0"),
            description="event_arguments_extraction",
        ),
    ]

    DEFAULT_CONFIG_NAME = "event_arguments_extraction"

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "event_type": {
                        "level1": datalabs.Value("string"),
                        "level2": datalabs.Value("string"),
                        "level3": datalabs.Value("string"),
                    },
                    "arguments": datalabs.features.Sequence(
                        {
                            "start": datalabs.Value("int32"),
                            "end": datalabs.Value("int32"),
                            "role": datalabs.Value("string"),
                            "entity": datalabs.Value("string"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.event_arguments_extraction)(
                    text_column="text",
                    event_column="arguments",
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
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                line = json.loads(line)
                yield id_, {
                    "text": line["text"],
                    "event_type": line["event_type"],
                    "arguments": line["arguments"],
                }
