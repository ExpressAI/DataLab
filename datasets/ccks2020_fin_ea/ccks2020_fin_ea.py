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

from ast import arguments
import json
import datalabs
from datalabs import get_task, TaskType
from datalabs.features.features import Sequence

_DESCRIPTION = """\
CCKS2020_fin_event_argument is an event arguments extraction dataset in the financial field, which is used for event extraction tasks. 
The task goal is to extract event arguments and event type based on the given description text.
"""

_CITATION = """\
@misc{
title={CCKS2020金融领域篇章级事件要素抽取数据集},
url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=111211},
author={Tianchi},
year={2021},
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/ccks2020_fin_event_element/train.txt"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/ccks2020_fin_event_element/test.txt"

_HOMEPAGE = "https://www.biendata.xyz/competition/ccks_2020_4_2"


class CCKS2020FinEAConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(CCKS2020FinEAConfig, self).__init__(**kwargs)

class CCKS2020FinEA(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CCKS2020FinEAConfig(
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
                    "event_type": datalabs.Value("string"),
                    "arguments": datalabs.features.Sequence(
                        { 
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
                    text_column = "text",
                    event_column = "arguments",
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as txt_file:
            for id_, line in enumerate(txt_file):
                line = json.loads(line)
                text, events = line["content"], line["events"][0]
                event_type = events["event_type"]
                del events["event_type"]
                del events["event_id"]
                role = []
                entity = []
                for key in events:
                    role.append(key)
                    entity.append(events[key])
                yield id_,{"text": text, "event_type": event_type, "arguments": {"role": role, "entity": entity}}