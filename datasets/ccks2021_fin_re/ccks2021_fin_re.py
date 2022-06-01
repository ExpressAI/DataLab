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
CCKS2021_fin_relation_extraction is an event relation extraction dataset in the financial field, which is used for event extraction tasks. 
The task goal is to extract the causality of events based on the given description text.
"""

_CITATION = """\
@misc{
title={CCKS2021金融领域事件因果关系抽取数据集},
url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=110901},
author={Tianchi},
year={2021},
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/ccks2021_fin_re/train.txt"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/ccks2021_fin_re/test.txt"

_HOMEPAGE = "https://www.biendata.xyz/competition/ccks_2021_task6_2"


class CCKS2021FinREConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(CCKS2021FinREConfig, self).__init__(**kwargs)

class CCKS2021FinRE(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CCKS2021FinREConfig(
            name="event_relation_extraction",
            version=datalabs.Version("1.0.0"),
            description="event_relation_extraction",
        ),
    ]

    DEFAULT_CONFIG_NAME = "event_relation_extraction"

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "relation": {
                        "reason_type": datalabs.features.Sequence(datalabs.Value("string")),
                        "reason_region": datalabs.features.Sequence(datalabs.Value("string")),
                        "reason_industry": datalabs.features.Sequence(datalabs.Value("string")),
                        "reason_product": datalabs.features.Sequence(datalabs.Value("string")),
                        "result_type": datalabs.features.Sequence(datalabs.Value("string")),
                        "result_region": datalabs.features.Sequence(datalabs.Value("string")),
                        "result_industry": datalabs.features.Sequence(datalabs.Value("string")),
                        "result_product": datalabs.features.Sequence(datalabs.Value("string")),
                    }
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.event_relation_extraction)(
                    text_column = "text",
                    event_column = "relation",
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
                text, result = line["text"], line["result"][0]
                relation = {"reason_type":[], "reason_region":[], "reason_industry":[], "reason_product":[], 
                            "result_type":[], "result_region":[], "result_industry":[], "result_product":[]}
                for key in result:
                    relation[key].extend(result[key].split(","))
                yield id_, {'text': text, 'relation': relation}