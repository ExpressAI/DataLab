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
from datalabs.features.features import Sequence

_DESCRIPTION = """\
DuEE1.0 is a Chinese event extraction dataset released by Baidu, 
which contains 17,000 sentences (20,000 events) with event information in 65 event types.
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=6. 
"""

_CITATION = """\
@InProceedings{10.1007/978-3-030-60457-8_44,
    author="Li, Xinyu
        and Li, Fayuan
        and Pan, Lu
        and Chen, Yuguang
        and Peng, Weihua
        and Wang, Quan
        and Lyu, Yajuan
        and Zhu, Yong",
    editor="Zhu, Xiaodan
        and Zhang, Min
        and Hong, Yu
        and He, Ruifang",
    title="DuEE: A Large-Scale Dataset for Chinese Event Extraction in Real-World Scenarios",
    booktitle="Natural Language Processing and Chinese Computing",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="534--545",
    abstract="This paper introduces DuEE, a new dataset for Chinese event extraction (EE) in real-world scenarios. DuEE has several advantages over previous EE datasets. (1) Scale: DuEE consists of 19,640 events categorized into 65 event types, along with 41,520 event arguments mapped to 121 argument roles, which, to our knowledge, is the largest Chinese EE dataset so far. (2) Quality: All the data is human annotated with crowdsourced review, ensuring that the annotation accuracy is higher than 95{\%}. (3) Reality: The schema covers trending topics from Baidu Search and the data is collected from news on Baijiahao. The task is also close to real-world scenarios, e.g., a single instance is allowed to contain multiple events, different event arguments are allowed to share the same argument role, and an argument is allowed to play different roles. To advance the research on Chinese EE, we release DuEE as well as a baseline system to the community. We also organize a shared competition on the basis of DuEE, which has attracted 1,206 participants. We analyze the results of top performing systems and hope to shed light on further improvements.",
    isbn="978-3-030-60457-8"
}
"""

_LICENSE = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuEE1.0/License.docx"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuEE1.0/duee_train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuEE1.0/duee_dev.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuEE1.0/duee_sample.json"

_HOMEPAGE = "https://www.luge.ai/#/luge/dataDetail?id=6"


class DuEEConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(DuEEConfig, self).__init__(**kwargs)

class DuEE(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DuEEConfig(
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
                    "trigger": datalabs.Value("string"),
                    "trigger_start_index": datalabs.Value("string"),
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
                    text_column = "text",
                    event_column = "arguments",
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        valid_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": valid_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        count = 1
        with open(filepath, encoding="utf-8") as txt_file:
            for id_, line in enumerate(txt_file):
                line = json.loads(line)
                text, event_list = line["text"], line["event_list"]
                for event in event_list:
                    event_type = event["event_type"]
                    trigger = event["trigger"]
                    trigger_start_index = event["trigger_start_index"]
                    argument_list = event["arguments"]
                    arguments = []
                    for a in argument_list:
                        start = int(a["argument_start_index"])
                        role = a["role"]
                        entity = a["argument"]
                        end = start + len(entity) - 1
                        argument = {"start": start, "end": end, "role": role, "entity": entity}
                        arguments.append(argument)
                    yield count, {"text": text, "event_type": event_type, "trigger": trigger, "trigger_start_index": trigger_start_index, "arguments": arguments}
                    count = count + 1
