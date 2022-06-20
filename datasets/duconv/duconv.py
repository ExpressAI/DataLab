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

from email import message
import json
import datalabs
from datalabs import get_task, TaskType
from datalabs.features import ClassLabel, Features, Sequence, Value
import os
logger = datalabs.logging.get_logger(__name__)

_DESCRIPTION = """\
DuConv is a large-scale proactive human-machine conversation dataset. It contains 270K utterances of 30K dialogues, with one agent proactively leading the conversation based on the knowledge graph.
"""

_CITATION = """\
@article{wu2019proactive,
  title={Proactive human-machine conversation with explicit conversation goals},
  author={Wu, Wenquan and Guo, Zhen and Zhou, Xiangyang and Wu, Hua and Zhang, Xiyuan and Lian, Rongzhong and Wang, Haifeng},
  journal={arXiv preprint arXiv:1906.05572},
  year={2019}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/duconv/train.txt"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/duconv/dev.txt"
_TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/duconv/test.txt"

_HOMEPAGE = "https://ai.baidu.com/broad/subordinate?dataset=duconv"

class DuconvConfig(datalabs.BuilderConfig):

    def __init__(self, **kwargs):

        super(DuconvConfig, self).__init__(**kwargs)

class Duconv(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DuconvConfig(
            name="train",
            version=datalabs.Version("1.0.0"),
            description="train set",
        ),
        DuconvConfig(
            name="dev",
            version=datalabs.Version("1.0.0"),
            description="dev set",
        ),
        DuconvConfig(
            name="test",
            version=datalabs.Version("1.0.0"),
            description="test set",

        ),
    ]

    def _info(self):
        features = {feature: Sequence(Value("string")) for feature in ["goal", "knowledge", "content"]}

        if self.config.name=='test':
            features['response'] = Value("string")
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features({
                "goal": Sequence(Value("string")),
                "knowledge": Sequence(Value("string")),
                "content":Sequence(Value("string")),
                'response': Value("string")

            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.goal_oriented_knowledge_driven_dialogue)(
                    knowledge_column = "knowledge",
                    goal_column = "goal",
                    content_column = "content",
                    response_column = "response",

                )
            ],
        )


    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
                datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
                datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
                datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
            ]




    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:

            for id_, line in enumerate(f):
                conv=json.loads(line)

                if 'response' not in conv:
                    yield id_,{
                        "goal":conv['goal'],
                        "knowledge":  conv["knowledge"],
                        "content": conv["conversation"],
                        "response": None,
                    }
                else:

                    yield id_,{
                        "goal":conv['goal'],
                        "knowledge":  conv["knowledge"],
                        "content": conv["history"],
                        'response': conv["response"],
                    }
