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
import os
logger = datalabs.logging.get_logger(__name__)

_DESCRIPTION = """\
KdConv is a Chinese multi-domain Knowledge-driven Conversionsation dataset, grounding the topics in multi-turn conversations to knowledge graphs. KdConv contains 4.5K conversations from three domains (film, music, and travel), and 86K utterances with an average turn number of 19.0. These conversations contain in-depth discussions on related topics and natural transition between multiple topics, while the corpus can also used for exploration of transfer learning and domain adaptation.
"""

_CITATION = """\
@inproceedings{zhou-etal-2020-kdconv,
    title = "{K}d{C}onv: A {C}hinese Multi-domain Dialogue Dataset Towards Multi-turn Knowledge-driven Conversation",
    author = "Zhou, Hao  and
      Zheng, Chujie  and
      Huang, Kaili  and
      Huang, Minlie  and
      Zhu, Xiaoyan",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.635",
    doi = "10.18653/v1/2020.acl-main.635",
    pages = "7098--7108",
}
"""

_LICENSE = "NA"

DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/kdconv"


_HOMEPAGE = "https://github.com/thu-coai/KdConv"

class KdconvConfig(datalabs.BuilderConfig):

    def __init__(self, **kwargs):

        super(KdconvConfig, self).__init__(**kwargs)

class Kdconv(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        KdconvConfig(
            name="film",
            version=datalabs.Version("1.0.0"),
            description="film",
        ),
        KdconvConfig(
            name="music",
            version=datalabs.Version("1.0.0"),
            description="music",
        ),
        KdconvConfig(
            name="travel",
            version=datalabs.Version("1.0.0"),
            description="travel",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "content": datalabs.features.Sequence(datalabs.Value("string")),
                    "knowledge": datalabs.features.Sequence({
                                'name':datalabs.Value("string"),
                                'attrname':datalabs.Value("string"),
                                'attrvalue':datalabs.Value("string"),
                            }
                    )
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.knowledge_driven_dialogue)(
                    content_column = "content",
                    knowledge_column = "knowledge"
                )
            ],
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(os.path.join(DOWNLOAD_URL , self.config.name, "train.json"))
        validation_path = dl_manager.download_and_extract(os.path.join(DOWNLOAD_URL , self.config.name, "dev.json"))
        test_path = dl_manager.download_and_extract(os.path.join(DOWNLOAD_URL , self.config.name, "test.json"))
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            count=0
            for id_, conversation in enumerate(data):
                name, messages = conversation['name'], conversation["messages"]
                contents=[]
                for content in messages:
                    if 'attrs' in content:
                        
                        length=len(content['attrs'])
                        attrnames=[content['attrs'][i]['attrname'] for i in range(length)]
                        attrvalues=[content['attrs'][i]['attrvalue'] for i in range(length)]
                        names=[content['attrs'][i]['name'] for i in range(length)]
                       
                        contents.append(content['message'])
                        yield count, {
                                "content": contents,
                                "knowledge":{
                                    'name':names,
                                    'attrname':attrnames,
                                    'attrvalue':attrvalues,
                            }
                        }
                        count+=1
                        
                    else:
                        contents.append(content['message'])


                     




