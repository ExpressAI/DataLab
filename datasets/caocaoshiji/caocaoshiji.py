# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
import os

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Features, Value, Sequence


_CITATION = """\
"""

_DESCRIPTION = """\
Cao Cao likes to use poetry and prose to express his political aspirations and reflect the suffering of people's livelihood. There are more than 20 poems of Cao Cao, all of them in Yuefu style. The content can be roughly divided into three categories. The first category is related to current affairs, the second category is mainly about expressing ideals, and the third category is  immortal poems.
"""

_HOMEPAGE = "https://github.com/chinese-poetry/chinese-poetry"
_LICENSE = "MIT"
_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/poetry/caocao.json"


class Caocaoshiji(datalabs.GeneratorBasedBuilder):
   
    VERSION = datalabs.Version("1.0.0")

    def _info(self):
       
        return datalabs.DatasetInfo(
           
            description=_DESCRIPTION,
          
            features=Features(
                {
                    "title": Value("string"),
                    "content": {
                        "paragraphs": Sequence(Value("string")),
                    } 
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.poetry)(
                    title_column= "title",
                    content_column= "content"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]


    def _generate_examples(self, filepath):
        with open(filepath, 'r') as f:
            data=json.load(f)
            for id_,item in enumerate(data):
                yield id_, {
                    'title': item['title'],
                    'content':{
                        "paragraphs": item['paragraphs']
                    }
                }