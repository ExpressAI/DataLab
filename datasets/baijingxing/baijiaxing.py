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
"Bai Jia Xing" is a work about Chinese surnames. According to literature records, it was written in the early Northern Song Dynasty. Originally, 411 surnames were collected, but they were added to 504, including 444 single surnames and 60 compound surnames.
"""

_HOMEPAGE = "https://github.com/chinese-poetry/chinese-poetry"
_LICENSE = "MIT"
_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/poetry/mengxue/baijiaxing.json"


class Baijiaxing(datalabs.GeneratorBasedBuilder):
   
    VERSION = datalabs.Version("1.0.0")

    def _info(self):
       
        return datalabs.DatasetInfo(
           
            description=_DESCRIPTION,
          
            features=Features(
                {
                    "title":Value("string"),
                    "author":Value("string"),
                    "tags": Value("string"),
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
                    author_column="author",
                    tags_column="tags",
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
            
            count=0

            yield count, {
                'title': data['title'],
                "author": data["author"],
                "tags": data["tags"],
                "content": {
                    "paragraphs":  data['paragraphs'],

                }
                    
            }
            count+=1