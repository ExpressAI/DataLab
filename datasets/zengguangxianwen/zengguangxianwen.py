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
"Zeng Guang Xian Wen" is an enlightenment book for children in ancient China. The title of the book was first seen in the opera "Peony Pavilion" in the Wanli period of the Ming Dynasty, so it can be inferred that the book was written in the Wanli period at the latest. Later, after the continuous addition of literati in the Ming and Qing dynasties, it was changed to what it is now, called "Zeng Guang Xi Shi Xian Wen", commonly known as "Zeng Guang Xian Wen". 
"""

_HOMEPAGE = "https://github.com/chinese-poetry/chinese-poetry"
_LICENSE = "MIT"
_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/poetry/mengxue/zengguangxianwen.json"


class ZengGuangXianWen(datalabs.GeneratorBasedBuilder):
   
    VERSION = datalabs.Version("1.0.0")

    def _info(self):
       
        return datalabs.DatasetInfo(
           
            description=_DESCRIPTION,
          
            features=Features(
                {
                    "title":Value("string"),
                    "author":Value("string"),
                    "abstract":Value("string"),
                    "content": {
                        "chapter":Value("string"),
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
                    content_column= "content",
                    abstract_column= "abstract",
                    
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
            for id_,item in enumerate(data['content']):

                    yield count, {
                        'title': data['title'],
                        "author": data["author"],
                        "abstract":data["abstract"],
                        "content": {
                            "chapter":item["chapter"],
                            "paragraphs":  item['paragraphs'],

                        }
                    
                    }
                    count+=1