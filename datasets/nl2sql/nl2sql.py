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
NL2SQL is a practical application-oriented dataset, including 200 databases, covering 164 fields, 
and the problems cover common forms in practical applications such as matching, calculation, and reasoning.
The input of the Text-to-SQL task is database (D) and natural language question (Q), 
and the output is the corresponding SQL query statement (Y).
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=12. 
"""

_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2006.06434,
  doi = {10.48550/ARXIV.2006.06434},
  url = {https://arxiv.org/abs/2006.06434},
  author = {Sun, Ningyuan and Yang, Xuefeng and Liu, Yunfeng},
  keywords = {Databases (cs.DB), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {TableQA: a Large-Scale Chinese Text-to-SQL Dataset for Table-Aware SQL Generation},
  publisher = {arXiv},
  year = {2020},
  copyright = {Creative Commons Zero v1.0 Universal}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/NL2SQL/train_revised.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/NL2SQL/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/NL2SQL/test_revised.json"

_HOMEPAGE = "https://www.luge.ai/#/luge/dataDetail?id=12"

class NL2SQLConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(NL2SQLConfig, self).__init__(**kwargs)

class NL2SQL(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NL2SQLConfig(
            name="text_to_sql",
            version=datalabs.Version("1.0.0"),
            description="text_to_sql",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "question": datalabs.Value("string"),
                    "query": datalabs.Value("string"),
                    "database_id": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.text_to_sql)(
                    question_column = "question",
                    query_column = "query",
                )
            ],
        )


    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        valid_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [

            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": valid_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding='utf8') as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                yield id_, {"question": line["question"], "query": line["query"], "database_id": line["database_id"]}