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
DuSQL is a practical application-oriented dataset, including 200 databases, covering 164 fields, 
and the problems cover common forms in practical applications such as matching, calculation, and reasoning.
The input of the Text-to-SQL task is database (D) and natural language question (Q), 
and the output is the corresponding SQL query statement (Y).
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=13. 
"""

_CITATION = """\
@inproceedings{wang-etal-2020-dusql,
    title = "{D}u{SQL}: A Large-Scale and Pragmatic {C}hinese Text-to-{SQL} Dataset",
    author = "Wang, Lijie  and
      Zhang, Ao  and
      Wu, Kun  and
      Sun, Ke  and
      Li, Zhenghua  and
      Wu, Hua  and
      Zhang, Min  and
      Wang, Haifeng",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.562",
    doi = "10.18653/v1/2020.emnlp-main.562",
    pages = "6923--6935",
    abstract = "Due to the lack of labeled data, previous research on text-to-SQL parsing mainly focuses on English. Representative English datasets include ATIS, WikiSQL, Spider, etc. This paper presents DuSQL, a larges-scale and pragmatic Chinese dataset for the cross-domain text-to-SQL task, containing 200 databases, 813 tables, and 23,797 question/SQL pairs. Our new dataset has three major characteristics. First, by manually analyzing questions from several representative applications, we try to figure out the true distribution of SQL queries in real-life needs. Second, DuSQL contains a considerable proportion of SQL queries involving row or column calculations, motivated by our analysis on the SQL query distributions. Finally, we adopt an effective data construction framework via human-computer collaboration. The basic idea is automatically generating SQL queries based on the SQL grammar and constrained by the given database. This paper describes in detail the construction process and data statistics of DuSQL. Moreover, we present and compare performance of several open-source text-to-SQL parsers with minor modification to accommodate Chinese, including a simple yet effective extension to IRNet for handling calculation SQL queries.",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/DuSQL/train_revised.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/DuSQL/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/DuSQL/test_revised.json"

_HOMEPAGE = "https://aclanthology.org/2020.emnlp-main.562"

class DuSQLConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(DuSQLConfig, self).__init__(**kwargs)

class DuSQL(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DuSQLConfig(
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