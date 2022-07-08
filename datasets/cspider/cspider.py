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
CSpider is a multilingual dataset, its problems are expressed in Chinese, and the database is stored in English. 
This dataset requires that the model is domain-agnostic, problem-agnostic, and capable of multilingual matching.
The input of the Text-to-SQL task is database (D) and natural language question (Q), 
and the output is the corresponding SQL query statement (Y).
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=11. 
"""

_CITATION = """\
@inproceedings{min-etal-2019-pilot,
    title = "A Pilot Study for {C}hinese {SQL} Semantic Parsing",
    author = "Min, Qingkai  and
      Shi, Yuefeng  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1377",
    doi = "10.18653/v1/D19-1377",
    pages = "3652--3658",
    abstract = "The task of semantic parsing is highly useful for dialogue and question answering systems. Many datasets have been proposed to map natural language text into SQL, among which the recent Spider dataset provides cross-domain samples with multiple tables and complex queries. We build a Spider dataset for Chinese, which is currently a low-resource language in this task area. Interesting research questions arise from the uniqueness of the language, which requires word segmentation, and also from the fact that SQL keywords and columns of DB tables are typically written in English. We compare character- and word-based encoders for a semantic parser, and different embedding schemes. Results show that word-based semantic parser is subject to segmentation errors and cross-lingual word embeddings are useful for text-to-SQL.",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/CSpider/train_revised.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/CSpider/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/semantic_parsing/CSpider/test_revised.json"

_HOMEPAGE = "https://aclanthology.org/D19-1377"

class CSpiderConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(CSpiderConfig, self).__init__(**kwargs)

class CSpider(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CSpiderConfig(
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