# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and DataLab Authors.
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
"""WikiHop: Reading Comprehension with Multiple Hops"""


import json
import os
import datalabs
from datalabs import get_task, TaskType



_CITATION = """\
@article{zhang2020conceptualized,
  title={Conceptualized Representation Learning for Chinese Biomedical Text Mining},
  author={Zhang, Ningyu and Jia, Qianghuai and Yin, Kangping and Dong, Liang and Gao, Feng and Hua, Nengwei},
  journal={arXiv preprint arXiv:2008.10813},
  year={2020}
}
"""

_DESCRIPTION = """\
Information retrieval aims to retrieve most related documents given search queries. IR can be regarded as a ranking task. We use the cMedIR dataset, which consists of queries with multiple documents and their relative scores.
"""
_HOMEPAGE = "https://github.com/alibaba-research/ChineseBLUE/"
_LICENSE = "Available for research use"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/retrieval/cMedIR/train.json"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/retrieval/cMedIR/dev.json"
_TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/retrieval/cMedIR/test.json"



class cMedIR(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "query": datalabs.Value("string"),
                    "answers": 
                        {
                            "title": datalabs.features.Sequence(datalabs.Value("string")),
                            "label": datalabs.features.Sequence(datalabs.Value("int32")),
                        },
                
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            languages=['zh'],
            version=self.VERSION,
            task_templates=[
                get_task(TaskType.retrieval)(
                    query_column= "query",
                    answers_column="answers"
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
            for id, line in enumerate(f):
                example=json.loads(line)
             
                label = [doc["label"] for doc in example["documents"]]
                title = [doc["title"] for doc in example["documents"]]
             
                yield id, {
                    "id": id,
                    "query":   example["query"],
                    "answers": {
                        "label":label ,
                        "title":title ,
                    },
                }





