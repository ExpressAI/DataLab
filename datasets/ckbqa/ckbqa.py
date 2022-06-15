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

import re

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
This evaluation task is Chinese Knowledge Base Question Answering (CKBQA).
The goal is, for a given Chinese question, to select several entities or attribute values ​​
from the given open-domain knowledge base as the answer to the question. 
This task uses PKU BASE as the specified knowledge graph.
The download url is https://pan.baidu.com/s/1MOv9PCTcALVIiodUP4bQ2Q and the password is hcu8. 
The corresponding knowledge base management system (eg. gStore system: http://gstore-pku.com/ ) can be used. 
We also provide an online query terminal of PKU BASE, and contestants can conduct SPARQL queries through browsers or calling APIs. 
Visit http://pkubase.gstore-pku.com/ for details.
"""

_CITATION = """\
For more information, please refer to http://www.sigkg.cn/ccks2019/?page_id=49.
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/CKBQA/train.txt"
)
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/CKBQA/valid.txt"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/CKBQA/test.txt"
)

_HOMEPAGE = "http://www.sigkg.cn/ccks2019/"


class CKBQAConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(CKBQAConfig, self).__init__(**kwargs)


class CKBQA(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CKBQAConfig(
            name="open_domain_question_answering",
            version=datalabs.Version("1.0.0"),
            description="open_domain_question_answering",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "question": datalabs.Value("string"),
                    "query": datalabs.Value("string"),
                    "answers": datalabs.features.Sequence(datalabs.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.qa_open_domain)(
                    question_column="question",
                    context_column="query",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):

        id_ = 0

        with open(filepath, encoding="utf8") as f:
            line = True
            str_pat = re.compile(r'"(.*?)"')
            while line:
                line = f.readline()
                if len(line.split(":")) > 1:
                    question = line.split(":")[1].rstrip()
                line = f.readline()
                query = line.rstrip()
                line = f.readline()
                answers = []
                answers.extend(line.rstrip().split("\t"))
                for index in range(len(answers)):
                    li = str_pat.findall(answers[index])
                    if len(li) > 0:
                        del answers[index]
                        [answers.insert(index, d) for d in li]
                line = f.readline()
                if line == "\n":
                    yield id_, {
                        "question": question,
                        "query": query,
                        "answers": answers,
                    }
                    id_ = id_ + 1
