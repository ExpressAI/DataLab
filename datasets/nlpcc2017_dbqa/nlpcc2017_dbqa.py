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

import csv

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
This is a DBQA (Document-based Question Answering) dataset.
Each piece of data has 3 columns, which are question (q), document sentence (s), and label (l). 
If the document sentence is the correct answer of the given question, the label will be 1, otherwise the label will be 0.
"""

_CITATION = """\
@inproceedings{duan2016nlpcc, 
    title={Overview of the NLPCC-ICCPOL 2016 Shared Task: Open Domain Chinese Question Answering}, 
    author={Nan Duan}, 
    booktitle={NLPCC/ICCPOL}, 
    year={2016} 
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/NLPCC2017DBQA/train.txt"
)
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/NLPCC2017DBQA/dev.txt"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/NLPCC2017DBQA/test.txt"
)

_HOMEPAGE = "http://tcci.ccf.org.cn/conference/2017/taskdata.php"


class NLPCC2017DBQAConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(NLPCC2017DBQAConfig, self).__init__(**kwargs)


class NLPCC2017DBQA(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NLPCC2017DBQAConfig(
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
                    "answer": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["0", "1"]),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.question_answering_classification)(
                    text1_column="question",
                    text2_column="answer",
                    label_column="label",
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

        with open(filepath, encoding="utf-8") as file:
            csv_reader = csv.reader(file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                if len(row) == 3:
                    label, question, answer = row
                    label = int(label)
                    if label == (0 or 1):
                        yield id_, {
                            "question": question,
                            "answer": answer,
                            "label": label,
                        }
