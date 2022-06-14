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

from pkg_resources import yield_lines
import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
A Jointly Labeled Large-Scale Dataset for Classification, Question Answering and Summarization.
For a given question, a description and an answer, the goal is to classify the question-answer pair.  
For more information, please refer to https://github.com/WHUIR/MATINF. 
"""

_CITATION = """\
@inproceedings{xu-etal-2020-matinf,
    title = "{MATINF}: A Jointly Labeled Large-Scale Dataset for Classification, Question Answering and Summarization",
    author = "Xu, Canwen  and
      Pei, Jiaxin  and
      Wu, Hongtao  and
      Liu, Yiyu  and
      Li, Chenliang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.330",
    pages = "3586--3596",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/matinf/train.csv"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/matinf/dev.csv"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/matinf/test.csv"

_HOMEPAGE = "https://github.com/WHUIR/MATINF"

class MATINFConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(MATINFConfig, self).__init__(**kwargs)

class MATINF(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        MATINFConfig(
            name="question_answering_classification",
            version=datalabs.Version("1.0.0"),
            description="question_answering_classification",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": {
                        "question": datalabs.Value("string"),
                        "description": datalabs.Value("string"),
                        "answers": datalabs.Value("string"),
                    },
                    "label": datalabs.features.ClassLabel(names=[
                        '0-1岁', '1-2岁', '2-3岁', 
                        '产褥期保健', '儿童过敏', '动作发育',
                        '婴幼保健', '婴幼心理', '婴幼早教', '婴幼期喂养', '婴幼营养',
                        '孕期保健', '家庭教育','幼儿园','未准父母','流产和不孕',
                        '疫苗接种','皮肤护理','宝宝上火','腹泻','婴幼常见病'
                    ])
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.question_answering_classification)(
                    text_column = "text",
                    label_column = "label",
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

        labels = [
                    '0-1岁', '1-2岁', '2-3岁', 
                    '产褥期保健', '儿童过敏', '动作发育',
                    '婴幼保健', '婴幼心理', '婴幼早教', '婴幼期喂养', '婴幼营养',
                    '孕期保健', '家庭教育','幼儿园','未准父母','流产和不孕',
                    '疫苗接种','皮肤护理','宝宝上火','腹泻','婴幼常见病'
                ]
        head = 0
        with open(filepath, encoding='utf8') as f:
            for id_, line in enumerate(f.readlines()):
                if head > 0:
                    line = line.rstrip().split(",")
                    question, description, answers, label = line[1], line[2], line[3], line[4]
                    text = {"question": question, "description": description, "answers": answers}
                    if label in labels:
                        yield id_, {"text": text, "label": label}
                head = head + 1