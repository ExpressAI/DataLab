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

_DESCRIPTION = """
FinRE is a manual-labeled financial news relation extraction dataset for chinese literature text. 
"""
_CITATION = """\
@inproceedings{li2019chinese,
    title={Chinese Relation Extraction with Multi-Grained Information and External Linguistic Knowledge},
    author={Li, Ziran and Ding, Ning and Liu, Zhiyuan and Zheng, Hai-Tao and Shen, Ying},
    booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    pages={4377--4386},
    year={2019}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/relation_extraction/FinRE/train.txt"
)
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/relation_extraction/FinRE/valid.txt"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/relation_extraction/FinRE/test.txt"
)

_HOMEPAGE = "https://github.com/thunlp/Chinese_NRE"


class FinREConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):
        super(FinREConfig, self).__init__(**kwargs)


class FinRE(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        FinREConfig(
            name="relation_extraction",
            version=datalabs.Version("1.0.0"),
            description="relation_extraction",
        ),
    ]

    DEFAULT_CONFIG_NAME = "relation_extraction"

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "span1": datalabs.Value("string"),
                    "span2": datalabs.Value("string"),
                    "text": datalabs.Value("string"),
                    "relation": datalabs.features.ClassLabel(
                        names=[
                            "unknown",
                            "注资",
                            "拥有",
                            "纠纷",
                            "自己",
                            "增持",
                            "重组",
                            "买资",
                            "签约",
                            "持股",
                            "交易",
                            "入股",
                            "转让",
                            "成立",
                            "分析",
                            "合作",
                            "帮助",
                            "发行",
                            "商讨",
                            "合并",
                            "竞争",
                            "订单",
                            "减持",
                            "合资",
                            "收购",
                            "借壳",
                            "欠款",
                            "被发行",
                            "被转让",
                            "被成立",
                            "被注资",
                            "被持股",
                            "被拥有",
                            "被收购",
                            "被帮助",
                            "被借壳",
                            "被买资",
                            "被欠款",
                            "被增持",
                            "拟收购",
                            "被减持",
                            "被分析",
                            "被入股",
                            "被拟收购",
                        ]
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.span_relation_prediction)(
                    text_column="text",
                    span1_column="span1",
                    span2_column="span2",
                    label_column="relation",
                ),
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

        relations = [
            "unknown",
            "注资",
            "拥有",
            "纠纷",
            "自己",
            "增持",
            "重组",
            "买资",
            "签约",
            "持股",
            "交易",
            "入股",
            "转让",
            "成立",
            "分析",
            "合作",
            "帮助",
            "发行",
            "商讨",
            "合并",
            "竞争",
            "订单",
            "减持",
            "合资",
            "收购",
            "借壳",
            "欠款",
            "被发行",
            "被转让",
            "被成立",
            "被注资",
            "被持股",
            "被拥有",
            "被收购",
            "被帮助",
            "被借壳",
            "被买资",
            "被欠款",
            "被增持",
            "拟收购",
            "被减持",
            "被分析",
            "被入股",
            "被拟收购",
        ]
        with open(filepath, encoding="utf-8") as txt_file:
            txt_file = csv.reader(txt_file, delimiter="\t")
            for id_, row in enumerate(txt_file):
                if len(row) == 4:
                    span1, span2, relation, text = row
                    if relation in relations:
                        yield id_, {
                            "text": text,
                            "span1": span1,
                            "span2": span2,
                            "relation": relation,
                        }
