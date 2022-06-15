# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
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
Question answering can be approximated as ranking candidate answer sentences based on their similarity. We assign 0,1 labels to the QA pairs, which convert to the binary classification problem. We use cMedQA released from the paper "Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection", which consists of questions and their answers.
"""

_CITATION = """\
@article{zhang2018multi,
  title={Multi-scale attentive interaction networks for chinese medical question answer selection},
  author={Zhang, Sheng and Zhang, Xin and Wang, Hui and Guo, Lixiang and Liu, Shanshan},
  journal={IEEE Access},
  volume={6},
  pages={74061--74071},
  year={2018},
  publisher={IEEE}
}
@article{zhang2020conceptualized,
  title={Conceptualized Representation Learning for Chinese Biomedical Text Mining},
  author={Zhang, Ningyu and Jia, Qianghuai and Yin, Kangping and Dong, Liang and Gao, Feng and Hua, Nengwei},
  journal={arXiv preprint arXiv:2008.10813},
  year={2020}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/cMedQA/train.txt"
)
_VALIDATION_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/cMedQA/dev.txt"
)
_TEST_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/cMedQA/test.txt"
)


class cMedQA(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text1": datalabs.Value("string"),
                    "text2": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["0", "1"]),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/alibaba-research/ChineseBLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.text_pair_classification)(
                    text1_column="text1", text2_column="text2", label_column="label"
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

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                example = row.strip().split("\t")

                yield id_, {
                    "text1": example[1],
                    "text2": example[2],
                    "label": example[0],
                }
