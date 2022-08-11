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
from datalabs.features import Features, Value, Sequence

_DESCRIPTION = """\
Interactive Argument Pair Identification in Online Forum.
Interactive argument opposition refers to the opposite views expressed by different participants on the same topic in a dialogical argumentation scenario (such as a debate contest, which involves two or more parties).
This task is to identify the argument pairs with interactive relationship in online forum. Given an original argument and five candidate arguments, this task aims to identify the correct one for the candidates. For each argument, its context are provided as well.
"""

_CITATION = """\
@article{ji2019discrete,
  title={Discrete argument representation learning for interactive argument pair identification},
  author={Ji, Lu and Wei, Zhongyu and Li, Jing and Zhang, Qi and Huang, Xuanjing},
  journal={arXiv preprint arXiv:1911.01621},
  year={2019}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/iapi/train.txt"
)
_TEST_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/iapi/test.txt"
)


class IAPI(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "quotation_context": Value("string"),
                    "quotation": Value("string"),
                    "positive_reply": Value("string"),
                    "positive_reply_context": Value("string"),
                    "negative_replies":{
                        "negative_reply": Sequence(Value("string")),
                        "negative_reply_context": Sequence(Value("string")),
                    }
                }
            ),
            supervised_keys=None,
            homepage="http://www.fudan-disc.com/sharedtask/AIDebater21/tracks.html",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.argument_pair_identification)(
                    quotation_context_column="quotation_context", 
                    quotation_column="quotation", 
                    positive_reply_column="positive_reply",
                    positive_reply_context_column="positive_reply_context",
                    negative_replies_column="negative_replies",
                     
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                line = line.split('#')

                yield id_, {
                    "quotation_context": line[0], 
                    "quotation": line[1], 
                    "positive_reply": line[2], 
                    "positive_reply_context": line[3],
                    "negative_replies":{
                        "negative_reply": [line[4],line[6],line[8],line[10]],
                        "negative_reply_context":[line[5],line[7],line[9],line[11]],
                   }
                }
