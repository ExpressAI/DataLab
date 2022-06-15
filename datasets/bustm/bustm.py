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
BUSTM: XiaoBu Dialogue Short Text Matching, a Dialog short text semantic matching data set, from XiaoBu.
XiaoBu is OPPO's own voice assistant for branded mobile phones and IoT devices, providing users with convenient conversational services.
Intention recognition is a core task in dialogue system, and short text semantic matching is one of the mainstream algorithm schemes for intention recognition, 
which requires to predict whether the two sentences in a short text Query-pair have the same meaning. 
Data scale: train set(32), validation set(32), test set with labels(1772), test set without labels(2000), unlabeled corpus(4251). 
For more information, please refer to https://github.com/CLUEbenchmark/FewCLUE. 
"""

_CITATION = """\
@article{Xu2021FewCLUEAC,
  title={FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark},
  author={Liang Xu and Xiaojing Lu and Chenyang Yuan and Xuanwei Zhang and Huining Yuan and Huilin Xu and Guoao Wei and Xiang Pan and Hai Hu},
  journal={ArXiv},
  year={2021},
  volume={abs/2107.07498}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bustm/train.json"
)
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bustm/dev.json"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bustm/test.json"
)
# _TEST_UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bustm/test_unlabeled.json"
# _UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bustm/unlabeled.json"


class BUSTM(datalabs.GeneratorBasedBuilder):
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
            homepage="https://github.com/CLUEbenchmark/FewCLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.paraphrase_identification)(
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
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                text1, text2, label = (
                    line["sentence1"],
                    line["sentence2"],
                    line["label"],
                )
                if label == ("0" or "1"):
                    yield id_, {"text1": text1, "text2": text2, "label": label}
