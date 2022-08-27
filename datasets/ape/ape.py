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
Argument Pair Extraction from Peer Review and Rebuttal.
Peer review and rebuttal, with rich interactions and argumentative discussions in between, are naturally a good resource to mine arguments. We introduce an argument pair extraction (APE) task on peer review and rebuttal in order to study the contents, the structure and the connections between them. Participants are required to detect the argument pairs from each passage pair of review and rebuttal.
"""

_CITATION = """\
@inproceedings{cheng2020ape,
  title={APE: argument pair extraction from peer review and rebuttal via multi-task learning},
  author={Cheng, Liying and Bing, Lidong and Yu, Qian and Lu, Wei and Si, Luo},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={7000--7011},
  year={2020}
}   
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/argument_pair_extraction/ape/train.txt"
)
_VALIDATION_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/argument_pair_extraction/ape/dev.txt"
)
_TEST_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/argument_pair_extraction/ape/test.txt"
)


class APE(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": Value("string"),
                    "sentences": Sequence(Value("string")),
                    "tags": Sequence(Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="http://www.fudan-disc.com/sharedtask/AIDebater21/tracks.html",
            citation=_CITATION,
            license=_LICENSE,
            languages=["en"],
            task_templates=[
                get_task(TaskType.argument_pair_extraction)(
                    sentences_column="sentences", labels_column="tags"
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
            sentences = []
            tags = []
            count = 0

            for row in f:
                row = row.strip()
                if row != "":
                    row_list = row.split("\t")
                    sentences.append(row_list[0])
                    tags.append(row_list[3]+'-'+row_list[2])
                else:
                    assert len(sentences) == len(tags), "mismatch between len of tokens & labels"
                    yield  count, {
                            "id": str(count),
                            "sentences": sentences,
                            "tags": tags,
                        },
                    sentences = []
                    tags = []
                    count += 1

            if sentences:
                yield count, {
                    "id": str(count),
                    "sentences": sentences,
                    "tags": tags,
                }