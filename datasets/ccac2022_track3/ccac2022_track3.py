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
Provide an interative single-turn debates with both the positive and negative side. For an argument on one side, select one of the 5 candidate arguments that has a direct interaction with the given argument. 
The candidate set consists of one manually annotated interactive argument and four unrelated arguments sampled from the same debate. An argument may contain 1 or more sentences.
"""

_CITATION = """\
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/argument_pair_identification/ccac2022_track3/train.txt"
)

class CCAC2022_track3(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                   "quotation": Value("string"),
                    "replies": Sequence(Value("string")),
                    "label": Value("int32"),
                    "quotation_context": Value("string"),
                    "replies_context": Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="http://www.fudan-disc.com/sharedtask/AIDebater22/tracks.html",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.argument_pair_identification)(
                    context_column= "quotation",
                    utterance_column="replies",
                    label_column="label",
                     
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath,encoding='latin-1') as f:
            for id_, line in enumerate(f):
                line = line.split('\t')

                yield id_, {
                    "quotation": line[2], 
                    "replies":  [line[3],line[4],line[5],line[6],line[7]],
                    "label": line[8],
                    "quotation_context": line[0], 
                    "replies_context": line[1],
                }