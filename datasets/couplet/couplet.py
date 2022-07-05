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

import json

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Chinese Couplets Dataset. Some couplets that contain vulgar words are removed from this dataset. This dataset contains around 740,000 couplets.
"""

_CITATION = """\
"""

_LICENSE = "NA"

IN_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/couplets/train/in.txt"
OUT_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/couplets/train/out.txt"
IN_TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/couplets/test/in.txt"
OUT_TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/couplets/test/out.txt"


class Couplet(datalabs.GeneratorBasedBuilder):
    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "source": datalabs.Value("string"),
                    "reference": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/v-zich/couplet-clean-dataset",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.conditional_generation)(
                    source_column="source", reference_column="reference"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        in_train_path = dl_manager.download_and_extract(IN_TRAIN_DOWNLOAD_URL)
        out_train_path = dl_manager.download_and_extract(OUT_TRAIN_DOWNLOAD_URL)
        in_test_path = dl_manager.download_and_extract(IN_TEST_DOWNLOAD_URL)
        out_test_path = dl_manager.download_and_extract(OUT_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"infilepath": in_train_path, "outfilepath": out_train_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"infilepath": in_test_path, "outfilepath": out_test_path},
            ),
        ]

    def _generate_examples(self, infilepath, outfilepath):
        """This function returns the examples."""

        with open(infilepath, encoding="utf-8") as fin, open(
            outfilepath, encoding="utf-8"
        ) as fout:
            sources = fin.readlines()
            references = fout.readlines()

            assert len(sources) == len(references)
            for id_, line in enumerate(sources):
                source = line.strip()
                reference = references[id_]
                yield id_, {"source": source, "reference": reference}
