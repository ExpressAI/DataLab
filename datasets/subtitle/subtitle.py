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
Subtitle Corpus.
"""

_CITATION = """\

"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/subtitle/subtitle.tsv"

class Subtitle(datalabs.GeneratorBasedBuilder):

    VERSION = datalabs.Version("1.0.0")
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
            homepage="https://github.com/fateleak/dgk_lost_conv",
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.single_turn_dialogue)(
                    source_column="source", reference_column="reference"
                )
            ],
        )

        

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_validation_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_validation_path})
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples."""

        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for id_, line in enumerate(reader):
                yield id_, {"source": line[0], "reference": line[1]}