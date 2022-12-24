# coding=utf-8
# Copyright 2022 DataLab Authors.
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




import os
import json
import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.file_utils import get_from_cache
from datalabs.utils.extract import ExtractManager

logger = datalabs.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{wang2022super,
  title={Super-naturalinstructions: generalization via declarative instructions on 1600+ tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and Mirzaei, Amirreza and Arunkumar, Anjana and Ashok, Arjun and Dhanasekaran, Arut Selvan and Naik, Atharva and Stap, David and others},
  year={2022},
  organization={EMNLP}
}
"""

_DESCRIPTION = """\
This is a benchmark of 1,616 diverse NLP tasks and
their expert-written instructions. It covers 76 distinct task types, including but
not limited to classification, extraction, infilling, sequence tagging, text rewriting, and text
composition. This large and diverse collection of tasks enables rigorous benchmarking of
cross-task generalization under instructions—training models to follow instructions on a subset of tasks and evaluating them on the remaining unseen ones

For more details see https://instructions.apps.allenai.org/
"""

_HOMEPAGE = "https://instructions.apps.allenai.org/"


_URL = "xxx"
_TRAINING_FILE = "xxx"
_TEST_FILE = "xxx"


class NaturalInstructionConfig(datalabs.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(
        self,
        task_id,
        instruction,
        input,
        target,
        **kwargs
    ):
        """BuilderConfig for NaturalInstruction

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.task_id = task_id
        self.instruction = instruction
        self.input = input
        self.target = target
        super(NaturalInstructionConfig, self).__init__(**kwargs)


class NaturalInstruction(datalabs.GeneratorBasedBuilder):
    """NaturalInstruction dataset."""

    # the following code should be modified based on specific situation
    url_of_data = "XXXXXXXX"
    file_path = get_from_cache(url)
    local_path = ExtractManager().extract(file_path)

    # read json file local file:
    with open(local_path, 'r') as fin:
        # the following line should be modified based on file type (json, tsv or others)
        pre_process_data = json.load(fin)





    BUILDER_CONFIGS = [
        NaturalInstructionConfig(
            name = sample["name"],
            version=datalabs.Version("1.0.0"),
            task_id = sample["task_id"],
            instruction = sample["instruction"],
            input = sample["input"],
            target = sample["target"],
        ) for sample in pre_process_data ]


    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "task_id": datalabs.Value("string"),
                    "instruction": datalabs.Value("string"),
                    "input": datalabs.Value("string"),
                    "target": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.conditional_generation)(
                    source_column="input", reference_column="target"
                )
            ],
            languages=["en"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        ...

    def _generate_examples(self, filepath):
        ...