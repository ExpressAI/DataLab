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
import pandas as pd
import requests

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



TRAIN_SPLIT_URL_NI = "https://raw.githubusercontent.com/allenai/natural-instructions/6174af63465999768fbc09f5dd8a7f1a5dfe9abc/splits/default/train_tasks.txt"
TEST_SPLIT_URL_NI = "https://raw.githubusercontent.com/allenai/natural-instructions/6174af63465999768fbc09f5dd8a7f1a5dfe9abc/splits/default/test_tasks.txt"
TASK_URL_NI = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/ni/"


TASKS_LIST_NI_TRAIN = pd.read_csv(TRAIN_SPLIT_URL_NI, delimiter="\t", header=None, names=["task_names"])["task_names"].tolist()
TASKS_LIST_NI_TEST = pd.read_csv(TEST_SPLIT_URL_NI, delimiter="\t", header=None, names=["task_names"])["task_names"].tolist()
       


class NaturalInstructionConfig(datalabs.BuilderConfig):

    def __init__(
        self,
        name,
        data_dir,
        **kwargs
    ):
        """BuilderConfig for NaturalInstruction
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NaturalInstructionConfig, self).__init__(**kwargs)
        self.name = name
        self.data_dir = data_dir
        


class NaturalInstruction(datalabs.GeneratorBasedBuilder):
    """NaturalInstruction dataset."""


    tasks=[]
    for task_name in TASKS_LIST_NI_TRAIN:
        task='_'.join(task_name.split('_')[1:])
        tasks.append(task)
    for task_name in TASKS_LIST_NI_TEST:
        task='_'.join(task_name.split('_')[1:])
        tasks.append(task)

    tasks=set(tasks)
    BUILDER_CONFIGS = [
        NaturalInstructionConfig(
            name = task_name,
            data_dir = TASK_URL_NI + task_name +".jsonl"
        ) for task_name in tasks ]


    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "definition": datalabs.Value("string"),
                    "inputs": datalabs.Value("string"),
                    "targets": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.conditional_generation)(
                    source_column="inputs", reference_column="targets"
                )
            ],
            languages=["en"],
        )

    def _split_generators(self, dl_manager):

        test_path = dl_manager.download_and_extract(self.config.data_dir)

        return [
            
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]


    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                line = json.loads(line)
           
                yield id_, {
                    "definition": line['definition'],
                    "inputs": line['inputs'],
                    "targets": line['targets']
                } 