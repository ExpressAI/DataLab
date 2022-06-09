# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and DataLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import os

import datalabs

# from datalabs.tasks import QuestionAnsweringMultipleChoicesWithoutContext
from datalabs import get_task, TaskType


_CITATION = """\
@inproceedings{dogwhistle,
author    = {Canwen Xu and
             Wangchunshu Zhou and
             Tao Ge and
             Ke Xu and
             Julian McAuley and
             Furu Wei},
title     = {Blow the Dog Whistle: A Chinese Dataset for Cant Understanding with Common Sense and World Knowledge},
booktitle = {{NAACL}},
year      = {2021}
}
"""


_DESCRIPTION = """\
A large and diverse Chinese dataset for creating and understanding cant from a computational linguistics perspective.

"""


DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/cant"

class CantConfig(datalabs.BuilderConfig):

    def __init__(self, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CantConfig, self).__init__(version=datalabs.Version("1.0.0", ""), **kwargs)


class Cant(datalabs.GeneratorBasedBuilder):
    
    VERSION = datalabs.Version("1.0.0")
    BUILDER_CONFIGS = [
        CantConfig(
            name="insider",
            description="""\
        Insider subtask. In this subtask, we mimic communication between insiders. The input (white background) is hidden words, cant context and a cant to decode. The model should output the index of the predicted hidden word (gray background). 
        The hidden words are visible in this subtask.
          """,
        ),
        CantConfig(
            name="outsider",
            description="""\
         Outsider subtask. In this subtask, an outsider tries to decrypt the communication by reading the cant history from previous rounds. The input is cant histories, cant context and a cant to decode (white background). The model should output the index of the predicted cant history (gray background).
         The hidden words are not visible in this subtask
          """,
        ),
    ]

    def _info(self):
       
        return datalabs.DatasetInfo(
           
            description=_DESCRIPTION,
          
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "context": datalabs.features.Sequence(datalabs.Value("string")),
                    "question": datalabs.Value("string"),
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers":
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },
                }
            ),

            supervised_keys=None,
            homepage="https://competitions.codalab.org/competitions/30451",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.qa_multiple_choice_c3)(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                    options_column="options",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        train_path = dl_manager.download_and_extract(os.path.join(DOWNLOAD_URL , self.config.name, "train.tsv"))
        validation_path = dl_manager.download_and_extract(os.path.join(DOWNLOAD_URL , self.config.name, "dev.tsv"))
        test_path = dl_manager.download_and_extract(os.path.join(DOWNLOAD_URL , self.config.name, "test.tsv"))
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]


    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            csvreader= csv.reader(f,delimiter="\t")
            for id, line in enumerate(csvreader):
              
                choices=line[1].split(',')
                option_index=int(line[4])
                yield id, {
                    "id": str(id),
                    "question": line[3],
                    "context":line[2].split(','),
                    "options": choices,
                    "answers": 
                        {
                            "text": choices[option_index],
                            "option_index": option_index,
                        }
                }
           







