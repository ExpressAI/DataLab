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

from pkg_resources import yield_lines
import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
CFT (Children Fairy Tale) is a Chinese reading comprehension dataset. 
For a given document and a question which is a sentence from the document, the goal is to 
determine the answer that should be put in the blank of the question. 
There are a human evaluation test set and a automatically generated test set. 
For more information, please refer to https://github.com/ymcui/Chinese-Cloze-RC. 
"""

_CITATION = """\
@InProceedings{cui-etal-2016-consensus,
  title		= {Consensus Attention-based Neural Networks for Chinese Reading Comprehension},
  author	= {Cui, Yiming and Liu, Ting and Chen, Zhipeng and Wang, Shijin and Hu, Guoping},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
  year      = {2016},
  address   = {Osaka, Japan},
  pages     = {1777--1786},
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/children_fairy_tale/train_revised.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/children_fairy_tale/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/children_fairy_tale/test_revised.json"

_HOMEPAGE = "https://github.com/ymcui/Chinese-Cloze-RC"

class CFTConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(CFTConfig, self).__init__(**kwargs)

class CFT(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CFTConfig(
            name="documents_reading_comprehension",
            version=datalabs.Version("1.0.0"),
            description="documents_reading_comprehension",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "documents": datalabs.features.Sequence(datalabs.Value("string")),
                    "documents_tokens": datalabs.features.Sequence(datalabs.features.Sequence(datalabs.Value("string"))),
                    "question": datalabs.Value("string"),
                    "question_tokens": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.cloze_documents)(
                    context_column = "documents",
                    question_column = "question",
                    answers_column = "answers",
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

        count = 0
        with open(filepath, encoding='utf8') as f:
            for id_, line in enumerate(f):
                line = json.loads(line)
                yield id_, {
                    "documents": line["documents"], "documents_tokens": line["documents_tokens"], 
                    "question": line["question"], "question_tokens": line["question_tokens"], 
                    "answers": line["answers"]
                }
                