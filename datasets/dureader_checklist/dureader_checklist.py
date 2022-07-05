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

logger = datalabs.logging.get_logger(__name__)

_DESCRIPTION = """\
The DuReaderchecklist dataset is a single-chapter, extractive reading comprehension dataset.
Given a question (q), a chapter (p) and its title (t), the system needs to judge 
whether the chapter (p) contains the answer to the given question according to the content of the chapter, 
and if so, give the answer (a) of the question; 
otherwise, output "No Answer".
For more information, please refer to https://github.com/baidu/DuReader/tree/master/DuReader-Checklist. 
"""

_CITATION = """\
@inproceedings{he-etal-2018-dureader,
    title = "DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications",
    author = "He, Wei  and
      Liu, Kai  and
      Liu, Jing  and
      Lyu, Yajuan  and
      Zhao, Shiqi  and
      Xiao, Xinyan  and
      Liu, Yuan  and
      Wang, Yizhong  and
      Wu, Hua  and
      She, Qiaoqiao  and
      Liu, Xuan  and
      Wu, Tian  and
      Wang, Haifeng",
    booktitle = "Proceedings of the Workshop on Machine Reading for Question Answering",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-2605",
    doi = "10.18653/v1/W18-2605",
    pages = "37--46",
    abstract = "This paper introduces DuReader, a new large-scale, open-domain Chinese machine reading comprehension (MRC) dataset, designed to address real-world MRC. DuReader has three advantages over previous MRC datasets: (1) data sources: questions and documents are based on Baidu Search and Baidu Zhidao; answers are manually generated. (2) question types: it provides rich annotations for more question types, especially yes-no and opinion questions, that leaves more opportunity for the research community. (3) scale: it contains 200K questions, 420K answers and 1M documents; it is the largest Chinese MRC dataset so far. Experiments show that human performance is well above current state-of-the-art baseline systems, leaving plenty of room for the community to make improvements. To help the community make these improvements, both DuReader and baseline systems have been posted online. We also organize a shared competition to encourage the exploration of more models. Since the release of the task, there are significant improvements over the baselines.",
}
"""

_LICENSE = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_checklist/License.pdf"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_checklist/train_revised.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_checklist/validation_revised.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_checklist/test_revised.json"

_HOMEPAGE = "https://github.com/baidu/DuReader/tree/master/DuReader-Checklist"


class DuReaderChecklistConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(DuReaderChecklistConfig, self).__init__(**kwargs)


class DuReaderChecklist(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DuReaderChecklistConfig(
            name="Reading Comprehension",
            version=datalabs.Version("1.0.0"),
            description="Reading Comprehension",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                    "answers": {
                        "text": datalabs.features.Sequence(datalabs.Value("string")),
                        "answer_start": datalabs.features.Sequence(
                            datalabs.Value("int32")
                        ),
                    },
                    "is_impossible": datalabs.Value("string"),
                    "type": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.qa_extractive)(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
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
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                yield id_, {
                    "title": ["title"],
                    "context": ["context"],
                    "question": line["question"],
                    "id": line["id"],
                    "answers": line["answers"],
                    "is_impossible": line["is_impossible"],
                    "type": line["type"],
                    # "answers":answer_list,
                }
