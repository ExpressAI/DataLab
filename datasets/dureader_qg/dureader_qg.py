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
DuReader_QG is a subset of question-generation task selected from DuReader robust. 
It can also be used for question-answering task.
You can get the dataset form for question-answering tasks by defining name = "question_answering". 
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=8. 
"""

_CITATION = """\
@inproceedings{tang-etal-2021-dureader,
    title = "DuReader{\_}robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications",
    author = "Tang, Hongxuan  and
      Li, Hongyu  and
      Liu, Jing  and
      Hong, Yu  and
      Wu, Hua  and
      Wang, Haifeng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.120",
    doi = "10.18653/v1/2021.acl-short.120",
    pages = "955--963",
    abstract = "Machine reading comprehension (MRC) is a crucial task in natural language processing and has achieved remarkable advancements. However, most of the neural MRC models are still far from robust and fail to generalize well in real-world applications. In order to comprehensively verify the robustness and generalization of MRC models, we introduce a real-world Chinese dataset {--} DuReader{\_}robust . It is designed to evaluate the MRC models from three aspects: over-sensitivity, over-stability and generalization. Comparing to previous work, the instances in DuReader{\_}robust are natural texts, rather than the altered unnatural texts. It presents the challenges when applying MRC models to real-world applications. The experimental results show that MRC models do not perform well on the challenge test set. Moreover, we analyze the behavior of existing models on the challenge test set, which may provide suggestions for future model development. The dataset and codes are publicly available at https://github.com/baidu/DuReader.",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/DuReaderQG/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/DuReaderQG/dev.json"

_HOMEPAGE = "https://aclanthology.org/2021.acl-short.120"


class DuReaderQG(datalabs.GeneratorBasedBuilder):
    """DuReaderQG is a Dataset containing contexts, answers and questions."""

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="question_answering",
            version=datalabs.Version("1.0.0"),
            description="""
                An answer should be generated according to the given context and question.
                """,
        ),
        datalabs.BuilderConfig(
            name="question_generation",
            version=datalabs.Version("1.0.0"),
            description="""
                    An question should be generated according to the given context and answer.
                """,
        ),
    ]
    DEFAULT_CONFIG_NAME = "question_generation"

    def _info(self):

        if self.config.name == "question_answering":
            return datalabs.DatasetInfo(
                description=_DESCRIPTION,
                features=datalabs.Features(
                    {
                        "context": datalabs.Value("string"),
                        "question": datalabs.Value("string"),
                        "answer": datalabs.Value("string"),
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
                        answers_column="answer",
                    )
                ],
            )

        if self.config.name == "question_generation":
            return datalabs.DatasetInfo(
                description=_DESCRIPTION,
                features=datalabs.Features(
                    {
                        "source": datalabs.Value("string"),
                        "guidance": datalabs.Value("string"),
                        "reference": datalabs.Value("string"),
                    }
                ),
                supervised_keys=None,
                homepage=_HOMEPAGE,
                citation=_CITATION,
                languages=["zh"],
                task_templates=[
                    get_task(TaskType.guided_conditional_generation)(
                        source_column="context",
                        guidance_column="answer",
                        reference_column="question",
                    )
                ],
            )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples."""

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                if self.config.name == "question_answering":
                    context = line["context"]
                    question = line["question"]
                    answer = line["answer"]
                    yield id_, {
                        "context": context,
                        "question": question,
                        "answer": answer,
                    }
                else:
                    source = line["context"]
                    guidance = line["answer"]
                    reference = line["question"]
                    yield id_, {
                        "source": source,
                        "guidance": guidance,
                        "reference": reference,
                    }
