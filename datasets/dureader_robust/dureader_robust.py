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
The DuReader_robust dataset is a single-chapter, extractive reading comprehension dataset.
Given a question (q) and a chapter (p), the system needs to give the answer (a) of the question. 
For more information, please refer to https://github.com/baidu/DuReader/tree/master/DuReader-Robust. 
"""

_CITATION = """\
@inproceedings{tang-etal-2021-dureader,
    title = "DuReader_robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications",
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

_LICENSE = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_robust/License.pdf"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_robust/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_robust/dev.json"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_robust/test.json"

_HOMEPAGE = "https://github.com/baidu/DuReader/tree/master/DuReader-Robust"


class DuReaderRobustConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(DuReaderRobustConfig, self).__init__(**kwargs)


class DuReaderRobust(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DuReaderRobustConfig(
            name="reading_comprehension",
            version=datalabs.Version("1.0.0"),
            description="reading_comprehension",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "question": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "answers": {
                        "text": datalabs.features.Sequence(datalabs.Value("string")),
                        "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                    }
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.qa)(
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
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples."""

        with open(filepath, encoding="utf-8") as f:
            file = json.load(f)
            data = file["data"][0]["paragraphs"]
            for id_, line in enumerate(data):
                qas, context = line["qas"][0], line["context"]
                question, answers = qas["question"], qas["answers"][0]
                if isinstance(answers["text"],list):
                    text = answers["text"]
                else:
                    text = []
                    text.append(answers["text"])
                if isinstance(answers["answer_start"],list):
                    answer_start = answers["answer_start"]
                else: 
                    answer_start = []
                    answer_start.append(answers["answer_start"])
                answers = {"text": text, "answer_start": answer_start}
                yield(id_, {"question": question, "context": context, "answers": answers})




            
