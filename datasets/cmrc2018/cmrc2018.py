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
The CMRC2018 dataset is a single-chapter, extractive reading comprehension dataset.
Given a question (q), a chapter (p) and its title (t), the system needs to give the answers (a) of the question. 
For more information, please refer to https://github.com/CLUEbenchmark/CLUE. 
"""

_CITATION = """\
@inproceedings{cui-etal-2019-span,
    title = "A Span-Extraction Dataset for Chinese Machine Reading Comprehension",
    author = "Cui, Yiming  and
      Liu, Ting  and
      Che, Wanxiang  and
      Xiao, Li  and
      Chen, Zhipeng  and
      Ma, Wentao  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1600",
    pages = "5886--5891",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/cmrc2018/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/cmrc2018/dev.json"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/cmrc2018/test.json"

_HOMEPAGE = "https://github.com/CLUEbenchmark/CLUE"

class CMRC2018Config(datalabs.BuilderConfig):

    def __init__(self, **kwargs):

        super(CMRC2018Config, self).__init__(**kwargs)

class CMRC2018(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CMRC2018Config(
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
                    "context": datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "answers":
                        {
                            "text": datalabs.features.Sequence(datalabs.Value("string")),
                            "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                        },
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.qa_extractive)(
                    question_column = "question",
                    context_column = "context",
                    answers_column = "answers",
                )
            ],
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        count = 0
        with open(filepath, encoding="utf-8") as f:
            file = json.load(f)
            data = file["data"]
            for article in data:
                title = article["title"]
                paragraphs = article["paragraphs"][0]
                context = paragraphs["context"]
                qas = paragraphs["qas"]
                # qas is a list, contaning many Q&A groups. 
                for qa in qas:
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    text = [answer["text"] for answer in qa["answers"]]
                    yield count, {
                        "title": title, 
                        "context": context, 
                        "question": qa["question"], 
                        "id": qa["id"],
                        "answers": {
                            "text": text,
                            "answer_start": answer_starts,
                        },
                    }
                    count = count + 1