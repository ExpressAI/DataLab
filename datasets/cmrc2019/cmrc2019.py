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
The CMRC2019 dataset is a Sentence Cloze-Style Machine Reading Comprehension (SC-MRC) dataset.
Given a context with blanks and choices containing several sentences, 
the goal is to determine the sequence of sentences to fill in the blanks. 
"""

_CITATION = """\
@inproceedings{cui-etal-2020-sentence,
    title = "A Sentence Cloze Dataset for {C}hinese Machine Reading Comprehension",
    author = "Cui, Yiming  and
      Liu, Ting  and
      Yang, Ziqing  and
      Chen, Zhipeng  and
      Ma, Wentao  and
      Che, Wanxiang  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.589",
    pages = "6717--6723",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/CMRC2019/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/CMRC2019/dev.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/CMRC2019/trial.json"

_HOMEPAGE = "https://hfl-rc.com/cmrc2019/"

class CMRC2019Config(datalabs.BuilderConfig):

    def __init__(self, **kwargs):

        super(CMRC2019Config, self).__init__(**kwargs)

class CMRC2019(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CMRC2019Config(
            name="sentence_cloze_reading_comprehension",
            version=datalabs.Version("1.0.0"),
            description="sentence_cloze_reading_comprehension",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers": datalabs.features.Sequence(datalabs.Value("int32")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.cloze_multiple_choice)(
                    question_column = "id",
                    context_column = "context",
                    options_column = "options",
                    answers_column = "answers",
                )
            ],
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        count = 0
        with open(filepath, encoding="utf-8") as f:
            file = json.load(f)
            data = file["data"]
            for article in data:
                id = article["context_id"]
                context = article["context"]
                options = article["choices"]
                answers = article["answers"]

                yield count, {"id": id, "context": context, "options": options, "answers": answers}
                count = count + 1