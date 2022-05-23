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
The DRCD (Delta Reading Comprehension Dataset) dataset is a general domain traditional Chinese reading comprehension corpus.
This dataset has 10,014 paragraphs from 2,108 wiki entries and over 30,000 questions annotated from these paragraphs.
For more information, please refer to https://github.com/DRCKnowledgeTeam/DRCD. 
"""

_CITATION = """\
@article{Shao2018DRCDAC,
  title={DRCD: a Chinese Machine Reading Comprehension Dataset},
  author={Chih-Chieh Shao and Trois Liu and Yuting Lai and Yiying Tseng and Sam Tsai},
  journal={ArXiv},
  year={2018},
  volume={abs/1806.00920}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/drcd/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/drcd/dev.json"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/drcd/test.json"

_HOMEPAGE = "https://github.com/DRCKnowledgeTeam/DRCD"

class DRCDConfig(datalabs.BuilderConfig):

    def __init__(self, **kwargs):

        super(DRCDConfig, self).__init__(**kwargs)

class DRCD(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DRCDConfig(
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
                    "title": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
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
                paragraphs = article["paragraphs"]
                for paragraph in paragraphs:
                    context = paragraph["context"]
                    qas = paragraph["qas"]
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