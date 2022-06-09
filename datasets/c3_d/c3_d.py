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
Multiple-Choice Chinese Machine Reading Comprehension
This dataset contains dialogue contexts and corresponding questions and answers.
For more information, please refer to https://github.com/CLUEbenchmark/CLUE. 
"""

_CITATION = """\
@article{sun-etal-2020-investigating,
    title = "Investigating Prior Knowledge for Challenging {C}hinese Machine Reading Comprehension",
    author = "Sun, Kai  and
      Yu, Dian  and
      Yu, Dong  and
      Cardie, Claire",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "8",
    year = "2020",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2020.tacl-1.10",
    doi = "10.1162/tacl_a_00305",
    pages = "141--155",
    abstract = "Machine reading comprehension tasks require a machine reader to answer questions relevant to the given document. In this paper, we present the first free-form multiple-Choice Chinese machine reading Comprehension dataset (C3), containing 13,369 documents (dialogues or more formally written mixed-genre texts) and their associated 19,577 multiple-choice free-form questions collected from Chinese-as-a-second-language examinations. We present a comprehensive analysis of the prior knowledge (i.e., linguistic, domain-specific, and general world knowledge) needed for these real-world problems. We implement rule-based and popular neural methods and find that there is still a significant performance gap between the best performing model (68.5{\%}) and human readers (96.0{\%}), especiallyon problems that require prior knowledge. We further study the effects of distractor plausibility and data augmentation based on translated relevant datasets for English on model performance. We expect C3 to present great challenges to existing systems as answering 86.8{\%} of questions requires both knowledge within and beyond the accompanying document, and we hope that C3 can serve as a platform to study how to leverage various kinds of prior knowledge to better understand a given written or orally oriented text. C3 is available at https://dataset.org/c3/.",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/c3/d-train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/c3/d-dev.json"

_HOMEPAGE = "https://github.com/CLUEbenchmark/CLUE"

class C3dConfig(datalabs.BuilderConfig):

    def __init__(self, **kwargs):

        super(C3dConfig, self).__init__(**kwargs)

class C3d(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        C3dConfig(
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
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.qa_multiple_choice_c3)(
                    question_column = "question",
                    context_column = "context",
                    answers_column = "answers",
                    options_column = "options",
                )
            ],
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        key = 0

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data:
                context, qas, id = article[0], article[1], article[2]
                for qa in qas:
                    question, options, text = qa["question"], qa["choice"], qa["answer"]
                    if options.index(text):
                        option_index = options.index(text)
                        yield key, {
                            "id": id,"context": context, "question": question, "options": options, 
                            "answers": {
                                "text": text,
                                "option_index": option_index,
                            }
                        }
                        key = key + 1