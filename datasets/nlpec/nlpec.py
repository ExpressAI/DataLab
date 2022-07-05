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
National Licensed Pharmacist Examination in China is the source of questions. 
The pharmacy comprehensive knowledge and skills part of the exam consists of 600 multiple-choice (included multi-answer type) problems over four categories. 
The test set includes examples of this part in the previous five years (2015-2019) and excludes questions of multi-answer type. 
The medical MRC task of NLPEC dataset is a multiple-choice problem with five answer candidates 
(e.g.: The patient, male, 38 years old, suffers from stomach spasmodic pain caused by abdominal cold. 
Which of the following drugs should be chosen?) 
using he structural medical knowledge (i.e. medical knowledge graph) 
and the reference medical plain text (i.e. text snippets retrieved from reference books) .
For more information, please refer to https://tianchi.aliyun.com/dataset/dataDetail?dataId=90134. 
"""

_CITATION = """\
@inproceedings{li2020medical,
  title={Towards Medical Machine Reading Comprehension with Structural Knowledge and Plain Text},
  author={Dongfang, Li and Baotian, Hu and Qingcai, Chen and Weihua, Peng and Anqi, Wang},
  booktitle={EMNLP},
  year={2020}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/NLPEC/train.json"

_HOMEPAGE = "https://tianchi.aliyun.com/dataset/dataDetail?dataId=90134"

class NLPECConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(NLPECConfig, self).__init__(**kwargs)

class NLPEC(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        NLPECConfig(
            name="question_answering_multiple_choice",
            version=datalabs.Version("1.0.0"),
            description="question_answering_multiple_choice",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "question_type": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "question_s": datalabs.Value("string"),
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "options_s": datalabs.features.Sequence(datalabs.Value("string")),
                    "context": datalabs.features.Sequence(datalabs.Value("string")),
                    "context_s": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers": {
                        "text": datalabs.Value("string"),
                        "option_index": datalabs.Value("int32"),
                    }
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.qa_multiple_choice_nlpec)(
                    question_column="question", 
                    context_column="context", 
                    answers_column="answers",
                    options_column = "options",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line)
                question_type, question, question_s = line["questionType"], line["questionText"], line["q_s"]
                option_index = int(line["answer"][0])-1
                options, options_s = line["option"], line["option_s"]
                context, context_s = line["context"], line["context_s"]
                answers = {"text": options[option_index], "option_index": option_index}
                yield id_, {
                    "question_type":question_type,
                    "question": question,
                    "question_s": question_s,
                    "options": options,
                    "options_s": options_s,
                    "context": context,
                    "context_s": context_s,
                    "answers": answers
                }


            