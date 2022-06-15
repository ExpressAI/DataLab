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
DuReader version 2.0 contains more than 300K question, 1.4M evidence documents and 660K human generated answers.
This dataset is the preprocessed version of  DuReader_v2.0_raw, 
the preprocessing includes word segmentation, best match paragraph targeting, answer span locating.
For more information, please refer to http://ai.baidu.com/broad/download?dataset=dureader. 
"""

_CITATION = """\
@inproceedings{he-etal-2018-dureader,
    title = "{D}u{R}eader: a {C}hinese Machine Reading Comprehension Dataset from Real-world Applications",
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

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_search/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_search/dev.json"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/dureader_search/test.json"

_HOMEPAGE = "http://ai.baidu.com/broad/download?dataset=dureader"


class DuReaderSearchConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(DuReaderSearchConfig, self).__init__(**kwargs)


class DuReaderSearch(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DuReaderSearchConfig(
            name="question_answering_reading_comprehension",
            version=datalabs.Version("1.0.0"),
            description="question_answering_reading_comprehension",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "documents": datalabs.features.Sequence(
                        {
                            "is_selected": datalabs.Value("string"),
                            "most_related_para": datalabs.Value("int32"),
                            "title": datalabs.Value("string"),
                            "segmented_title": datalabs.features.Sequence(
                                datalabs.Value("string")
                            ),
                            "paragraphs": datalabs.features.Sequence(
                                datalabs.Value("string")
                            ),
                            "segmented_paragraphs": datalabs.features.Sequence(
                                datalabs.features.Sequence(datalabs.Value("string"))
                            ),
                        }
                    ),
                    "answers": datalabs.features.Sequence(datalabs.Value("string")),
                    "segmented_answers": datalabs.features.Sequence(
                        datalabs.features.Sequence(datalabs.Value("string"))
                    ),
                    "fake_answers": datalabs.features.Sequence(
                        datalabs.Value("string")
                    ),
                    "answer_spans": datalabs.features.Sequence(
                        datalabs.features.Sequence(datalabs.Value("int32"))
                    ),
                    "question": datalabs.Value("string"),
                    "segmented_question": datalabs.features.Sequence(
                        datalabs.Value("string")
                    ),
                    "question_type": datalabs.Value("string"),
                    "fact_or_opinion": datalabs.Value("string"),
                    "question_id": datalabs.Value("string"),
                    "match_scores": datalabs.features.Sequence(datalabs.Value("float")),
                    "answer_docs": datalabs.features.Sequence(datalabs.Value("int32")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.qa_extractive)(
                    question_column="question",
                    context_column="documents",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
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
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line)
                documents = line["documents"]
                answers, segmented_answers, fake_answers, answer_spans = (
                    line["answers"],
                    line["segmented_answers"],
                    line["fake_answers"],
                    line["answer_spans"],
                )
                (
                    question,
                    segmented_question,
                    question_type,
                    fact_or_opinion,
                    question_id,
                ) = (
                    line["question"],
                    line["segmented_question"],
                    line["question_type"],
                    line["fact_or_opinion"],
                    line["question_id"],
                )
                match_scores, answer_docs = line["match_scores"], line["answer_docs"]
                yield id_, {
                    "documents": documents,
                    "answers": answers,
                    "segmented_answers": segmented_answers,
                    "fake_answers": fake_answers,
                    "answer_spans": answer_spans,
                    "question": question,
                    "segmented_question": segmented_question,
                    "question_type": question_type,
                    "fact_or_opinion": fact_or_opinion,
                    "question_id": question_id,
                    "match_scores": match_scores,
                    "answer_docs": answer_docs,
                }
