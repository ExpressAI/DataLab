# coding=utf-8
# Copyright 2022 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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

# Lint as: python3



import json

import datalabs
from datalabs import get_task, TaskType

logger = datalabs.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{zhu-etal-2021-tat,
    title = "{TAT}-{QA}: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance",
    author = "Zhu, Fengbin  and
      Lei, Wenqiang  and
      Huang, Youcheng  and
      Wang, Chao  and
      Zhang, Shuo  and
      Lv, Jiancheng  and
      Feng, Fuli  and
      Chua, Tat-Seng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.254",
    doi = "10.18653/v1/2021.acl-long.254",
    pages = "3277--3287"
}
"""

_DESCRIPTION = """\
This is a new large-scale QA dataset
containing both Tabular And Textual data,
named TAT-QA, where numerical reasoning
is usually required to infer the answer, such
as addition, subtraction, multiplication, division, counting, comparison/sorting, and their
compositions
"""

_URL = "https://datalab-hub.s3.amazonaws.com/tat-qa/"
_URLS = {
    "test": _URL + "test_data.json",
}



class SquadConfig(datalabs.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadConfig, self).__init__(**kwargs)


class TATQA(datalabs.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        SquadConfig(
            name="plain_text",
            version=datalabs.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "table_id":datalabs.Value("string"),
                    "table":datalabs.features.Sequence(datalabs.features.Sequence(
                                                                datalabs.Value("string"))),
                    "paragraphs":datalabs.features.Sequence({"uid":datalabs.Value("string"),
                                                            "order":datalabs.Value("int32"),
                                                            "text":datalabs.Value("string")}),
                    "q_id": datalabs.Value("string"),
                    "q_order": datalabs.Value("int32"),
                    "question": datalabs.Value("string"),
                    "scale": datalabs.Value("string"),
                    "derivation": datalabs.Value("string"),
                    "answer":datalabs.features.Sequence(datalabs.Value("string")),
                    "answer_type": datalabs.Value("string"),
                    "answer_from": datalabs.Value("string"),
                    "facts": datalabs.features.Sequence(datalabs.Value("string")),
                    # "mapping":datalabs.features.dict("table"),
                    # "paragraph": datalabs.features.Sequence({
                    #     "id": datalabs.Value("string"),
                    #     "content": datalabs.features.Sequence(
                    #         datalabs.features.Sequence(
                    #             datalabs.Value("string"))),
                    # }),
                    "counterfactual": datalabs.Value("int32"),

                    # "answers": datalabs.features.Sequence(
                    #     {
                    #         "text": datalabs.Value("string"),
                    #         "answer_start": datalabs.Value("int32"),
                    #     }
                    # ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            task_templates=get_task(TaskType.qa_table_text_hybrid)(
                question_column="question",
                context_column= "paragraphs",
                table_column= "table",
                answer_column = "answer",
                answer_type_column = "answer_type",
                answer_scale_column = "scale",

            ),
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data_all = json.load(f)
            for article in data_all:
                table = article["table"]["table"]
                table_id = article["table"]["uid"]
                paragraphs = article["paragraphs"]
                questions = article["questions"]
                for qa in questions:
                    q_id = qa["uid"]
                    q_order = qa["order"]
                    question = qa["question"]
                    scale = qa["scale"]
                    derivation = qa["derivation"]

                    answer = qa["answer"] if isinstance(qa["answer"], list) else [qa["answer"]]
                    answer_type = qa["answer_type"]
                    answer_from = qa["answer_from"]
                    facts = qa["facts"]
                    counterfactual = qa["counterfactual"]
                    # paragraph = []
                    # for k, v in qa["paragraph"].items():
                    #     paragraph.append({
                    #         "id":k,
                    #         "content":v
                    #     })

                    yield key, {
                        "table_id": table_id,
                        "table": table,
                        "paragraphs": paragraphs,
                        "q_id": q_id,
                        "q_order":q_order,
                        "question":question,
                        "scale":scale,
                        "derivation": derivation,
                        "answer": answer,
                        "answer_type": answer_type,
                        "answer_from": answer_from,
                        "facts": facts,
                        "counterfactual": counterfactual,

                    }
                    key += 1
