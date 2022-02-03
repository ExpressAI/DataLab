# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors and DataLab Authors..
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
"""HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering."""


import json
import textwrap

# import datasets
import datalabs
from datalabs.tasks import QuestionAnsweringExtractive,QuestionAnsweringHotpot


_CITATION = """
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}
"""

_DESCRIPTION = """\
HotpotQA is a new dataset with 113k Wikipedia-based question-answer pairs with four key features:
(1) the questions require finding and reasoning over multiple supporting documents to answer;
(2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas;
(3) we provide sentence-level supporting facts required for reasoning, allowingQA systems to reason with strong supervisionand explain the predictions;
(4) we offer a new type of factoid comparison questions to testQA systems’ ability to extract relevant facts and perform necessary comparison.
"""

_URL_BASE = "http://curtis.ml.cmu.edu/datasets/hotpot/"


class HotpotQA(datalabs.GeneratorBasedBuilder):
    """HotpotQA is a Dataset for Diverse, Explainable Multi-hop Question Answering."""

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="distractor",
            version=datalabs.Version("1.0.0"),
            description=textwrap.dedent(
                """
                     In the distractor setting, a question-answering system reads 10 paragraphs to provide an answer to a question.
                     They must also justify these answers with supporting facts. This setting challenges the model to find the true
                     supporting facts in the presence of noise, for each example we employ bigram tf-idf (Chen et al., 2017) to retrieve
                     8 paragraphs from Wikipedia as distractors, using the question as the query. We mix them with the 2 gold paragraphs
                     (the ones used to collect the question and answer) to construct the distractor setting.
                """
            ),
        ),
        datalabs.BuilderConfig(
            name="fullwiki",
            version=datalabs.Version("1.0.0"),
            description=textwrap.dedent(
                """
                     In the fullwiki setting, a question-answering system must find the answer to a question in the scope of the
                     entire Wikipedia. We fully test the model’s ability to locate relevant facts as well as reasoning about them
                     by requiring it to answer the question given the first paragraphs of all Wikipedia articles without the gold
                     paragraphs specified. This full wiki setting truly tests the performance of the systems’ ability at multi-hop
                     reasoning in the wild.
                """
            ),
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    # "answer": datalabs.Value("string"),
                    "type": datalabs.Value("string"),
                    "level": datalabs.Value("string"),
                    "supporting_facts":
                        {
                            "title": datalabs.features.Sequence(datalabs.Value("string")),
                            "sent_id": datalabs.features.Sequence(datalabs.Value("int32")),
                        },
                    "context":
                        {
                            "title": datalabs.features.Sequence(datalabs.Value("string")),
                            "sentences": datalabs.features.Sequence(datalabs.Value("string")),
                        },
                    "answers":
                        {
                            "text": datalabs.features.Sequence(datalabs.Value("string")),
                            "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                        }
                    # "supporting_facts": datalabs.features.Sequence(
                    #     {
                    #         "title": datalabs.Value("string"),
                    #         "sent_id": datalabs.Value("int32"),
                    #     }
                    # ),
                    # "context": datalabs.features.Sequence(
                    #     {
                    #         "title": datalabs.Value("string"),
                    #         "sentences": datalabs.features.Sequence(datalabs.Value("string")),
                    #     }
                    # ),

                }
            ),
            supervised_keys=None,
            homepage="https://hotpotqa.github.io/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringHotpot(
                    question_column="question", context_column="context", answers_column="answers", supporting_column="supporting_facts"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        paths = {
            datalabs.Split.TRAIN: _URL_BASE + "hotpot_train_v1.1.json",
            datalabs.Split.VALIDATION: _URL_BASE + "hotpot_dev_" + self.config.name + "_v1.json",
        }
        if self.config.name == "fullwiki":
            paths[datalabs.Split.TEST] = _URL_BASE + "hotpot_test_fullwiki_v1.json"

        files = dl_manager.download(paths)

        split_generators = []
        for split in files:
            split_generators.append(datalabs.SplitGenerator(name=split, gen_kwargs={"data_file": files[split]}))

        return split_generators

    def _generate_examples(self, data_file):
        """This function returns the examples."""
        data = json.load(open(data_file))
        for idx, example in enumerate(data):

            # Test set has missing keys
            for k in ["answer", "type", "level"]:
                if k not in example.keys():
                    example[k] = None

            if "supporting_facts" not in example.keys():
                example["supporting_facts"] = []

            supporting_titles = [x[0] for x in example["supporting_facts"]]
            supporting_sent_ids = [x[1] for x in example["supporting_facts"]]

            context_titles = [x[0] for x in example["context"]]
            context_sentences = [x[1] for x in example["context"]]


            yield idx, {
                "id": example["_id"],
                "question": example["question"],
                # "answer": example["answer"],
                "type": example["type"],
                "level": example["level"],
                "supporting_facts": {
                    "title": supporting_titles,
                    "sent_id": supporting_sent_ids,
                },
                "context": {
                    "title": context_titles,
                    "sentences": context_sentences,
                },
                "answers": {
                    "answer_start": [-1] * 1,
                    "text": [example["answer"]],
                },
                # "supporting_facts": [{"title": f[0], "sent_id": f[1]} for f in example["supporting_facts"]],
                # "context": [{"title": f[0], "sentences": f[1]} for f in example["context"]],
            }