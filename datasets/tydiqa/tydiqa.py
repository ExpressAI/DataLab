# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and DataLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import textwrap
import os
import datalabs
from datalabs import get_task, TaskType


# TODO(tydiqa): BibTeX citation
_CITATION = """\
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
"""

# TODO(tydiqa):
_DESCRIPTION = """\
TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs.
The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language
expresses -- such that we expect models performing well on this set to generalize across a large number of the languages
in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic
information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but
don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without
the use of translation (unlike MLQA and XQuAD).
"""
LANG_URLS = {
    "ar": "https://drive.google.com/uc?export=download&id=13iwx6_erfmLylo7bzjVjsXHmQYDxBetV",
    "bn": "https://drive.google.com/uc?export=download&id=1798tXN8s8SRS16s78VI3eUrtGbnQjkGG",
    "en": "https://drive.google.com/uc?export=download&id=1EnYyyNPRK8x9Urwh87369BlqQCXj4Eo3",
    "fi": "https://drive.google.com/uc?export=download&id=1fOU65kZzpHgJ2ylNlHf4JoKcuH8Mf_Pz",
    "id": "https://drive.google.com/uc?export=download&id=1HDa1u6kfIuEO1SUyGTkVi7XwuBrNMhor",
    "ko": "https://drive.google.com/uc?export=download&id=1qyq-3oX2g9qsZwpf2eKasJWnkWw0ew5j",
    "ru": "https://drive.google.com/uc?export=download&id=1c9dCxmTXA35H1Iq5Gu4h-TxGT1WGaFV7",
    "sw": "https://drive.google.com/uc?export=download&id=1Yab535N9Sbp9doaoSpZwGNYo4UDiPAZr",
    "te": "https://drive.google.com/uc?export=download&id=10ccDq-3JZqD-TbnUk7lAKuxMlb4fzs4m",
}
class TydiqaConfig(datalabs.BuilderConfig):

    """BuilderConfig for Tydiqa"""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(TydiqaConfig, self).__init__(version=datalabs.Version("2.0.0", ""), **kwargs)


class Tydiqa(datalabs.GeneratorBasedBuilder):
    """TODO(tydiqa): Short description of my dataset."""

    # TODO(tydiqa): Set up version.
    VERSION = datalabs.Version("2.0.0")
    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="{}".format(lang),
            version=datalabs.Version("2.0.0")
        )
        for lang in list(LANG_URLS.keys())
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
                    }
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/google-research-datasets/tydiqa",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.qa_extractive)(
                    question_column="question",
                    context_column="context",
                    answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang = str(self.config.name)
        url =LANG_URLS[lang]
        # url = _URL.format(lang, self.VERSION.version_str[:-2])
        data_dir = dl_manager.download_and_extract(url)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir,lang+ "/train-" + lang + ".json"),
                    # "filepath": os.path.join(data_dir,  "train-en.json" + lang + "_train.jsonl"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang+ "/test-" + lang + ".json"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang+ "/dev-" + lang + ".json"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        yield key, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1