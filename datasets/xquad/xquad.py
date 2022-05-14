# coding=utf-8
# Copyright 2022 The HuggingFace datasets Authors, DataLab Authors and the current dataset script contributor.
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


_CITATION = """\
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
"""

_DESCRIPTION = """\
XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering
performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set
of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into ten languages: Spanish, German,
Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi and Romanian. Consequently, the dataset is entirely parallel
across 12 languages.
"""

# _URL = "https://gidatalabsthub.com/deepmind/xquad/raw/master/"
_URL = "https://github.com/deepmind/xquad/raw/master/"
# https://github.com/deepmind/xquad/raw/master/xquad.ar.json
# https://github.com/deepmind/xquad
_LANG = ["ar", "de", "zh", "vi", "en", "es", "hi", "el", "th", "tr", "ru", "ro"]


class XquadConfig(datalabs.BuilderConfig):

    """BuilderConfig for Xquad"""

    def __init__(self, lang, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XquadConfig, self).__init__(version=datalabs.Version("2.0.0", ""), **kwargs)
        self.lang = lang


class Xquad(datalabs.GeneratorBasedBuilder):
    """TODO(xquad): Short description of my dataset."""

    # TODO(xquad): Set up version.
    VERSION = datalabs.Version("2.0.0")
    # BUILDER_CONFIGS = [XquadConfig(name=f"xquad.{lang}", description=_DESCRIPTION, lang=lang) for lang in _LANG]
    BUILDER_CONFIGS = [XquadConfig(name=f"{lang}", description=_DESCRIPTION, lang=lang) for lang in _LANG]

    def _info(self):
        # TODO(xquad): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "answers":
                        {
                            "text": datalabs.features.Sequence(datalabs.Value("string")),
                            "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                        }
                }
            ),
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/deepmind/xquad",
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
        # TODO(xquad): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = {lang: _URL + f"xquad.{lang}.json" for lang in _LANG}
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": downloaded_files[self.config.lang]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(xquad): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            xquad = json.load(f)
            id_ = 0
            for article in xquad["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]



                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "context": context,
                            "question": question,
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        id_ += 1