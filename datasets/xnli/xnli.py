# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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
"""XNLI: The Cross-Lingual NLI Corpus."""


import collections
from contextlib import ExitStack
import csv
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@InProceedings{conneau2018xnli,
  author = {Conneau, Alexis
                 and Rinott, Ruty
                 and Lample, Guillaume
                 and Williams, Adina
                 and Bowman, Samuel R.
                 and Schwenk, Holger
                 and Stoyanov, Veselin},
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  location = {Brussels, Belgium},
}"""

_DESCRIPTION = """\
XNLI is a subset of a few thousand examples from MNLI which has been translated
into a 14 different languages (some low-ish resource). As with MNLI, the goal is
to predict textual entailment (does sentence A imply/contradict/neither sentence
B) and is a classification task (given two sentences, predict one of three
labels).
"""

_TRAIN_DATA_URL = "https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip"
_TESTVAL_DATA_URL = "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip"

_LANGUAGES = (
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)


class XnliConfig(datalabs.BuilderConfig):
    """BuilderConfig for XNLI."""

    def __init__(self, language: str, languages=None, **kwargs):
        """BuilderConfig for XNLI.

        Args:
        language: One of ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh, or all_languages
          **kwargs: keyword arguments forwarded to super.
        """
        super(XnliConfig, self).__init__(**kwargs)
        self.language = language
        if language != "all_languages":
            self.languages = [language]
        else:
            self.languages = languages if languages is not None else _LANGUAGES


class Xnli(datalabs.GeneratorBasedBuilder):
    """XNLI: The Cross-Lingual NLI Corpus. Version 1.0."""

    VERSION = datalabs.Version("1.1.0", "")
    BUILDER_CONFIG_CLASS = XnliConfig
    BUILDER_CONFIGS = [
        XnliConfig(
            name=lang,
            language=lang,
            version=datalabs.Version("1.1.0", ""),
            description=f"Plain text import of XNLI for the {lang} language",
        )
        for lang in _LANGUAGES
    ] + [
        XnliConfig(
            name="all_languages",
            language="all_languages",
            version=datalabs.Version("1.1.0", ""),
            description="Plain text import of XNLI for all languages",
        )
    ]

    def _info(self):
        if self.config.language == "all_languages":
            features = datalabs.Features(
                {
                    "premise": datalabs.Translation(
                        languages=_LANGUAGES,
                    ),
                    "hypothesis": datalabs.TranslationVariableLanguages(
                        languages=_LANGUAGES,
                    ),
                    "label": datalabs.ClassLabel(
                        names=["entailment", "neutral", "contradiction"]
                    ),
                }
            )
            languages = _LANGUAGES
        else:
            features = datalabs.Features(
                {
                    "premise": datalabs.Value("string"),
                    "hypothesis": datalabs.Value("string"),
                    "label": datalabs.ClassLabel(
                        names=["entailment", "neutral", "contradiction"]
                    ),
                }
            )
            languages = [self.config.language]
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://www.nyu.edu/projects/bowman/xnli/",
            citation=_CITATION,
            languages=languages,
            task_templates=[
                get_task(TaskType.natural_language_inference)(
                    text1_column="premise",
                    text2_column="hypothesis",
                    label_column="label",
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        dl_dirs = dl_manager.download_and_extract(
            {
                "train_data": _TRAIN_DATA_URL,
                "testval_data": _TESTVAL_DATA_URL,
            }
        )
        train_dir = os.path.join(dl_dirs["train_data"], "XNLI-MT-1.0", "multinli")
        testval_dir = os.path.join(dl_dirs["testval_data"], "XNLI-1.0")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepaths": [
                        os.path.join(train_dir, f"multinli.train.{lang}.tsv")
                        for lang in self.config.languages
                    ],
                    "data_format": "XNLI-MT",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepaths": [os.path.join(testval_dir, "xnli.test.tsv")],
                    "data_format": "XNLI",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": [os.path.join(testval_dir, "xnli.dev.tsv")],
                    "data_format": "XNLI",
                },
            ),
        ]

    def _generate_examples(self, data_format, filepaths):
        """This function returns the examples in the raw (text) form."""

        if self.config.language == "all_languages":
            if data_format == "XNLI-MT":
                with ExitStack() as stack:
                    files = [
                        stack.enter_context(open(filepath, encoding="utf-8"))
                        for filepath in filepaths
                    ]
                    readers = [
                        csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
                        for file in files
                    ]
                    for row_idx, rows in enumerate(zip(*readers)):
                        yield row_idx, {
                            "premise": {
                                lang: row["premise"]
                                for lang, row in zip(self.config.languages, rows)
                            },
                            "hypothesis": {
                                lang: row["hypo"]
                                for lang, row in zip(self.config.languages, rows)
                            },
                            "label": rows[0]["label"].replace(
                                "contradictory", "contradiction"
                            ),
                        }
            else:
                rows_per_pair_id = collections.defaultdict(list)
                for filepath in filepaths:
                    with open(filepath, encoding="utf-8") as f:
                        reader = csv.DictReader(
                            f, delimiter="\t", quoting=csv.QUOTE_NONE
                        )
                        for row in reader:
                            rows_per_pair_id[row["pairID"]].append(row)

                for rows in rows_per_pair_id.values():
                    premise = {row["language"]: row["sentence1"] for row in rows}
                    hypothesis = {row["language"]: row["sentence2"] for row in rows}
                    yield rows[0]["pairID"], {
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": rows[0]["gold_label"],
                    }
        else:
            if data_format == "XNLI-MT":
                for file_idx, filepath in enumerate(filepaths):
                    file = open(filepath, encoding="utf-8")
                    reader = csv.DictReader(
                        file, delimiter="\t", quoting=csv.QUOTE_NONE
                    )
                    for row_idx, row in enumerate(reader):
                        key = str(file_idx) + "_" + str(row_idx)
                        yield key, {
                            "premise": row["premise"],
                            "hypothesis": row["hypo"],
                            "label": row["label"].replace(
                                "contradictory", "contradiction"
                            ),
                        }
            else:
                for filepath in filepaths:
                    with open(filepath, encoding="utf-8") as f:
                        reader = csv.DictReader(
                            f, delimiter="\t", quoting=csv.QUOTE_NONE
                        )
                        for row in reader:
                            if row["language"] == self.config.language:
                                yield row["pairID"], {
                                    "premise": row["sentence1"],
                                    "hypothesis": row["sentence2"],
                                    "label": row["gold_label"],
                                }
