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
"""SQUAD: The Stanford Question Answering Dataset."""


import json
import os
import textwrap

import datalabs
from datalabs import get_task, TaskType

logger = datalabs.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{faisal-etal-21-sdqa,
 title = {{SD-QA}: {S}poken {D}ialectal {Q}uestion {A}nswering for the {R}eal {W}orld},
  author = {Faisal, Fahim and Keshava, Sharlina and ibn Alam, Md Mahfuz and Anastasopoulos, Antonios},
  url={https://arxiv.org/abs/2109.12072},
  year = {2021},
  booktitle = {Findings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP Findings)},
  publisher = {Association for Computational Linguistics},
  month = {November},
}
"""

_DESCRIPTION = """\
SDQA: : Spoken Dialectal Question Answering for the Real World
is a multi-dialect, spoken QA benchmark on five languages (Arabic, Bengali, English, Korean, Kiswahili) 
with more than 68k audio prompts in 22 dialects from 245 speakers. Google Speech API with regional units 
(eg. en-US, sw-TZ) are used to perform speech to text conversion of the questions. These questions are
replaced in place of the original TyDiQA gold questions to prepare dialectal development and test set. As a benchmark 
train set, a discarded version of original TyDiQA training set is used.

"""

# LANG_URLS = {
#     "ar": "https://drive.google.com/uc?export=download&id=1tUsnP9xPFEsLjShtnsN1osgK84jPc3ED",
#     "bn": "https://drive.google.com/uc?export=download&id=1qqOiZy1lz0pIzw_rgrVHwx2XJveIJzGb",
#     "en": "https://drive.google.com/uc?export=download&id=1WGfgc7_0DCkxBsheaxfke4TK_j1G5OFT",
#     "fi": "https://drive.google.com/uc?export=download&id=1-pbAbt-v2QHWDj2dnbKcOSNbJHLEiGpd",
#     "id": "https://drive.google.com/uc?export=download&id=1QuMrKvWJ74sl-MrTAo6Mz-_bs953YZM6",
#     "ko": "https://drive.google.com/uc?export=download&id=1PzbuXVeCuYU3CBPLgqjM4GuRvY1fCvVl",
#     "ru": "https://drive.google.com/uc?export=download&id=1NIlGs1o_1AYPlyIIcBnXJHIRh5Zi9nU1",
#     "sw": "https://drive.google.com/uc?export=download&id=1JkXf6rHyFRWyQGTCdwIxPVuRoY2dmDwd",
#     "te": "https://drive.google.com/uc?export=download&id=1NyliC5GairK8TvxrQOvH-ktNgsKbA5Fz",
# }
# DATA_URL="https://drive.google.com/uc?export=download&id=1_kGzqqiJJLGpKN6pfNzguyzG_vU8figf"
# TRAIN_LANGS = {'en':'english','bn':'bengali','sw':'swahili','ko':'korean','ar':'arabic','fi':'finnish','te':'telugu','id':'indonesian','ru':'russian'}
# TRAIN_LANGS = {'en':'english','bn':'bengali'}
# SDQA_LANGS = {'en':'english','bn':'bengali','sw':'swahili','ko':'korean','ar':'arabic'}
# DIALECTS = {
#     'english':['nga','nzl','usa','ind_s','ind_n','irl','kenya','gbr','zaf','aus','phl'],
#     'arabic':['bhr','dza','egy','jor','mar','sau','tun'],
#     'swahili':['kenya','tanzania'],
#     'bengali':['dhaka','ind'],
#     'korean':['korn','kors']
# }


_URLS = {
    "train": "https://drive.google.com/uc?export=download&id=16M6bglY6ttiMcYximOkQ5jKX12KyRYxL",
    "dev": "https://drive.google.com/uc?export=download&id=18SNCH2uQ7PBKAeOWgyRlWS_QmbJi3oDj",
    "test": "https://drive.google.com/uc?export=download&id=18iyLut6C6SVu3cZFlocBGtarFgQ03qzn",
}


class SdqaConfig(datalabs.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUADV2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SdqaConfig, self).__init__(**kwargs)


class Sdqa(datalabs.GeneratorBasedBuilder):
    """TODO(squad_v2): Short description of my dataset."""

    # TODO(squad_v2): Set up version.
    BUILDER_CONFIGS = [
        SdqaConfig(
            name="sdqa", version=datalabs.Version("1.0.0"), description="SDQA 1.0.0"
        ),
    ]

    def _info(self):
        # TODO(squad_v2): Specifies the datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datalab page.
            description=_DESCRIPTION,
            # datalab.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "answers": {
                        "text": datalabs.features.Sequence(datalabs.Value("string")),
                        "answer_start": datalabs.features.Sequence(
                            datalabs.Value("int32")
                        ),
                    }
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://nlp.cs.gmu.edu/publication/faisal-etal-21-sdqa/",
            citation=_CITATION,
            languages=["en", "ar", "bn", "sw", "ko"],
            task_templates=[
                get_task(TaskType.qa_extractive)(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(squad_v2): Downloads the data and defines the splits
        # dl_manager is a datalab.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(squad_v2): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for example in squad["data"]:
                title = example.get("title", "")
                for paragraph in example["paragraphs"]:
                    context = paragraph[
                        "context"
                    ]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        id_ = qa["id"]

                        answer_starts = [
                            answer["answer_start"] for answer in qa["answers"]
                        ]
                        answers = [answer["text"] for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
