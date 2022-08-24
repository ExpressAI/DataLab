# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
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

import csv

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
SUBJ is a subjectivity dataset where the goal is to classify each instance (snippet) as being 
subjective or objective. For more information, please refer to https://aclanthology.org/P04-1035.pdf
"""

_CITATION = """\
@inproceedings{pang-lee-2004-sentimental,
    title = "A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts",
    author = "Pang, Bo  and
      Lee, Lillian",
    booktitle = "Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics ({ACL}-04)",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    url = "https://aclanthology.org/P04-1035",
    doi = "10.3115/1218955.1218990",
    pages = "271--278",
}
"""


_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1YgJuZWo6F7hxwUmP-iggpYmoS255dalU&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1fG6Xl0IxrVtSY1M2K6gUtj-NqQpEf472&export=download"


class SUBJ(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=["subjective", "objective"]
                    ),
                }
            ),
            homepage="https://www.cs.cornell.edu/people/pabo/movie-review-data/",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.sentiment_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate SUBJ examples."""

        # map the label into textual string
        textualize_label = {"0": "subjective", "1": "objective"}

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                text, label = row
                label = textualize_label[label]
                yield id_, {"text": text, "label": label}
