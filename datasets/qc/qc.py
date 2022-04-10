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
from datalabs import Dataset
from datalabs.tasks import TextClassification

_DESCRIPTION = """\
The QC question classification dataset involves six different question types.
For more information, please refer to https://aclanthology.org/C02-1150.pdf
"""

_CITATION = """\
@inproceedings{li-roth-2002-learning,
    title = "Learning Question Classifiers",
    author = "Li, Xin  and
      Roth, Dan",
    booktitle = "{COLING} 2002: The 19th International Conference on Computational Linguistics",
    year = "2002",
    url = "https://aclanthology.org/C02-1150",
}
"""

_TRAIN_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/qc/train-QC.tsv"
_TEST_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/qc/test-QC.tsv"


class QC(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=[
                            "abbreviation",
                            "entity",
                            "description",
                            "human",
                            "location",
                            "numeric value",
                        ]
                    ),
                }
            ),
            homepage="https://aclanthology.org/C02-1150.pdf",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                TextClassification(
                    text_column="text",
                    label_column="label",
                    task="question-classification",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate QC examples."""

        # map the label into textual string
        textualize_label = {
            "ABBR": "abbreviation",
            "DESC": "description",
            "ENTY": "entity",
            "HUM": "human",
            "LOC": "location",
            "NUM": "numeric value",
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                label = row[-1]
                text = " ".join(row[:-1])
                label = textualize_label[label]
                yield id_, {"text": text, "label": label}
