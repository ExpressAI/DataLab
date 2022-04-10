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
The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows 
for a complete analysis of the compositional effects of sentiment in language. The corpus 
is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single 
sentences extracted from movie reviews. It was parsed with the Stanford parser and includes 
a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.
SST-5 is the 5-label version of SST. For more information, please refer to the link 
https://nlp.stanford.edu/sentiment/
"""

_CITATION = """\
@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard  and
      Perelygin, Alex  and
      Wu, Jean  and
      Chuang, Jason  and
      Manning, Christopher D.  and
      Ng, Andrew  and
      Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D13-1170",
    pages = "1631--1642",
}
"""


_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1fMyWb7cNZ6RMvGhlx_VCZF-cd5E_Roo9&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1tAHbYQJdql2R6zqC_3txpmElqg6i41yc&export=download"


class SST5(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=[
                            "very positive",
                            "positive",
                            "neutral",
                            "negative",
                            "very negative",
                        ]
                    ),
                }
            ),
            homepage="https://nlp.stanford.edu/sentiment/",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                TextClassification(
                    text_column="text",
                    label_column="label",
                    task="sentiment-classification",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate SST5 examples."""

        # map the label into textual string
        textualize_label = {
            "0": "very negative",
            "1": "negative",
            "2": "neutral",
            "3": "positive",
            "4": "very positive",
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                text, label = row
                label = textualize_label[label]
                yield id_, {"text": text, "label": label}
