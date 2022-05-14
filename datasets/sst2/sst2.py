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
The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows 
for a complete analysis of the compositional effects of sentiment in language. The corpus 
is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single 
sentences extracted from movie reviews. It was parsed with the Stanford parser and includes 
a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.
SST-2 is the binary version of SST. For more information, please refer to the link 
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

_TRAIN_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/sst2/train-SST2.tsv"
_VALIDATION_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/sst2/valid-SST2.tsv"
_TEST_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/sst2/test-SST2.tsv"


class SST2(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative"]),
                }
            ),
            homepage="https://nlp.stanford.edu/sentiment/",
            citation=_CITATION,
            languages=["en"],
            task_templates=[get_task(TaskType.sentiment_classification)(
                text_column="text",
                label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        print(f"validation_path: \t{validation_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        """Generate SST2 examples."""

        # map the label into textual string
        textualize_label = {
            "1": "positive",
            "0": "negative"
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for id_, row in enumerate(csv_reader):
                text, label = row
                label = textualize_label[label]
                yield id_, {"text": text, "label": label}
