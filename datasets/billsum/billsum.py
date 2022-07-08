# coding=utf-8
# Copyright 2020 The TensorFlow datasets Authors and the HuggingFace datasets and DataLab Authors.
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
"""BillSum Dataset."""


import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """
@inproceedings{kornilova-eidelman-2019-billsum,
    title = "{B}ill{S}um: A Corpus for Automatic Summarization of {US} Legislation",
    author = "Kornilova, Anastassia  and
      Eidelman, Vladimir",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5406",
    doi = "10.18653/v1/D19-5406",
    pages = "48--56",
    abstract = "Automatic summarization methods have been studied on a variety of domains, including news and scientific articles. Yet, legislation has not previously been considered for this task, despite US Congress and state governments releasing tens of thousands of bills every year. In this paper, we introduce BillSum, the first dataset for summarization of US Congressional and California state bills. We explain the properties of the dataset that make it more challenging to process than other domains. Then, we benchmark extractive methods that consider neural sentence representations and traditional contextual features. Finally, we demonstrate that models built on Congressional bills can be used to summarize California billa, thus, showing that methods developed on this dataset can transfer to states without human-written summaries.",
}
"""

_DESCRIPTION = """
BillSum, summarization of US Congressional and California state bills.

There are several features:
  - text: bill text.
  - summary: summary of the bills.
  - title: title of the bills.
features for us bills. ca bills does not have.
  - text_len: number of chars in text.
  - sum_len: number of chars in summary.
"""

_URL = "https://drive.google.com/uc?export=download&id=1g89WgFHMRbr4QrvA0ngh26PY081Nv3lx&confirm=yes"

_DOCUMENT = "text"
_SUMMARY = "summary"


class Billsum(datalabs.GeneratorBasedBuilder):
    """BillSum Dataset."""

    # 2.0.0 data source updated to filter near duplicates.
    # 3.0.0  none of the test examples are 'near duplicates' of an example in the
    #   train set AND they dont have the same title, regardless of similarity.
    VERSION = datalabs.Version("3.0.0")

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _DOCUMENT: datalabs.Value("string"),
                    _SUMMARY: datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                }
            ),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            homepage="https://github.com/FiscalNote/BillSum",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_DOCUMENT, reference_column=_SUMMARY
                ),
            ],
            languages=["en"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_path = dl_manager.download_and_extract(_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(dl_path, "us_train_data_final_OFFICIAL.jsonl"),
                    "key": "bill_id",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(dl_path, "us_test_data_final_OFFICIAL.jsonl"),
                    "key": "bill_id",
                },
            ),
            datalabs.SplitGenerator(
                name="ca_test",
                gen_kwargs={
                    "path": os.path.join(dl_path, "ca_test_data_final_OFFICIAL.jsonl"),
                    "key": "external_id",
                },
            ),
        ]

    def _generate_examples(self, path=None, key=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                # in us bills, json has fields:
                #   text, summary, title, bill_id, text_len, sum_len
                # in ca bills, json has fields:
                #   text, summary, title, external_id
                d = json.loads(line)
                yield d[key], {k: d[k] for k in [_DOCUMENT, _SUMMARY, "title"]}
