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
Twitter is an aspect-based sentiment classification dataset, processed from tweets. Each tweet is labeled as positive,
neutral or negative w.r.t. a specific aspect. For more information, please refer to https://aclanthology.org/P14-2009.pdf
"""

_CITATION = """\
@inproceedings{dong-etal-2014-adaptive,
    title = "Adaptive Recursive Neural Network for Target-dependent {T}witter Sentiment Classification",
    author = "Dong, Li  and
      Wei, Furu  and
      Tan, Chuanqi  and
      Tang, Duyu  and
      Zhou, Ming  and
      Xu, Ke",
    booktitle = "Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jun,
    year = "2014",
    address = "Baltimore, Maryland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P14-2009",
    doi = "10.3115/v1/P14-2009",
    pages = "49--54",
}
"""

_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1ICFDl-MH5eyumIVd_-KsRzg5S0qqEcho&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1ZbFD6ePUPR_y2CUUrTVioxWB6kv5xrKf&export=download"


class Twitter(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "aspect": datalabs.Value("string"),
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative", "neutral"]),
                }
            ),
            homepage="https://aclanthology.org/P14-2009.pdf",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.aspect_based_sentiment_classification)(
                    span_column="aspect",
                    text_column="text",
                    label_column="label"
                )]
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate Twitter examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for id_, row in enumerate(csv_reader):
                aspect, text, label = row
                yield id_, {"aspect": aspect, "text": text, "label": label}
