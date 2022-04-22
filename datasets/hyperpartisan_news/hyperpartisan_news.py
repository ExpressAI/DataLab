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

import json
import datalabs
from datalabs.tasks import TextClassification

_DESCRIPTION = """\
Hyperpartisan news is news that takes an extreme left-wing or right-wing standpoint. If
one is able to reliably compute this meta information, news articles may be
automatically tagged, this way encouraging or discouraging readers to consume the text.
It is an open question how successfully hyperpartisan news detection can be automated,
and the goal of this SemEval task was to shed light on the state of the art. The SemEval
2019 shared task developed new resources for this purpose, including a manually labeled
dataset with 1,273 articles, and a second dataset with 754,000 articles, labeled via
distant supervision.
https://aclanthology.org/S19-2145/

It was curated into a format for text classification by https://arxiv.org/abs/2004.10964
"""

_CITATION = """\
@inproceedings{kiesel-etal-2019-semeval,
    title = "{S}em{E}val-2019 Task 4: Hyperpartisan News Detection",
    author = "Kiesel, Johannes  and
      Mestre, Maria  and
      Shukla, Rishabh  and
      Vincent, Emmanuel  and
      Adineh, Payam  and
      Corney, David  and
      Stein, Benno  and
      Potthast, Martin",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S19-2145",
    doi = "10.18653/v1/S19-2145",
    pages = "829--839",
}
"""

_TRAIN_DOWNLOAD_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/hyperpartisan_news/train.jsonl"
_DEV_DOWNLOAD_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/hyperpartisan_news/dev.jsonl"
_TEST_DOWNLOAD_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/hyperpartisan_news/test.jsonl"
_CLASS_LABELS = [
    "false",
    "true",
]

class CitationIntent(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=_CLASS_LABELS),
                }
            ),
            homepage="",
            citation=_CITATION,
            languages=["en"],
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        dev_path = dl_manager.download_and_extract(_DEV_DOWNLOAD_URL)
        print(f"dev_path: \t{dev_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""

        with open(filepath, encoding="utf-8") as jsonl_file:
            for id_, line in enumerate(jsonl_file):
                # Necessary to fix poorly formatted file:
                # datas = json.loads('['+line.replace('}{','},{')+']')
                # for data in datas:
                data = json.loads(line)
                yield data["id"], {"text": data["text"], "label": data["label"]}
