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
ACL-ARC is a dataset of nearly 2,000 citations annotated for their function.
https://aclanthology.org/Q18-1028/

It was curated into a format for text classification by https://arxiv.org/abs/2004.10964
"""

_CITATION = """\
@article{jurgens-etal-2018-measuring,
    title = "Measuring the Evolution of a Scientific Field through Citation Frames",
    author = "Jurgens, David  and
      Kumar, Srijan  and
      Hoover, Raine  and
      McFarland, Dan  and
      Jurafsky, Dan",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    year = "2018",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q18-1028",
    doi = "10.1162/tacl_a_00028",
    pages = "391--406",
}
"""

_TRAIN_DOWNLOAD_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/citation_intent/train.jsonl"
_DEV_DOWNLOAD_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/citation_intent/dev.jsonl"
_TEST_DOWNLOAD_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/citation_intent/test.jsonl"
_CLASS_LABELS = [
    "Background",
    "CompareOrContrast",
    "Extends",
    "Future",
    "Motivation",
    "Uses",
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
                data = json.loads(line)
                yield id_, {"text": data["text"], "label": data["label"]}
